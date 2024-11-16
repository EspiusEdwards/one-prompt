
import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)


import torch


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_one(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    torch.autograd.set_detect_anomaly(True)

    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:

        for pack in train_loader:
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            b_size = imgs.size(0)  # Batch size

            # Generate prompts for each image in the batch
            if 'pt' not in pack:
                # Generate prompts for each image
                imgs, pts, masks = generate_click_prompt(imgs, masks)
                point_labels = torch.ones(b_size, dtype=torch.int, device=GPUdevice)
            else:
                pts = pack['pt']
                point_labels = pack['p_label']
            
            # Prepare the prompts
            coords_torch = torch.as_tensor(pts, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            # Reshape to [batch_size, num_points, 2] and [batch_size, num_points]
            coords_torch = coords_torch[:, None, :]  # [batch_size, 1, 2]
            labels_torch = labels_torch[:, None]     # [batch_size, 1]
            pt = (coords_torch, labels_torch)

            # Now, process tmp_img and tmp_mask
            if ind == 0:
                tmp_img = imgs[0].unsqueeze(0)  # Shape [1, C, H, W]
                tmp_mask = masks[0].unsqueeze(0)
                # Generate prompt for tmp_img
                tmp_img, tmp_pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                tmp_point_labels = torch.ones(1, dtype=torch.int, device=GPUdevice)
                tmp_coords_torch = torch.as_tensor(tmp_pt, dtype=torch.float, device=GPUdevice)
                tmp_labels_torch = torch.as_tensor(tmp_point_labels, dtype=torch.int, device=GPUdevice)
                tmp_coords_torch = tmp_coords_torch[None, :, :]  # [1, 1, 2]
                tmp_labels_torch = tmp_labels_torch[None]        # [1, 1]
                tmp_pt = (tmp_coords_torch, tmp_labels_torch)
                # Get embedding for tmp_img
                with torch.no_grad():
                    tmp_imge, tmp_skips = net.image_encoder(tmp_img.clone())
            else:
                # Use the same tmp_imge and tmp_skips from the first iteration
                pass

            # Process images
            imge, skips = net.image_encoder(imgs)

            # Encode prompts
            p1, p2, se, de = net.prompt_encoder(
                points=pt,
                boxes=None,
                doodles=None,
                masks=None,
            )
            tmp_p1, tmp_p2, tmp_se, tmp_de = net.prompt_encoder(
                points=tmp_pt,
                boxes=None,
                doodles=None,
                masks=None,
            )

            # Expand tmp_imge and tmp_skips to match batch size
            tmp_imge_expanded = tmp_imge.expand(b_size, -1, -1, -1)
            tmp_skips_expanded = [s.expand(b_size, -1, -1, -1) for s in tmp_skips]

            # Expand tmp_p1 and tmp_p2 if necessary (depends on your model)
            # For now, assume they are not needed in mask_decoder
            
            skips = [s.expand(b_size, -1, -1, -1).clone() for s in skips]
            
            # print the skips shape from the comprehension
            # for s in skips:
            #     print(f"skip shape: {s.shape}")
            
            # tmp_skips_expanded = [s.expand(b_size, -1, -1, -1) for s in tmp_skips]
            
            # for s in tmp_skips_expanded:
            #     print(f"tmp_skip shape: {s.shape}")

            pred, _ = net.mask_decoder(
                skips_raw=skips,
                skips_tmp=tmp_skips_expanded,
                raw_emb=imge,
                tmp_emb=tmp_imge_expanded,
                pt1=p1,
                pt2=p2,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )
            if pred.shape != masks.shape:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            

            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat += na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'), reverse=False)

            ind += 1
            pbar.update()
        
    return loss

def validation_one(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batches
    ave_res, mix_res = (0, 0, 0, 0), (0, 0, 0, 0)
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        
        for ind, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            b_size, c, w, h = imgs.size()

            # Generate or get pt (prompt) coordinates and labels, based on train_one
            if 'pt' not in pack:
                # Generate prompts for each image if 'pt' is not present
                imgs, pts, masks = generate_click_prompt(imgs, masks)
                point_labels = torch.ones(b_size, dtype=torch.int, device=GPUdevice)
            else:
                pts = pack['pt']
                point_labels = pack['p_label']
            
            # Prepare the prompts
            coords_torch = torch.as_tensor(pts, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            # Reshape to [batch_size, num_points, 2] and [batch_size, num_points]
            coords_torch = coords_torch[:, None, :]  # [batch_size, 1, 2]
            labels_torch = labels_torch[:, None]     # [batch_size, 1]
            pt = (coords_torch, labels_torch)

            # Initialize tmp_img and tmp_mask for reference, only for the first iteration
            if ind == 0:
                tmp_img = imgs[0].unsqueeze(0)  # Shape [1, C, H, W]
                tmp_mask = masks[0].unsqueeze(0)
                # Generate prompt for tmp_img
                tmp_img, tmp_pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                tmp_point_labels = torch.ones(1, dtype=torch.int, device=GPUdevice)
                tmp_coords_torch = torch.as_tensor(tmp_pt, dtype=torch.float, device=GPUdevice)
                tmp_labels_torch = torch.as_tensor(tmp_point_labels, dtype=torch.int, device=GPUdevice)
                tmp_coords_torch = tmp_coords_torch[None, :, :]  # [1, 1, 2]
                tmp_labels_torch = tmp_labels_torch[None]        # [1, 1]
                tmp_pt = (tmp_coords_torch, tmp_labels_torch)
                
                # Get embedding for tmp_img (without gradients)
                with torch.no_grad():
                    tmp_imge, tmp_skips = net.image_encoder(tmp_img)

            # Expand tmp_imge and tmp_skips to match the batch size
            tmp_imge_expanded = tmp_imge.expand(b_size, -1, -1, -1)
            tmp_skips_expanded = [s.expand(b_size, -1, -1, -1) for s in tmp_skips]

            # Process images
            with torch.no_grad():
                imge, skips = net.image_encoder(imgs)

            # Encode prompts
            p1, p2, se, de = net.prompt_encoder(
                points=pt,
                boxes=None,
                doodles=None,
                masks=None,
            )
            tmp_p1, tmp_p2, tmp_se, tmp_de = net.prompt_encoder(
                points=tmp_pt,
                boxes=None,
                doodles=None,
                masks=None,
            )
            # Wrap in torch.no_grad() to avoid gradient computation
            with torch.no_grad():
                # Pass expanded embeddings into mask_decoder
                pred, _ = net.mask_decoder(
                    skips_raw=skips,
                    skips_tmp=tmp_skips_expanded,
                    raw_emb=imge,
                    tmp_emb=tmp_imge_expanded,
                    pt1=p1,
                    pt2=p2,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )

            # Resize pred if necessary to match masks
            if pred.shape != masks.shape:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute loss
            tot += lossfunc(pred, masks)
            # Visualization for debugging (every `vis` steps)
            # Set default value for vis if None
            vis = args.vis if args.vis is not None else 50  # Set default to 50 or any other value
            if ind % vis == 0:
                namecat = 'Test'
                for na in name:
                    img_name = na.split('/')[-1].split('.')[0]
                    namecat = namecat + img_name + '+'
                vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)

            # Evaluate segmentation
            temp = eval_seg(pred, masks, threshold)
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot / n_val, tuple([a / n_val for a in mix_res])




# def train_one(args, net: nn.Module, optimizer, train_loader,
#           epoch, writer, schedulers=None, vis = 50):
#     hard = 0
#     epoch_loss = 0
#     ind = 0
#     # train mode
#     net.train()
#     optimizer.zero_grad()

#     epoch_loss = 0
#     GPUdevice = torch.device('cuda:' + str(args.gpu_device))

#     if args.thd:
#         lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#     else:
#         lossfunc = criterion_G

#     with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:

#         for pack in train_loader:
#             if ind == 0:
#                 tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
#                 tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
#                 if 'pt' not in pack:
#                     tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
#                 else:
#                     pt = pack['pt']
#                     point_labels = pack['p_label']
                
#                 if point_labels[0] != -1:
#                     point_coords = pt
#                     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
#                     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
#                     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#                     pt = (coords_torch, labels_torch)

#             imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
#             masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

#             name = pack['image_meta_dict']['filename_or_obj']

#             if args.thd:
#                 pt = rearrange(pt, 'b n d -> (b d) n')
#                 imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
#                 masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

#                 imgs = imgs.repeat(1,3,1,1)
#                 point_labels = torch.ones(imgs.size(0))

#                 imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
#                 masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            
#             showp = pt

#             mask_type = torch.float32
#             ind += 1
#             b_size,c,w,h = imgs.size()
#             longsize = w if w >=h else h

#             '''init'''
#             if hard:
#                 true_mask_ave = (true_mask_ave > 0.5).float()
#             imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
#             with torch.no_grad():
#                 imge, skips= net.image_encoder(imgs)
#                 timge, tskips = net.image_encoder(tmp_img)

#                     # imge= net.image_encoder(imgs)
#             p1, p2, se, de = net.prompt_encoder(
#                     points=pt,
#                     boxes=None,
#                     doodles= None,
#                     masks=None,
#                 )
#             print(f"imge shape: {imge.shape}")
#             print(f"timge shape: {timge.shape}")
#             print(f"p1 shape: {p1.shape}")
#             print(f"p2 shape: {p2.shape}")
#             pred, _ = net.mask_decoder(
#                 skips_raw = skips,
#                 skips_tmp = tskips,
#                 raw_emb = imge,
#                 tmp_emb = timge,
#                 pt1 = p1,
#                 pt2 = p2,
#                 image_pe=net.prompt_encoder.get_dense_pe(), 
#                 sparse_prompt_embeddings=se,
#                 dense_prompt_embeddings=de, 
#                 multimask_output=False,
#               )

#             loss = lossfunc(pred, masks)

#             pbar.set_postfix(**{'loss (batch)': loss.item()})
#             epoch_loss += loss.item()
#             loss.backward()

#             # nn.utils.clip_grad_value_(net.parameters(), 0.1)
#             optimizer.step()
#             optimizer.zero_grad()

#             '''vis images'''
#             if vis:
#                 if ind % vis == 0:
#                     namecat = 'Train'
#                     for na in name:
#                         namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
#                     vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)

#             pbar.update()
        


#     return loss

# def validation_one(args, val_loader, epoch, net: nn.Module, clean_dir=True):
#      # eval mode
#     net.eval()

#     mask_type = torch.float32
#     n_val = len(val_loader)  # the number of batch
#     ave_res, mix_res = (0,0,0,0), (0,0,0,0)
#     rater_res = [(0,0,0,0) for _ in range(6)]
#     tot = 0
#     hard = 0
#     threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
#     GPUdevice = torch.device('cuda:' + str(args.gpu_device))
#     device = GPUdevice

#     if args.thd:
#         lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#     else:
#         lossfunc = criterion_G

#     with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        
#         for ind, pack in enumerate(val_loader):
#             if ind == 0:
#                 tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
#                 tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
#                 if 'pt' not in pack:
#                     tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
#                 else:
#                     pt = pack['pt']
#                     point_labels = pack['p_label']
                
#                 if point_labels[0] != -1:
#                     # point_coords = onetrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
#                     point_coords = pt
#                     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
#                     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
#                     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#                     pt = (coords_torch, labels_torch)


#             imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
#             masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

#             name = pack['image_meta_dict']['filename_or_obj']
            
#             showp = pt

#             mask_type = torch.float32
#             ind += 1
#             b_size,c,w,h = imgs.size()
#             longsize = w if w >=h else h

#             '''init'''
#             if hard:
#                 true_mask_ave = (true_mask_ave > 0.5).float()
#                 #true_mask_ave = cons_tensor(true_mask_ave)
#             imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
#             '''test'''
#             with torch.no_grad():
#                 imge, skips= net.image_encoder(imgs)
#                 timge, tskips = net.image_encoder(tmp_img)

#                 p1, p2, se, de = net.prompt_encoder(
#                         points=pt,
#                         boxes=None,
#                         doodles= None,
#                         masks=None,
#                     )
#                 pred, _ = net.mask_decoder(
#                     skips_raw = skips,
#                     skips_tmp = tskips,
#                     raw_emb = imge,
#                     tmp_emb = timge,
#                     pt1 = p1,
#                     pt2 = p2,
#                     image_pe=net.prompt_encoder.get_dense_pe(), 
#                     sparse_prompt_embeddings=se,
#                     dense_prompt_embeddings=de, 
#                     multimask_output=False,
#                 )
            
#                 tot += lossfunc(pred, masks)

#                 '''vis images'''
#                 if ind % args.vis == 0:
#                     namecat = 'Test'
#                     for na in name:
#                         img_name = na.split('/')[-1].split('.')[0]
#                         namecat = namecat + img_name + '+'
#                     vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)
                

#                 temp = eval_seg(pred, masks, threshold)
#                 mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

#             pbar.update()


#     return tot/ n_val , tuple([a/n_val for a in mix_res])




import os
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
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

if args.dataset == 'oneprompt':
    # nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)
    # train_loader, val_loader = get_isic_loader(args)
    train_loader, val_loader = get_btcv_loader(args)
    # Loop through the train_loader to inspect the first batch
    # Loop through the train_loader to inspect the first batch
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        p_labels = batch['p_label']
        pts = batch['pt']
        meta_data = batch['image_meta_dict']
        
        # Print shapes, types, and all returned items from get_item
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("p_label:", p_labels)
        print("pt:", pts)
        print("image_meta_dict:", meta_data)
        
        print("Image batch dtype:", images.dtype)
        print("Label batch dtype:", labels.dtype)
        
        # Print actual images and labels if necessary (optional)
        print("Image batch:", images)
        print("Label batch:", labels)
        
        break  # Exit after the first batch for testing
        
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
   
        break
'''checkpoint path and tensorboard'''
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 10000
for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()

    # loss = function.train_one(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    loss = function.train_one(args, net, optimizer, train_loader, epoch, writer, vis = args.vis)

    logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        # tol, (eiou, edice) = function.validation_one(args, nice_test_loader, epoch, net, writer)
        tol, (eiou, edice) = function.validation_one(args, val_loader, epoch, net, writer)

        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()

        if tol < best_tol:
            best_tol = tol
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
        else:
            is_best = False

writer.close()

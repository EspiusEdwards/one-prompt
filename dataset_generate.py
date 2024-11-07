import sys
sys.argv += ['-net', 'oneprompt', '-mod', 'one_adpt', '-exp_name', 'basic_exp']

import os
import json
import torch
from dataset import ISIC2016
from torchvision import transforms
import debugpy
# debugpy.listen(("0.0.0.0", 5678))  # Port 5678 can be any open port
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # This line will wait until the debugger attaches
# print("Debugger is attached!")

# Define arguments
class Args:
    image_size = 224
    out_size = 224
    prompt = None  # Add if your dataset uses prompts

args = Args()

# Define data path
data_path = '/fred/oz345/khoa/one-prompt/data/ISIC'

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

# Instantiate the dataset
dataset = ISIC2016(
    args, 
    data_path, 
    transform=image_transform, 
    transform_msk=mask_transform, 
    mode='Training'
)
print(dataset.__getitem__(0))
# Collect data entries
# data_entries = []

# for idx in range(len(dataset)):
#     sample = dataset[idx]
#     image_path = sample['image_path']
#     label_path = sample['label_path']

#     data_entry = {
#         'image': image_path,
#         'label': label_path
#     }

#     data_entries.append(data_entry)

# # Shuffle and split data
# import random
# random.shuffle(data_entries)
# split_ratio = 0.8
# split_index = int(len(data_entries) * split_ratio)

# train_data_entries = data_entries[:split_index]
# val_data_entries = data_entries[split_index:]

# # Create JSON structure
# dataset_json = {
#     "name": "ISIC2016",
#     "description": "ISIC 2016 Skin Lesion Segmentation Dataset",
#     "labels": {
#         "0": "background",
#         "1": "lesion"
#     },
#     "numTraining": len(train_data_entries),
#     "numTest": len(val_data_entries),
#     "training": train_data_entries,
#     "validation": val_data_entries
# }

# # Save to dataset_0.json
# output_json_path = os.path.join(data_path, 'dataset_0.json')

# with open(output_json_path, 'w') as f:
#     json.dump(dataset_json, f)

# print(f"dataset_0.json saved to {output_json_path}")

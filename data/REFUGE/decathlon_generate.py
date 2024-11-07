import sys
sys.path.append('/fred/oz345/khoa/one-prompt')

from dataset import REFUGE
from types import SimpleNamespace
import os

# Define sample args with image and mask sizes
args = SimpleNamespace(image_size=224, out_size=128)

# Set the data path to the directory where `REFUGE-Multirater` is located
data_path = '/fred/oz345/khoa/one-prompt/data/REFUGE/REFUGE-Multirater'

# Initialize the dataset in training mode
dataset = REFUGE(args, data_path, mode='Training')

# Check the number of items in the dataset
print(f"Total samples: {len(dataset)}")

# Get a sample from the dataset
sample = dataset[0]
print(sample)

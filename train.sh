#!/bin/bash
#SBATCH --job-name=oneprompt_train_job       # Job name
#SBATCH --output=oneprompt_train_output_%j.log # Output log file (%j will be replaced with the job ID)
#SBATCH --error=oneprompt_train_error_%j.log   # Error log file (%j will be replaced with the job ID)
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for a single script)
#SBATCH --cpus-per-task=3             # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU (adjust as needed)
#SBATCH --mem=8G                      # Memory per node (adjust as needed)
#SBATCH --time=15:00:00                # Maximum time the job can run (adjust as needed)
#SBATCH --partition=gpu                # Partition to submit to (adjust based on your cluster setup)

# Load required modules
module load cuda/12.4.1
module -q load mamba

# Activate your conda environment using mamba
mamba activate oneprompt

# Navigate to your project directory
cd /fred/oz345/khoa/one-prompt # Adjust this path to your project directory

# Run the training Python script with the specified arguments
python3 train.py -net oneprompt -mod one_adpt -exp_name basic_exp -b 4 -dataset oneprompt -patch_size 16 -data_path /fred/oz345/khoa/one-prompt/data/ISIC -baseline 'unet'

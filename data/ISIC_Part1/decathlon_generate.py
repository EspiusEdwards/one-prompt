import os
import json

# Define paths
data_dir = '/fred/oz345/khoa/one-prompt/data/ISIC'
train_images_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_Data')
train_labels_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
test_images_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Test_Data')
test_labels_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Test_GroundTruth')

# Load prompts
prompts_file = os.path.join(data_dir, 'prompts.json')  # Update with the actual path
with open(prompts_file, 'r') as f:
    prompts = json.load(f)

# Build a mapping from image filenames to prompts
prompt_dict = {}
for item in prompts:
    dataset_name = item.get("Dataset:").strip()
    if dataset_name != "ISIC":
        continue
    index = item.get("Index").strip()
    # Extract the sample number from 'Sample0', 'Sample1', etc.
    sample_num = int(index.replace('Sample', ''))
    prompt_type = item.get("Prompt_type")
    prompt_values = item.get("Prompt")
    # Build a key that will match the image filenames
    image_key = f"ISIC_{sample_num:07d}.jpg"
    prompt_dict[image_key] = {
        "prompt_type": prompt_type,
        "prompt_values": prompt_values
    }

def create_data_list(images_dir, labels_dir, split):
    data_list = []
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_files = {}
    if labels_dir and os.path.exists(labels_dir):
        for lbl in os.listdir(labels_dir):
            if lbl.lower().endswith('.png'):
                # Remove '_Segmentation' suffix and extension to match with image filename
                lbl_base = lbl.replace('_Segmentation.png', '')
                label_files[lbl_base] = os.path.join(labels_dir, lbl)

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        base_name = os.path.splitext(image_file)[0]  # e.g., 'ISIC_0000000'
        # Try to find the corresponding label file
        label_path = label_files.get(base_name, None)
        if label_path is None:
            print(f"Warning: No label found for image {image_file}. Skipping.")
            continue  # Skip images without labels
        # Get the prompt for this image
        prompt = prompt_dict.get(image_file, None)
        if prompt is None:
            print(f"Warning: No prompt found for image {image_file}. Using default prompt.")
            prompt = {
                "prompt_type": "0",
                "prompt_values": []
            }
        data_item = {
            'image': image_path,
            'label': label_path,
            'prompt': prompt
        }
        data_list.append(data_item)
    return data_list

# Initialize dataset dict
dataset = {}

# Create training data list
dataset['training'] = create_data_list(train_images_dir, train_labels_dir, 'training')

# Create validation data list
dataset['validation'] = create_data_list(test_images_dir, test_labels_dir, 'validation')

# Save dataset_0.json
dataset_json_path = os.path.join(data_dir, 'dataset_0.json')
with open(dataset_json_path, 'w') as f:
    json.dump(dataset, f, indent=4)

import json

# Load the original JSON file
with open('prompt_data.json', 'r') as file:
    data = json.load(file)

# Filter entries with "Dataset:" as "ISIC"
isic_data = [entry for entry in data if entry.get("Dataset:").strip() == "ISIC"]

# Save the filtered data to a new JSON file named dataset_0.json
with open('dataset_0.json', 'w') as output_file:
    json.dump(isic_data, output_file, indent=4)

print("Filtered data with dataset name 'ISIC' has been saved to dataset_0.json")

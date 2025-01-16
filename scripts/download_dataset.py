from datasets import load_dataset

# Replace 'dataset_name' with the name of the dataset you want to download
dataset_name = "liuhaotian/LLaVA-Pretrain"

# Load the dataset (and subset if applicable)
dataset = load_dataset(dataset_name)  # Omit subset_name if not needed

# Print details about the dataset
print(dataset)

# # Access the training split
# train_data = dataset['train']
# print(f"Number of training examples: {len(train_data)}")

# Save the dataset locally (optional)
dataset.save_to_disk("/home/annie/LLaVA/pretraining_data")
print("Dataset saved locally.")

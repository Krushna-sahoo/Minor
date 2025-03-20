import os
import shutil
import random

# Define paths
dataset_path = r"C:\Users\KIIT\Documents\Minor\preprocessed"
output_path = os.path.join(r"C:\Users\KIIT\Documents\Minor", "splitted_data")
os.makedirs(output_path, exist_ok=True)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def split_and_copy_files(source_folder, dest_folder, train_ratio, val_ratio):
    """Splits dataset into train, validation, and test sets."""
    files = os.listdir(source_folder)
    random.shuffle(files)  
    
    total_files = len(files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    split_mapping = {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:]
    }
    
    for split, file_list in split_mapping.items():
        split_path = os.path.join(dest_folder, split)
        os.makedirs(split_path, exist_ok=True)
        for file in file_list:
            shutil.copy(os.path.join(source_folder, file), os.path.join(split_path, file))
        print(f"{split.capitalize()} set: {len(file_list)} files")


for category in ["real", "fake"]:
    print(f"Processing {category} videos...")
    split_and_copy_files(
        source_folder=os.path.join(dataset_path, category),
        dest_folder=os.path.join(output_path, category),
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

print("Dataset splitting complete!")

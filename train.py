import os
import shutil
import random

# Define paths
CAT_SOURCE = "grayscale_cat_image"
DOG_SOURCE = "grayscale_dog_image"
UNKNOWN_SOURCE = "grayscale_unknown"

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

# Create train and test directories
for category in ["cat", "dog", "unknown"]:
    os.makedirs(os.path.join(TRAIN_PATH, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_PATH, category), exist_ok=True)

# Function to split dataset
def split_images(source_folder, train_dest, test_dest, split_ratio=0.8):
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)  # Shuffle images for randomness

    split_index = int(len(files) * split_ratio)
    train_files, test_files = files[:split_index], files[split_index:]

    # Move files to respective folders
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_dest, file))
    
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_dest, file))

# Split images
split_images(CAT_SOURCE, os.path.join(TRAIN_PATH, "cat"), os.path.join(TEST_PATH, "cat"))
split_images(DOG_SOURCE, os.path.join(TRAIN_PATH, "dog"), os.path.join(TEST_PATH, "dog"))
split_images(UNKNOWN_SOURCE, os.path.join(TRAIN_PATH, "unknown"), os.path.join(TEST_PATH, "unknown"))


print("✅ Dataset organized into train and test folders successfully!")

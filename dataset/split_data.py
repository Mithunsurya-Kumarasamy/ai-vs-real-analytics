import os
import shutil
import random
from pathlib import Path

# Updated to look inside the 'dataset' folder
SOURCE_AI = Path("dataset/Ai_generated_dataset") 
SOURCE_REAL = Path("dataset/real_dataset")

# Where your pipeline expects them to go
DEST_DIR = Path("dataset")

def split_and_copy(source_folder, category_name, split_ratio=0.8):
    if not source_folder.exists():
        print(f"Error: Could not find '{source_folder}'.")
        return

    # Look deeply into all subfolders for valid image types
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in source_folder.rglob("*.*") if f.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"No valid images found anywhere inside {source_folder}!")
        return
        
    # Shuffle them
    random.shuffle(images) 
    
    # Calculate the 80/20 split
    split_index = int(len(images) * split_ratio)
    train_imgs = images[:split_index]
    test_imgs = images[split_index:]
    
    # Create the destination folders safely
    train_dest = DEST_DIR / "train" / category_name
    test_dest = DEST_DIR / "test" / category_name
    train_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Flattening and copying {len(train_imgs)} {category_name} images to train...")
    for img in train_imgs:
        # Prepend the subfolder name to prevent naming collisions
        safe_name = f"{img.parent.name}_{img.name}"
        shutil.copy(img, train_dest / safe_name)
        
    print(f"Flattening and copying {len(test_imgs)} {category_name} images to test...")
    for img in test_imgs:
        safe_name = f"{img.parent.name}_{img.name}"
        shutil.copy(img, test_dest / safe_name)

# Execute the split
print("Starting the split...")
split_and_copy(SOURCE_AI, "AI")
split_and_copy(SOURCE_REAL, "Real")
print("All done! You can now run ml_pipeline.py")
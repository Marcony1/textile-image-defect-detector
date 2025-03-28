import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
AUGMENTED_TRAIN_DIR = PROJECT_ROOT / "data" / "augmented_train"

AUGMENT_FACTOR = 3  # Images to generate per original training image
TEST_SIZE = 0.2     # 20% for test set
SEED = 42           # For reproducibility

# --- 1. TRAIN/TEST SPLIT (USING ORIGINAL IMAGES) ---
def split_data():
    # Create split directories
    for split in ["train", "test"]:
        for class_name in ["OK", "NO_OK"]:
            (SPLIT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in ["OK", "NO_OK"]:
        src_dir = RAW_DATA_DIR / class_name
        all_images = os.listdir(src_dir)
        
        # Split original images
        train_files, test_files = train_test_split(
            all_images, test_size=TEST_SIZE, random_state=SEED
        )
        
        # Copy original images to split directories
        for f in train_files:
            shutil.copy2(src_dir / f, SPLIT_DIR / "train" / class_name / f)
        for f in test_files:
            shutil.copy2(src_dir / f, SPLIT_DIR / "test" / class_name / f)

# --- 2. AUGMENT ONLY TRAINING DATA ---
def augment_training_data():
    datagen = ImageDataGenerator(
        rotation_range=10,          
        # width_shift_range=0.1,      
        # height_shift_range=0.1,    
        brightness_range=[0.8, 1.2], 
        zoom_range=0.05,            
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'         
    )

    # Create augmented training directory
    (AUGMENTED_TRAIN_DIR / "OK").mkdir(parents=True, exist_ok=True)
    (AUGMENTED_TRAIN_DIR / "NO_OK").mkdir(parents=True, exist_ok=True)

    for class_name in ["OK", "NO_OK"]:
        input_dir = SPLIT_DIR / "train" / class_name
        output_dir = AUGMENTED_TRAIN_DIR / class_name
        
        print(f"Augmenting training images from: {input_dir}")
        
        for img_name in os.listdir(input_dir):
            img_path = input_dir / img_name
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prefix = os.path.splitext(img_name)[0]
            i = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                   save_to_dir=output_dir,
                                   save_prefix=prefix,
                                   save_format='jpg'):
                i += 1
                if i >= AUGMENT_FACTOR:
                    break

if __name__ == "__main__":
    print("Splitting original data...")
    split_data()
    print("Augmenting training data...")
    augment_training_data()
    
    print("\nProcessing complete!")
    print(f"Original train/test splits saved to: {SPLIT_DIR}")
    print(f"Augmented training data saved to: {AUGMENTED_TRAIN_DIR}")
    print(f"\nFile counts:")
    print(f"- Original training OK: {len(os.listdir(SPLIT_DIR/'train'/'OK'))}")
    print(f"- Augmented training OK: {len(os.listdir(AUGMENTED_TRAIN_DIR/'OK'))}")
    print(f"- Test set OK: {len(os.listdir(SPLIT_DIR/'test'/'OK'))}")
#!/usr/bin/env python3
import os
import json
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.7, 0.1, 0.2), seed=42):
    """
    Splits YOLO dataset into train/val/test while preserving class distribution
    
    Args:
        input_dir: Path to input folder (e.g., 'data/01_annotated_data')
        output_parent_dir: Parent directory for output (e.g., 'data')
        ratios: Tuple of (train, val, test) ratios
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Validate and prepare paths
    input_path = Path(input_dir)
    output_path = Path(output_parent_dir) / f"{input_path.name}_splitted"
    
    # Clear existing output if it exists
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Validate ratios
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    assert len(ratios) == 3, "Need exactly 3 ratios (train, val, test)"
    
    # Read classes
    classes_file = input_path / "classes.txt"
    with open(classes_file) as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Read notes.json if exists
    metadata = {}
    if (input_path / "notes.json").exists():
        with open(input_path / "notes.json") as f:
            metadata = json.load(f)
    
    # Get all image files
    image_files = list((input_path / "images").glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    # Create stratified split (multi-label aware)
    def get_image_classes(img_path):
        txt_path = input_path / "labels" / f"{img_path.stem}.txt"
        if not txt_path.exists():
            return []
        with open(txt_path) as f:
            lines = f.readlines()
        return list(set(int(line.split()[0]) for line in lines if line.strip()))
    
    # First split: train vs temp (val+test)
    train_ratio, val_ratio, test_ratio = ratios
    temp_ratio = val_ratio + test_ratio
    
    X = image_files
    y = [get_image_classes(img) for img in X]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=temp_ratio,
        random_state=seed,
        stratify=y
    )
    
    # Split temp into val and test
    val_test_ratio = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1-val_test_ratio,
        random_state=seed,
        stratify=y_temp
    )
    
    # Create output directory structure
    splits = {
        'train': X_train,
        'val': X_val,
        'test': X_test
    }
    
    for split_name, split_files in splits.items():
        # Create directories
        (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_path in split_files:
            # Copy image
            shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
            
            # Copy corresponding label
            label_path = input_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)
        
        # Copy classes.txt to each split
        shutil.copy(classes_file, output_path / split_name / 'classes.txt')
        
        # Copy notes.json if exists
        if metadata:
            with open(output_path / split_name / 'notes.json', 'w') as f:
                json.dump(metadata, f)
    
    print(f"""
    Dataset split complete!
    - Input: {input_path}
    - Output: {output_path}
    - Train: {len(X_train)} images
    - Val: {len(X_val)} images
    - Test: {len(X_test)} images
    - Classes: {len(classes)} ({', '.join(classes[:3])}{'...' if len(classes) > 3 else ''})
    """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val/test')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input folder (e.g., data/01_annotated_data)')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.2, 0.1],
                       help='Train/val/test ratios (must sum to 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Automatically determine output parent directory
    input_path = Path(args.input)
    output_parent = input_path.parent
    
    split_yolo_dataset(args.input, output_parent, args.ratios, args.seed)
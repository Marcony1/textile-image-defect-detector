# import os
# import json
# import random
# import shutil
# import numpy as np
# from pathlib import Path
# from collections import Counter
# from sklearn.model_selection import train_test_split

# def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.6, 0.1, 0.2), seed=42):
#     """
#     Splits YOLO dataset into train/val/test with fallback for rare classes
    
#     Args:
#         input_dir: Path to input folder (e.g., 'data/01_annotated_data')
#         output_parent_dir: Parent directory for output (e.g., 'data')
#         ratios: Tuple of (train, val, test) ratios
#         seed: Random seed for reproducibility
#     """
#     random.seed(seed)
    
#     # Validate and prepare paths
#     input_path = Path(input_dir)
#     output_path = Path(output_parent_dir) / f"{input_path.name}_splitted"
    
#     # Clear existing output if it exists
#     if output_path.exists():
#         shutil.rmtree(output_path)
    
#     # Validate ratios
#     assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
#     assert len(ratios) == 3, "Need exactly 3 ratios (train, val, test)"
    
#     # Read classes
#     classes_file = input_path / "classes.txt"
#     with open(classes_file) as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     # Get all image files
#     image_files = list((input_path / "images").glob("*"))
#     image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
#     # Create multi-label compatible split
#     def get_image_classes(img_path):
#         txt_path = input_path / "labels" / f"{img_path.stem}.txt"
#         if not txt_path.exists():
#             return []
#         with open(txt_path) as f:
#             lines = f.readlines()
#         return list(set(int(line.split()[0]) for line in lines if line.strip()))
    
#     # Create binary matrix and track class counts
#     class_matrix = []
#     class_counts = Counter()
    
#     for img_path in image_files:
#         class_indices = get_image_classes(img_path)
#         binary_vec = [1 if i in class_indices else 0 for i in range(len(classes))]
#         class_matrix.append(binary_vec)
#         for cls in class_indices:
#             class_counts[cls] += 1
    
#     class_matrix = np.array(class_matrix)
#     X = np.array(image_files)
    
#     # Identify rare classes (appearing in <2 images)
#     rare_classes = [cls for cls, count in class_counts.items() if count < 2]
    
#     try:
#         # Attempt stratified splitting
#         X_train, X_temp = train_test_split(
#             X,
#             test_size=ratios[1] + ratios[2],
#             random_state=seed,
#             stratify=class_matrix
#         )
#         temp_matrix = class_matrix[np.isin(X, X_temp)]
#         X_val, X_test = train_test_split(
#             X_temp,
#             test_size=ratios[2]/(ratios[1]+ratios[2]),
#             random_state=seed,
#             stratify=temp_matrix
#         )
#     except ValueError as e:
#         print(f"\n⚠️ Stratified split failed: {e}")
#         print("Falling back to random split without stratification...\n")
#         X_train, X_temp = train_test_split(
#             X,
#             test_size=ratios[1] + ratios[2],
#             random_state=seed
#         )
#         X_val, X_test = train_test_split(
#             X_temp,
#             test_size=ratios[2]/(ratios[1]+ratios[2]),
#             random_state=seed
#         )

    
#     # Create output directory structure
#     splits = {
#         'train': X_train,
#         'val': X_val,
#         'test': X_test
#     }
    
#     for split_name, split_files in splits.items():
#         # Create directories
#         (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
#         (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        
#         # Copy files
#         for img_path in split_files:
#             shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
#             label_path = input_path / "labels" / f"{img_path.stem}.txt"
#             if label_path.exists():
#                 shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)
        
#         shutil.copy(classes_file, output_path / split_name / 'classes.txt')
#         if (input_path / "notes.json").exists():
#             shutil.copy(input_path / "notes.json", output_path / split_name / 'notes.json')
    
#     # Print summary
#     print(f"\nDataset split complete!")
#     print(f"- Input: {input_path}")
#     print(f"- Output: {output_path}")
#     print(f"- Train: {len(X_train)} images")
#     print(f"- Val: {len(X_val)} images")
#     print(f"- Test: {len(X_test)} images")
#     print(f"- Classes: {len(classes)}")
    
#     if rare_classes:
#         print(f"\nWarning: Used fallback splitting for rare classes:")
#         for cls in rare_classes:
#             print(f"  - {classes[cls]} (appears in {class_counts[cls]} images)")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val/test')
#     parser.add_argument('--input', type=str, required=True,
#                        help='Path to input folder (e.g., data/01_annotated_data)')
#     parser.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.2, 0.1],
#                        help='Train/val/test ratios (must sum to 1.0)')
#     parser.add_argument('--seed', type=int, default=42,
#                        help='Random seed for reproducibility')
    
#     args = parser.parse_args()
    
#     # Automatically determine output parent directory
#     input_path = Path(args.input)
#     output_parent = input_path.parent
    
#     split_yolo_dataset(args.input, output_parent, args.ratios, args.seed)


import os
import json
import random
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.8, 0.2), seed=42):
    """
    Splits YOLO dataset into train/val with fallback for rare classes

    Args:
        input_dir: Path to input folder (e.g., 'data/01_annotated_data')
        output_parent_dir: Parent directory for output (e.g., 'data')
        ratios: Tuple of (train, val) ratios (must sum to 1.0)
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
    assert len(ratios) == 2, "Need exactly 2 ratios (train, val)"

    # Read classes
    classes_file = input_path / "classes.txt"
    with open(classes_file) as f:
        classes = [line.strip() for line in f.readlines()]

    # Get all image files
    image_files = list((input_path / "images").glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    def get_image_classes(img_path):
        txt_path = input_path / "labels" / f"{img_path.stem}.txt"
        if not txt_path.exists():
            return []
        with open(txt_path) as f:
            lines = f.readlines()
        return list(set(int(line.split()[0]) for line in lines if line.strip()))

    class_matrix = []
    class_counts = Counter()

    for img_path in image_files:
        class_indices = get_image_classes(img_path)
        binary_vec = [1 if i in class_indices else 0 for i in range(len(classes))]
        class_matrix.append(binary_vec)
        for cls in class_indices:
            class_counts[cls] += 1

    class_matrix = np.array(class_matrix)
    X = np.array(image_files)

    rare_classes = [cls for cls, count in class_counts.items() if count < 2]

    try:
        X_train, X_val = train_test_split(
            X,
            test_size=ratios[1],
            random_state=seed,
            stratify=class_matrix
        )
    except ValueError as e:
        print(f"\n⚠️ Stratified split failed: {e}")
        print("Falling back to random split without stratification...\n")
        X_train, X_val = train_test_split(
            X,
            test_size=ratios[1],
            random_state=seed
        )

    # Create output directory structure
    splits = {
        'train': X_train,
        'val': X_val
    }

    for split_name, split_files in splits.items():
        (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)

        for img_path in split_files:
            shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
            label_path = input_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)

        shutil.copy(classes_file, output_path / split_name / 'classes.txt')
        if (input_path / "notes.json").exists():
            shutil.copy(input_path / "notes.json", output_path / split_name / 'notes.json')

    print(f"\n✅ Dataset split complete!")
    print(f"- Input: {input_path}")
    print(f"- Output: {output_path}")
    print(f"- Train: {len(X_train)} images")
    print(f"- Val: {len(X_val)} images")
    print(f"- Classes: {len(classes)}")

    if rare_classes:
        print(f"\n⚠️ Warning: Fallback splitting used for rare classes:")
        for cls in rare_classes:
            print(f"  - {classes[cls]} (appears in {class_counts[cls]} images)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input folder (e.g., data/01_annotated_data)')
    parser.add_argument('--ratios', type=float, nargs=2, default=[0.8, 0.2],
                        help='Train/val ratios (must sum to 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_parent = input_path.parent

    split_yolo_dataset(args.input, output_parent, args.ratios, args.seed)

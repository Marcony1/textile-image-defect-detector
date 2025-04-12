# import os
# import json
# import random
# import shutil
# import numpy as np
# from pathlib import Path
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.8, 0.2), seed=42):
#     """
#     Splits YOLO dataset into train/val with fallback for rare classes

#     Args:
#         input_dir: Path to input folder (e.g., 'data/01_annotated_data')
#         output_parent_dir: Parent directory for output (e.g., 'data')
#         ratios: Tuple of (train, val) ratios (must sum to 1.0)
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
#     assert len(ratios) == 2, "Need exactly 2 ratios (train, val)"

#     # Read classes
#     classes_file = input_path / "classes.txt"
#     with open(classes_file) as f:
#         classes = [line.strip() for line in f.readlines()]

#     # Get all image files
#     image_files = list((input_path / "images").glob("*"))
#     image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

#     def get_image_classes(img_path):
#         txt_path = input_path / "labels" / f"{img_path.stem}.txt"
#         if not txt_path.exists():
#             return []
#         with open(txt_path) as f:
#             lines = f.readlines()
#         return list(set(int(line.split()[0]) for line in lines if line.strip()))

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

#     rare_classes = [cls for cls, count in class_counts.items() if count < 2]

#     try:
#         msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratios[1], random_state=seed)
#         train_idx, val_idx = next(msss.split(X, class_matrix))
#         X_train, X_val = X[train_idx], X[val_idx]
#     except Exception as e:
#         print(f"\n⚠️ Iterative stratification failed: {e}")
#         print("Falling back to random split without stratification...\n")
#         X_train, X_val = train_test_split(
#             X,
#             test_size=ratios[1],
#             random_state=seed
#         )


#     # Create output directory structure
#     splits = {
#         'train': X_train,
#         'val': X_val
#     }

#     for split_name, split_files in splits.items():
#         (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
#         (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)

#         for img_path in split_files:
#             shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
#             label_path = input_path / "labels" / f"{img_path.stem}.txt"
#             if label_path.exists():
#                 shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)

#         shutil.copy(classes_file, output_path / split_name / 'classes.txt')
#         if (input_path / "notes.json").exists():
#             shutil.copy(input_path / "notes.json", output_path / split_name / 'notes.json')

#     print(f"\n✅ Dataset split complete!")
#     print(f"- Input: {input_path}")
#     print(f"- Output: {output_path}")
#     print(f"- Train: {len(X_train)} images")
#     print(f"- Val: {len(X_val)} images")
#     print(f"- Classes: {len(classes)}")

#     if rare_classes:
#         print(f"\n⚠️ Warning: Fallback splitting used for rare classes:")
#         for cls in rare_classes:
#             print(f"  - {classes[cls]} (appears in {class_counts[cls]} images)")

#     # Print class balance stats
#     def count_class_distribution(split_files, input_path, num_classes):
#         counts = np.zeros(num_classes)
#         for img_path in split_files:
#             txt_path = input_path / "labels" / f"{img_path.stem}.txt"
#             if not txt_path.exists():
#                 continue
#             with open(txt_path) as f:
#                 for line in f:
#                     if line.strip():
#                         class_id = int(line.split()[0])
#                         counts[class_id] += 1
#         return counts

#     print("\n📊 Class distribution:")
#     print("Train:", count_class_distribution(X_train, input_path, len(classes)))
#     print("Val  :", count_class_distribution(X_val, input_path, len(classes)))

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Path to input folder (e.g., data/01_annotated_data)')
#     parser.add_argument('--ratios', type=float, nargs=2, default=[0.8, 0.2],
#                         help='Train/val ratios (must sum to 1.0)')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed for reproducibility')

#     args = parser.parse_args()

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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.8, 0.2), seed=42, stratify=True):
    """
    Splits YOLO dataset into train/valid sets.

    Args:
        input_dir: Path to input folder (e.g., 'data/01_annotated_data')
        output_parent_dir: Parent directory for output (e.g., 'data')
        ratios: Tuple of (train, valid) ratios (must sum to 1.0)
        seed: Random seed
        stratify: Whether to use multi-label stratification
    """
    random.seed(seed)

    # Paths
    input_path = Path(input_dir)
    output_path = Path(output_parent_dir) / f"{input_path.name}_splitted"

    # Reset output folder
    if output_path.exists():
        shutil.rmtree(output_path)

    # Validate ratios
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    assert len(ratios) == 2, "Provide exactly 2 ratios (train, valid)"

    # Read class list
    classes_file = input_path / "classes.txt"
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]

    # Read image paths
    image_files = [f for f in (input_path / "images").glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    def get_image_classes(img_path):
        label_path = input_path / "labels" / f"{img_path.stem}.txt"
        if not label_path.exists():
            return []
        with open(label_path) as f:
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

    if stratify:
        try:
            splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratios[1], random_state=seed)
            train_idx, valid_idx = next(splitter.split(X, class_matrix))
            X_train, X_valid = X[train_idx], X[valid_idx]
        except Exception as e:
            print(f"\n⚠️ Stratification failed: {e}\nFalling back to random split.")
            stratify = False

    if not stratify:
        X_train, X_valid = train_test_split(X, test_size=ratios[1], random_state=seed)

    splits = {
        'train': X_train,
        'valid': X_valid  # renamed from val to valid
    }

    for split_name, split_files in splits.items():
        (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)

        for img_path in split_files:
            shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
            label_path = input_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)

    # Generate YOLO-compatible data.yaml
    data_yaml = {
        'train': str((output_path / 'train' / 'images').resolve()),
        'val': str((output_path / 'valid' / 'images').resolve()),
        'nc': len(classes),
        'names': classes
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml_content = f"train: {data_yaml['train']}\nval: {data_yaml['val']}\nnc: {data_yaml['nc']}\nnames: {data_yaml['names']}"
        f.write(yaml_content)

    # Summary
    print(f"\n✅ Dataset split complete:")
    print(f"- Output folder: {output_path}")
    print(f"- Train: {len(X_train)} images")
    print(f"- Valid: {len(X_valid)} images")
    print(f"- Classes: {len(classes)}")
    if rare_classes:
        print("\n⚠️ Rare classes:")
        for cls in rare_classes:
            print(f"  - {classes[cls]} ({class_counts[cls]} occurrences)")

    # Class distribution
    def count_class_distribution(split_files, base_path, num_classes):
        counts = np.zeros(num_classes)
        for img_path in split_files:
            label_path = base_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            counts[class_id] += 1
        return counts

    print("\n📊 Class distribution:")
    print("Train:", count_class_distribution(X_train, input_path, len(classes)))
    print("Valid:", count_class_distribution(X_valid, input_path, len(classes)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Split a YOLOv5 dataset into train/valid')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the annotated YOLO dataset (e.g., data/01_annotated_data)')
    parser.add_argument('--ratios', type=float, nargs=2, default=[0.8, 0.2],
                        help='Train/valid split ratios (e.g., 0.8 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--stratify', type=bool, default=True,
                        help='Whether to use multilabel stratification (default: True)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_parent = input_path.parent

    split_yolo_dataset(
        input_dir=args.input,
        output_parent_dir=output_parent,
        ratios=args.ratios,
        seed=args.seed,
        stratify=args.stratify
    )

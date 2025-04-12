import os
import json
import random
import shutil
import numpy as np
import yaml
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def create_data_yaml(output_dir, names, include_test=False):
    """Create a YOLO-compatible data.yaml file with relative paths"""
    data = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': '# test images (optional)',
        'nc': len(names),
        'names': names
    }
    if include_test:
        data['test'] = 'test/images'

    with open(Path(output_dir) / 'data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Created YOLO data.yaml at: {Path(output_dir) / 'data.yaml'}")


def split_yolo_dataset(input_dir, output_parent_dir, ratios=(0.8, 0.2, 0.0), seed=42, stratify=True):
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_parent_dir) / f"{input_path.name}_splitted"

    if output_path.exists():
        shutil.rmtree(output_path)

    assert sum(ratios) <= 1.0, "Ratios must sum to 1.0 or less (train + valid + test)"
    assert len(ratios) == 3, "Provide 3 ratios: train, valid, test"

    # Read classes
    classes_file = input_path / "classes.txt"
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]

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

    if stratify:
        try:
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratios[1] + ratios[2], random_state=seed)
            train_idx, temp_idx = next(msss.split(X, class_matrix))
            X_train, X_temp = X[train_idx], X[temp_idx]
            class_matrix_temp = class_matrix[temp_idx]

            if ratios[2] > 0:
                relative_valid_ratio = ratios[1] / (ratios[1] + ratios[2])
                msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - relative_valid_ratio, random_state=seed)
                valid_idx, test_idx = next(msss2.split(X_temp, class_matrix_temp))
                X_valid, X_test = X_temp[valid_idx], X_temp[test_idx]
            else:
                X_valid, X_test = X_temp, np.array([])

        except Exception as e:
            print(f"⚠️ Stratified split failed: {e}\nFalling back to random split.")
            stratify = False

    if not stratify:
        X_train, X_temp = train_test_split(X, test_size=ratios[1] + ratios[2], random_state=seed)
        if ratios[2] > 0:
            relative_valid_ratio = ratios[1] / (ratios[1] + ratios[2])
            X_valid, X_test = train_test_split(X_temp, test_size=1 - relative_valid_ratio, random_state=seed)
        else:
            X_valid, X_test = X_temp, np.array([])

    splits = {
        'train': X_train,
        'valid': X_valid
    }
    if ratios[2] > 0:
        splits['test'] = X_test

    for split_name, split_files in splits.items():
        (output_path / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split_name / 'labels').mkdir(parents=True, exist_ok=True)

        for img_path in split_files:
            shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
            label_path = input_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_path / split_name / 'labels' / label_path.name)

    create_data_yaml(output_path, classes, include_test=ratios[2] > 0)

    print(f"\n✅ Split completed → {output_path}")
    for split_name in splits:
        print(f"- {split_name.capitalize():<6}: {len(splits[split_name])} images")

    print("\n📊 Class distribution per set:")
    def count_class_distribution(split_files):
        counts = np.zeros(len(classes))
        for img_path in split_files:
            label_path = input_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            counts[class_id] += 1
        return counts

    for split_name, split_files in splits.items():
        print(f"{split_name.capitalize():<6}: {count_class_distribution(split_files)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Split YOLO dataset into train/valid[/test] with optional stratification.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to YOLO-annotated dataset (e.g., data/01_annotated_data)')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.8, 0.2, 0.0],
                        help='Ratios for train, valid, test (must sum to ≤ 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--stratify', type=bool, default=True,
                        help='Use multilabel stratified sampling (default=True)')

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

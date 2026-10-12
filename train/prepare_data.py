"""
Prepare dataset manifest and stratified train/validation/test splits for khana dataset.

Creates a CSV manifest with image paths, class labels, and split assignments.
Handles extensionless JPEG files and excludes macOS metadata.
Uses stratified split to maintain class distribution across train/val/test.
"""

import json
from pathlib import Path
from collections import Counter
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import yaml


def is_valid_image_file(filepath: Path) -> bool:
    """Check if file is a valid image (exclude metadata files)."""
    name = filepath.name
    if name.startswith('._') or name == '.DS_Store':
        return False
    return filepath.is_file()


def scan_dataset(dataset_root: Path) -> dict:
    """
    Scan khana dataset and collect all valid image files by category.

    Returns:
        dict: {category_name: [Path, Path, ...], ...}
    """
    categories = {}
    dataset_root = Path(dataset_root)

    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue

        cat_name = cat_dir.name
        images = []

        for f in cat_dir.iterdir():
            if is_valid_image_file(f):
                images.append(f)

        if images:
            categories[cat_name] = sorted(images)

    return categories


def create_manifest(categories: dict, train_split: float, val_split: float,
                   test_split: float, random_seed: int) -> list:
    """
    Create dataset manifest with stratified splits.

    Returns:
        list: [{image_path, class_label, split}, ...]
    """
    manifest = []
    np.random.seed(random_seed)

    class_id_map = {cat: idx for idx, cat in enumerate(sorted(categories.keys()))}

    for cat_name, images in categories.items():
        class_id = class_id_map[cat_name]
        class_labels = [cat_name] * len(images)

        # First split: train vs (val+test)
        indices = np.arange(len(images))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(1.0 - train_split),
            random_state=random_seed,
            stratify=class_labels
        )

        # Second split: val vs test
        temp_labels = [class_labels[i] for i in temp_idx]
        val_size = val_split / (val_split + test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_size),
            random_state=random_seed,
            stratify=temp_labels
        )

        # Assign splits
        for idx in train_idx:
            manifest.append({
                'image_path': str(images[idx]),
                'class_label': cat_name,
                'class_id': class_id,
                'split': 'train'
            })

        for idx in val_idx:
            manifest.append({
                'image_path': str(images[idx]),
                'class_label': cat_name,
                'class_id': class_id,
                'split': 'val'
            })

        for idx in test_idx:
            manifest.append({
                'image_path': str(images[idx]),
                'class_label': cat_name,
                'class_id': class_id,
                'split': 'test'
            })

    return manifest, class_id_map


def save_manifest(manifest: list, output_path: Path):
    """Save manifest as CSV."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'class_label', 'class_id', 'split'])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"✓ Manifest saved to {output_path}")


def save_class_labels(class_id_map: dict, output_path: Path):
    """Save class label mapping as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(class_id_map, f, indent=2)

    print(f"✓ Class labels saved to {output_path}")


def print_split_summary(manifest: list):
    """Print summary of splits and class distribution."""
    splits = Counter(m['split'] for m in manifest)
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total images: {len(manifest)}")
    print(f"  Train: {splits['train']} ({splits['train']/len(manifest)*100:.1f}%)")
    print(f"  Val:   {splits['val']} ({splits['val']/len(manifest)*100:.1f}%)")
    print(f"  Test:  {splits['test']} ({splits['test']/len(manifest)*100:.1f}%)")

    # Class distribution
    train_classes = Counter(m['class_label'] for m in manifest if m['split'] == 'train')
    print(f"\nClasses: {len(train_classes)}")
    print(f"Images per class (train set):")
    print(f"  Min: {min(train_classes.values())}")
    print(f"  Max: {max(train_classes.values())}")
    print(f"  Avg: {sum(train_classes.values())/len(train_classes):.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset manifest and splits')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config['dataset_root'])
    manifest_output = Path(config['manifest_output'])
    class_labels_output = Path(config['class_labels_output'])

    print(f"Scanning dataset: {dataset_root}")
    categories = scan_dataset(dataset_root)
    print(f"✓ Found {len(categories)} categories")

    print(f"Creating stratified splits (seed={config['random_seed']})...")
    manifest, class_id_map = create_manifest(
        categories,
        train_split=config['train_split'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        random_seed=config['random_seed']
    )
    print(f"✓ Created splits for {len(manifest)} images")

    save_manifest(manifest, manifest_output)
    save_class_labels(class_id_map, class_labels_output)

    print_split_summary(manifest)


if __name__ == '__main__':
    main()

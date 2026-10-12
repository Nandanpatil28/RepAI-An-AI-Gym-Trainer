"""
Custom data loader for khana dataset with extensionless JPEG support.

Loads images directly from their original paths without copying.
Handles extensionless JPEG files by detecting their format.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(image_path: str, image_size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image from disk.

    Handles extensionless JPEG files by detecting format.

    Args:
        image_path: Path to image file (may be extensionless)
        image_size: Target image size (height, width)

    Returns:
        np.ndarray: Preprocessed image [0, 1] in RGB format
    """
    filepath = Path(image_path)

    try:
        # Attempt to open with PIL (auto-detects format)
        img = Image.open(filepath)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize(image_size, Image.Resampling.LANCZOS)

        # Convert to array and normalize [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        return img_array

    except Exception as e:
        raise ValueError(f"Failed to load image {filepath}: {e}")


def create_dataset_from_manifest(manifest_path: str, split: str,
                                 class_id_map: dict, image_size: tuple = (224, 224),
                                 batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from manifest CSV.

    Args:
        manifest_path: Path to dataset_manifest.csv
        split: 'train', 'val', or 'test'
        class_id_map: {class_name: class_id, ...}
        image_size: Target image size
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset with (images, labels) batches
    """
    import csv

    # Read manifest and filter by split
    image_paths = []
    labels = []

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == split:
                image_paths.append(row['image_path'])
                class_label = row['class_label']
                class_id = class_id_map[class_label]
                labels.append(class_id)

    if not image_paths:
        raise ValueError(f"No images found for split '{split}'")

    print(f"Loaded {len(image_paths)} images for split '{split}'")

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    # Load and preprocess images
    def load_and_preprocess(path, label):
        image = tf.py_function(
            func=lambda p: load_image(p.numpy().decode('utf-8'), image_size),
            inp=[path],
            Tout=tf.float32
        )
        image.set_shape((image_size[0], image_size[1], 3))
        return image, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_class_weights(manifest_path: str, class_id_map: dict,
                     split: str = 'train') -> dict:
    """
    Compute class weights to handle imbalance.

    Args:
        manifest_path: Path to dataset_manifest.csv
        class_id_map: {class_name: class_id, ...}
        split: Which split to use for weight computation

    Returns:
        dict: {class_id: weight, ...}
    """
    import csv
    from collections import Counter

    class_counts = Counter()

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == split:
                class_id = class_id_map[row['class_label']]
                class_counts[class_id] += 1

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    # Compute weights: weight = total / (num_classes * count)
    class_weights = {
        class_id: total / (num_classes * count)
        for class_id, count in class_counts.items()
    }

    print(f"Class weights (sample):")
    for cid in sorted(class_weights.keys())[:3]:
        print(f"  Class {cid}: {class_weights[cid]:.3f}")

    return class_weights

"""
Training entry point for MobileNetV2 food classification.

Supports two phases:
  1. head: Train classification head with frozen base
  2. finetune: Fine-tune base layers with lower learning rate

Usage:
  python train/train.py --config train/config.yaml --phase head
  python train/train.py --config train/config.yaml --phase finetune --checkpoint train/models/model.keras
"""

import json
import argparse
from pathlib import Path

import yaml
import tensorflow as tf

from prepare_data import create_manifest, scan_dataset
from dataloader import create_dataset_from_manifest, get_class_weights
from model import create_mobilenetv2_model, compile_model, unfreeze_base_for_finetuning, save_model, load_model


def train_head(config: dict):
    """
    Train classification head with frozen base.

    Args:
        config: Configuration dict from YAML
    """
    print("\n" + "="*60)
    print("PHASE 1: Training Classification Head (Frozen Base)")
    print("="*60 + "\n")

    dataset_root = Path(config['dataset_root'])
    manifest_path = Path(config['manifest_output'])
    model_dir = Path(config['model_output_dir'])
    logs_dir = Path(config['logs_dir'])
    class_labels_path = Path(config['class_labels_output'])

    # Load class labels
    with open(class_labels_path, 'r') as f:
        class_id_map = json.load(f)

    num_classes = len(class_id_map)
    print(f"Number of classes: {num_classes}")

    # Create datasets
    print(f"\nLoading training data...")
    train_dataset = create_dataset_from_manifest(
        str(manifest_path), 'train', class_id_map,
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        shuffle=True
    )

    print(f"Loading validation data...")
    val_dataset = create_dataset_from_manifest(
        str(manifest_path), 'val', class_id_map,
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Get class weights
    print(f"\nComputing class weights...")
    class_weights = get_class_weights(str(manifest_path), class_id_map, split='train')

    # Create and compile model
    print(f"\nCreating model...")
    model = create_mobilenetv2_model(
        num_classes,
        image_size=tuple(config['image_size']),
        use_imagenet_weights=config['use_imagenet_weights']
    )
    model.summary()

    compile_model(model, learning_rate=config['learning_rate_head'])

    # Callbacks
    checkpoint_path = model_dir / 'model_head.keras'
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir / 'head'),
            histogram_freq=1
        )
    ]

    # Train
    print(f"\nTraining (epochs={config['epochs_head']}, batch_size={config['batch_size']})...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs_head'],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=config['verbose']
    )

    # Save model
    save_model(model, model_dir, 'model_head', 'keras', class_id_map)

    print("\n✓ Head training complete!")
    return model, class_id_map


def train_finetune(config: dict, checkpoint_path: str = None):
    """
    Fine-tune base model layers.

    Args:
        config: Configuration dict from YAML
        checkpoint_path: Path to head checkpoint (optional)
    """
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Base Model")
    print("="*60 + "\n")

    manifest_path = Path(config['manifest_output'])
    model_dir = Path(config['model_output_dir'])
    logs_dir = Path(config['logs_dir'])
    class_labels_path = Path(config['class_labels_output'])

    # Load class labels
    with open(class_labels_path, 'r') as f:
        class_id_map = json.load(f)

    num_classes = len(class_id_map)

    # Load model from checkpoint
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}...")
        model = load_model(checkpoint_path)
    else:
        print(f"Loading model from {model_dir / 'model_head.keras'}...")
        model = load_model(str(model_dir / 'model_head.keras'))

    # Create datasets
    print(f"\nLoading training data...")
    train_dataset = create_dataset_from_manifest(
        str(manifest_path), 'train', class_id_map,
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        shuffle=True
    )

    print(f"Loading validation data...")
    val_dataset = create_dataset_from_manifest(
        str(manifest_path), 'val', class_id_map,
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Get class weights
    print(f"\nComputing class weights...")
    class_weights = get_class_weights(str(manifest_path), class_id_map, split='train')

    # Unfreeze base and recompile
    print(f"\nUnfreezing base model for fine-tuning...")
    unfreeze_base_for_finetuning(model, num_layers_to_unfreeze=50)
    compile_model(model, learning_rate=config['learning_rate_finetune'])

    # Callbacks
    checkpoint_path = model_dir / 'model_finetune.keras'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir / 'finetune'),
            histogram_freq=1
        )
    ]

    # Train
    print(f"\nFine-tuning (epochs={config['epochs_finetune']}, lr={config['learning_rate_finetune']})...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs_finetune'],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=config['verbose']
    )

    # Save model
    save_model(model, model_dir, 'model_finetune', 'keras', class_id_map)

    print("\n✓ Fine-tuning complete!")
    return model


def evaluate_on_test_set(model, config: dict, class_id_map: dict):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        config: Configuration dict
        class_id_map: Class label mapping
    """
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60 + "\n")

    manifest_path = Path(config['manifest_output'])

    print(f"Loading test data...")
    test_dataset = create_dataset_from_manifest(
        str(manifest_path), 'test', class_id_map,
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        shuffle=False
    )

    print(f"\nEvaluating on test set...")
    results = model.evaluate(test_dataset, verbose=1)

    print(f"\nTest Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"  Top-5 Accuracy: {results[2]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 food classifier')
    parser.add_argument('--config', type=str, default='train/config.yaml',
                       help='Path to config file')
    parser.add_argument('--phase', type=str, choices=['head', 'finetune', 'both'],
                       default='head', help='Training phase')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config loaded: image_size={config['image_size']}, batch_size={config['batch_size']}")

    # Verify manifest exists
    if not Path(config['manifest_output']).exists():
        print(f"\n⚠ Dataset manifest not found at {config['manifest_output']}")
        print(f"Run: python train/prepare_data.py --config {args.config}")
        return

    # Train
    if args.phase == 'head' or args.phase == 'both':
        model, class_id_map = train_head(config)
        evaluate_on_test_set(model, config, class_id_map)

    if args.phase == 'finetune' or args.phase == 'both':
        model = train_finetune(config, args.checkpoint)
        evaluate_on_test_set(model, config, class_id_map)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()

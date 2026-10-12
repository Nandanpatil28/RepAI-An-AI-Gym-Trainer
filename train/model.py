"""
MobileNetV2 transfer-learning model for food classification.

Supports two training phases:
  1. Head-only: Pretrained base frozen, only classification head trained
  2. Fine-tune: Base layers unfrozen, entire model fine-tuned at lower learning rate
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


def create_mobilenetv2_model(num_classes: int,
                            image_size: tuple = (224, 224),
                            use_imagenet_weights: bool = True) -> tf.keras.Model:
    """
    Create MobileNetV2 model with classification head.

    Args:
        num_classes: Number of output classes
        image_size: Input image size (height, width)
        use_imagenet_weights: Use pretrained ImageNet weights

    Returns:
        tf.keras.Model: Uncompiled model
    """

    # Load pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights='imagenet' if use_imagenet_weights else None
    )

    # Freeze base model initially
    base_model.trainable = False

    # Create model
    inputs = layers.Input(shape=(*image_size, 3))

    # Preprocess input (ImageNet normalization)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Pass through base model
    x = base_model(x, training=False)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001):
    """
    Compile model with Adam optimizer and categorical crossentropy loss.

    Args:
        model: tf.keras.Model to compile
        learning_rate: Learning rate for Adam optimizer
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )


def unfreeze_base_for_finetuning(model: tf.keras.Model, num_layers_to_unfreeze: int = 50):
    """
    Unfreeze base model layers for fine-tuning.

    Args:
        model: Compiled model with frozen base
        num_layers_to_unfreeze: Number of layers from end of base to unfreeze
    """
    base_model = model.layers[2]  # Second layer is the MobileNetV2 base

    # Unfreeze the last num_layers_to_unfreeze layers
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"Unfroze {num_layers_to_unfreeze} layers of base model for fine-tuning")


def save_model(model: tf.keras.Model, model_dir: Path, model_name: str = 'model',
              format: str = 'keras', class_labels: dict = None):
    """
    Save model and class labels.

    Args:
        model: tf.keras.Model to save
        model_dir: Directory to save model
        model_name: Name prefix for saved model
        format: 'keras' (.keras) or 'savedmodel' (SavedModel format)
        class_labels: {class_name: class_id, ...} to save alongside
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if format == 'keras':
        model_path = model_dir / f'{model_name}.keras'
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    elif format == 'savedmodel':
        model_path = model_dir / model_name
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    else:
        raise ValueError(f"Unknown format: {format}")

    # Save class labels
    if class_labels:
        labels_path = model_dir / f'{model_name}_labels.json'
        with open(labels_path, 'w') as f:
            json.dump(class_labels, f, indent=2)
        print(f"✓ Class labels saved to {labels_path}")


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load saved model.

    Args:
        model_path: Path to .keras or SavedModel directory

    Returns:
        Loaded tf.keras.Model
    """
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Loaded model from {model_path}")
    return model


def load_class_labels(labels_path: str) -> dict:
    """
    Load class label mapping.

    Args:
        labels_path: Path to class_labels.json

    Returns:
        dict: {class_name: class_id, ...}
    """
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    print(f"✓ Loaded class labels from {labels_path}")
    return labels

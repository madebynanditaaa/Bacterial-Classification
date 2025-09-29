# src/dataloader.py
"""
Creates tf.data datasets from the folder splits created by dataset_split.py.

Functions:
    get_datasets(batch_size=32, img_size=(224,224), augment=True)
returns train_ds, val_ds, test_ds, class_names

Uses Keras image_dataset_from_directory + preprocessing layers for augmentation.
"""

import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from config import OUTPUT_SPLIT_DIR, IMG_SIZE, BATCH_SIZE, AUTOTUNE

def get_augmentation_pipeline():
    """
    Returns a tf.keras.Sequential of preprocessing layers to apply as augmentation.
    Layers are stateless and can be used inside the model or as preprocessing for datasets.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),  # ~30 degrees
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.05, 0.05),
            # Random contrast/brightness via Lambda because Keras lacks RandomBrightness
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.1)),
            layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.9, upper=1.1)),
        ],
        name="augmentation_pipeline",
    )


def preprocess_for_model(image, label):
    """
    image: uint8 [0..255]. Resize already applied by image_dataset_from_directory.
    Convert to float and apply MobileNetV2 preprocess_input (scales to [-1,1]).
    """
    image = tf.cast(image, tf.float32)
    image = mobilenet_preprocess(image)
    return image, label


def get_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=True, shuffle=True):
    """
    Loads datasets from OUTPUT_SPLIT_DIR with structure:
        OUTPUT_SPLIT_DIR/train/<class>/
        OUTPUT_SPLIT_DIR/val/<class>/
        OUTPUT_SPLIT_DIR/test/<class>/
    Returns: train_ds, val_ds, test_ds, class_names
    """

    train_dir = os.path.join(OUTPUT_SPLIT_DIR, "train")
    val_dir = os.path.join(OUTPUT_SPLIT_DIR, "val")
    test_dir = os.path.join(OUTPUT_SPLIT_DIR, "test")

    # Basic datasets (resizing done by the loader)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        seed=123,
        shuffle=shuffle,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Prefetch and map preprocessing
    AUTOTUNE_VAL = tf.data.experimental.AUTOTUNE if AUTOTUNE else None

    # Augmentation pipeline applied only to training
    augmentation = get_augmentation_pipeline() if augment else None

    def prepare(ds, training=False):
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE_VAL)
        if training and augmentation is not None:
            # augmentation expects images in [0,255], but layers operate on floats; keep range, apply augmentations then preprocess
            ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE_VAL)
        # apply model-specific preprocessing (e.g., mobilenet)
        ds = ds.map(preprocess_for_model, num_parallel_calls=AUTOTUNE_VAL)
        if training and shuffle:
            ds = ds.shuffle(1000, seed=123)
        ds = ds.prefetch(AUTOTUNE_VAL)
        return ds

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds, training=False)
    test_ds = prepare(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names

# src/models.py
"""
Model builders. Each function returns a compiled Keras model ready to train.

Supported model names:
    - mobilenet_v2
    - efficientnet_b0
    - resnet50 (Keras' ResNet50 used as ResNet variant)

You can swap the backbone by passing model_name to build_model().
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from config import IMG_SIZE, LEARNING_RATE

def build_mobilenet_v2(num_classes, input_shape=(224,224,3), fine_tune_at=100):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = True
    # Optionally freeze earlier layers
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="MobileNetV2_ft")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_efficientnet_b0(num_classes, input_shape=(224,224,3), fine_tune_at=None):
    base = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    if fine_tune_at is not None:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EfficientNetB0")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_resnet50(num_classes, input_shape=(224,224,3), fine_tune_at=None):
    base = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    if fine_tune_at is not None:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="ResNet50")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_model(model_name, num_classes, input_shape=(224,224,3), **kwargs):
    model_name = model_name.lower()
    if model_name == "mobilenet_v2" or model_name == "mobilenetv2":
        return build_mobilenet_v2(num_classes, input_shape, **kwargs)
    if model_name == "efficientnet_b0" or model_name == "efficientnetb0":
        return build_efficientnet_b0(num_classes, input_shape, **kwargs)
    if model_name == "resnet50" or model_name == "resnet":
        return build_resnet50(num_classes, input_shape, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")

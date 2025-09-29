# src/predict.py
"""
Single-image prediction script.

Usage:
    python src/predict.py --model_path results/models/mobilenet_v2_best.h5 --image path/to/img.jpg
"""

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

from dataloader import get_datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    return parser.parse_args()

def load_and_preprocess(img_path, target_size=(224,224)):
    img = keras_image.load_img(img_path, target_size=target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # MobileNetV2 preprocess (scales to [-1,1])
    return arr

def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    # Get class names from splits (assumes splits exist)
    _, _, _, class_names = get_datasets(batch_size=1, img_size=(224,224), augment=False)
    x = load_and_preprocess(args.image, target_size=(224,224))
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds, axis=1)[0])
    print(f"Predicted: {class_names[idx]} (confidence: {prob:.4f})")

if __name__ == "__main__":
    main()

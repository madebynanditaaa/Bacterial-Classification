# src/utils.py
import os
import random
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_json(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_history(history_dict, out_path):
    """
    history_dict: dict returned by Keras History.history
    Saves a plot of accuracy and loss.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10,4))

    # Accuracy
    plt.subplot(1,2,1)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="train_acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], label="train_loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

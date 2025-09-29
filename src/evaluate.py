# src/evaluate.py
"""
Evaluate a trained model on the test set and save confusion matrix and classification report.

Usage:
    python src/evaluate.py --model_path results/models/mobilenet_v2_best.h5
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from dataloader import get_datasets
from config import MODELS_DIR, PLOTS_DIR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, out_path, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)

    # Load datasets (no augmentation)
    train_ds, val_ds, test_ds, class_names = get_datasets(batch_size=args.batch_size, img_size=(224,224), augment=False)
    # Get true labels and predictions for test set
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification Report:\n", report)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, normalize=False)
    norm_cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_normalized.png")
    plot_confusion_matrix(cm, class_names, norm_cm_path, normalize=True)
    print(f"Saved confusion matrices to {PLOTS_DIR}")

    # Save report to file
    report_path = os.path.join(PLOTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

if __name__ == "__main__":
    main()

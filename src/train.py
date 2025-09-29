# src/train.py
"""
Train the model.

Usage examples:
    python src/train.py
    python src/train.py --model mobilenet_v2 --epochs 25 --batch_size 32

Outputs:
    - Best model saved under results/models/
    - TensorBoard logs under results/logs/
    - Training plots under results/plots/
"""

import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from config import DEFAULT_MODEL, EPOCHS, BATCH_SIZE, IMG_SIZE, MODELS_DIR, LOGS_DIR, PLOTS_DIR, SEED
from dataloader import get_datasets
from models import build_model
from utils import set_seed, plot_history, save_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="mobilenet_v2 | efficientnet_b0 | resnet50")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None, help="filename (without path) for best model")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)

    train_ds, val_ds, test_ds, class_names = get_datasets(batch_size=args.batch_size, img_size=IMG_SIZE, augment=True)
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(args.model, num_classes, input_shape=IMG_SIZE + (3,), fine_tune_at=100)

    # callbacks
    ckpt_name = args.checkpoint_name or f"{args.model}_best.h5"
    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    checkpoint_cb = ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    reduce_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    tb_dir = os.path.join(LOGS_DIR, args.model)
    tensorboard_cb = TensorBoard(log_dir=tb_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_cb, tensorboard_cb],
    )

    # Save final model and history
    final_path = os.path.join(MODELS_DIR, f"{args.model}_final.h5")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # save history and plots
    hist_path = os.path.join(PLOTS_DIR, f"{args.model}_history.json")
    save_json(history.history, hist_path)
    plot_history(history.history, os.path.join(PLOTS_DIR, f"{args.model}_training_plot.png"))

    # Optionally evaluate on test set and print metrics
    results = model.evaluate(test_ds)
    print("Test results:", results)

if __name__ == "__main__":
    main()

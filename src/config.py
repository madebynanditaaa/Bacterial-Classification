# src/config.py
import os

# === Paths (change if your repo root differs) ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATASET_DIR = os.path.join(ROOT, "Dataset")                # original data
AUG_DATASET_DIR = os.path.join(ROOT, "Dataset_Augmented")     # augmented copies
OUTPUT_SPLIT_DIR = os.path.join(ROOT, "data", "splits")      # train/val/test output

RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# === Image & training params ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
AUTOTUNE = True

# === Split proportions (train, val, test) ===
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15   # test = 1 - train - val

# === Training defaults ===
DEFAULT_MODEL = "resnet50"  # choices: mobilenet_v2, efficientnet_b0, resnet50
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

# Create dirs if missing
for d in [OUTPUT_SPLIT_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

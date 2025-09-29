# 🧫 Bacterial Image Classification Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project for classifying bacterial species from microscopy images using **MobileNetV2** and TensorFlow/Keras.

---

## 📌 Overview

This project uses **convolutional neural networks (CNNs)** to classify images of bacterial species.  
It supports:

✅ **Preprocessing & Augmentation** (resize, crop, flip, zoom)  
✅ **Transfer Learning** (MobileNetV2, ResNet, EfficientNet)  
✅ **Model Evaluation** (confusion matrix, classification report)  
✅ **Inference** on single images or batches  

---

## 🧬 Classes

The dataset is organized into subfolders representing bacterial species. Example classes:

- `Clostridium_perfringens`
- `Enterococcus_faecalis`
- `Enterococcus_faecium`
- `Escherichia_coli`
- `Listeria_monocytogenes`
- `Proteus`
- `Pseudomonas_aeruginosa`
- `Staphylococcus_aureus`

> Class names are automatically extracted from the dataset folder structure.

---

## 🏗️ Model

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Input Size:** 224×224 pixels
- **Training:** Fine-tuned final layers on custom bacterial dataset
- **Optimizer:** Adam (with learning rate scheduling)
- **Loss:** Categorical Crossentropy

---

## 📁 Directory Structure

```
Bacterial-Classification-Project/
│
├── data/
│ └── splits/
│ ├── train/
│ ├── val/
│ └── test/
│
├── results/
│ ├── models/
│ │ └── mobilenet_v2_best.h5
│ ├── plots/
│ └── logs/
│
├── src/
│ ├── config.py
│ ├── dataset_split.py
│ ├── dataloader.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
│
├── requirements.txt
└── README.md

```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Bacterial-Classification-Project.git
cd Bacterial-Classification-Project
```

###2️⃣ Create a virtual environment (recommended)

```bash
python -m venv env
# Activate it
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate
```
###3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
#4️⃣ Preprocess & split dataset

```bash
python src/dataset_split.py
```
###5️⃣ Train the model

```bash
python src/train.py
```
🔍 Inference (Predict on New Image)

You can classify a single image using predict.py:

```bash
python src/predict.py --model_path results/models/mobilenet_v2_best.h5 --image "data/splits/test/Clostridium_perfringens/01_aug2.jpg"
```
Example Output:

```bash
Predicted: Clostridium_perfringens (confidence: 0.9981)
```

📊 Model Evaluation

You can evaluate the trained model on the test set:
```bash
python src/evaluate.py --model_path results/models/mobilenet_v2_best.h5
```
This will output:

Accuracy
Precision, Recall, F1-score
Confusion matrix (optional: saved as plot in results/plots/)

⚙️ Requirements

```bash
Python ≥ 3.8
TensorFlow ≥ 2.6
NumPy
scikit-learn
Matplotlib
Pillow
OpenCV (optional, for additional augmentations)
```

Install them with:
```
pip install -r requirements.txt
```

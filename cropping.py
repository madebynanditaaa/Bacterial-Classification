import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

# Paths
augmented_dataset_path = "Dataset_Augmented"
cropped_dataset_path = "Dataset_Cropped_Improved"

os.makedirs(cropped_dataset_path, exist_ok=True)

# Parameters
MIN_SIZE = 32
OUTPUT_SIZE = 224
PADDING = 10  # add margin around bacteria crop
MIN_VARIANCE = 50  # filter very flat crops

folders = [f for f in os.listdir(augmented_dataset_path) if os.path.isdir(os.path.join(augmented_dataset_path, f))]
print("Found classes:", folders)

for folder in folders:
    class_input_path = os.path.join(augmented_dataset_path, folder)
    class_output_path = os.path.join(cropped_dataset_path, folder)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\nProcessing class: {folder}")
    images = glob.glob(os.path.join(class_input_path, "*.jpg")) + glob.glob(os.path.join(class_input_path, "*.png"))

    for img_path in tqdm(images, desc=f"Cropping {folder}"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Range for pink/magenta (adjust based on your data)
        lower_pink = np.array([140, 40, 40])  # lower HSV bound
        upper_pink = np.array([180, 255, 255])  # upper HSV bound
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_count = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_SIZE or h < MIN_SIZE:
                continue

            # Add padding but stay within image bounds
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(img.shape[1], x + w + PADDING)
            y2 = min(img.shape[0], y + h + PADDING)

            crop = img[y1:y2, x1:x2]

            # Filter by variance (skip flat regions)
            if crop.var() < MIN_VARIANCE:
                continue

            # Optional denoising
            crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)

            # Resize
            crop_resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)

            save_path = os.path.join(class_output_path, f"{img_name}_crop{crop_count}.jpg")
            cv2.imwrite(save_path, crop_resized)
            crop_count += 1

print("\n✅ Improved cropping complete!")
print(f"Cropped images saved to: {cropped_dataset_path}")

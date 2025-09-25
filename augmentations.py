import os
import cv2
import glob
from tqdm import tqdm
import albumentations as A

# Input and output paths
dataset_path = "Bacterial-Classification/Dataset"
augmented_dataset_path = "Bacterial-Classification/Dataset_Augmented"

# Define augmentation pipeline
augment = A.Compose([
    A.Rotate(limit=360, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ElasticTransform(alpha=40, sigma=5, alpha_affine=5, p=0.5),
])

# Make sure output directories exist
os.makedirs(augmented_dataset_path, exist_ok=True)

# List all class folders
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print("Found classes:", folders)

# Loop through each class folder
for folder in folders:
    class_input_path = os.path.join(dataset_path, folder)
    class_output_path = os.path.join(augmented_dataset_path, folder)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\nProcessing class: {folder}")
    images = glob.glob(os.path.join(class_input_path, "*.jpg")) + glob.glob(os.path.join(class_input_path, "*.png"))
    
    for img_path in tqdm(images, desc=f"Augmenting {folder}"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Generate multiple augmentations per image
        for i in range(10):  # adjust multiplier based on dataset size
            augmented = augment(image=img)
            aug_img = augmented["image"]
            save_path = os.path.join(class_output_path, f"{img_name}_aug{i}.jpg")
            cv2.imwrite(save_path, aug_img)

print("\n✅ Augmentation complete!")
print(f"Augmented dataset saved to: {augmented_dataset_path}")
print("You can now zip and upload this folder to your Google Drive.")

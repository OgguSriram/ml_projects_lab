import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter

TRAIN_DIR = "Training"

classes = os.listdir(TRAIN_DIR)

print("Classes found:", classes)

image_count = {}
sample_images = []

for cls in classes:
    cls_path = os.path.join(TRAIN_DIR, cls)
    images = os.listdir(cls_path)
    image_count[cls] = len(images)

    img_path = os.path.join(cls_path, images[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sample_images.append((cls, img))

print("\nImage count per class:")
for k, v in image_count.items():
    print(f"{k}: {v}")

# Plot class distribution
plt.figure(figsize=(6,4))
plt.bar(image_count.keys(), image_count.values())
plt.title("Class Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Show one sample image per class
plt.figure(figsize=(8,8))
for i, (cls, img) in enumerate(sample_images):
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()

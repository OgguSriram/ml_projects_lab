import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Dataset paths
base_dir = "chest_xray"
train_normal_dir = os.path.join(base_dir, "train", "NORMAL")
train_pneumonia_dir = os.path.join(base_dir, "train", "PNEUMONIA")

# Pick random images
normal_image = random.choice(os.listdir(train_normal_dir))
pneumonia_image = random.choice(os.listdir(train_pneumonia_dir))

# Load images
img_normal = Image.open(os.path.join(train_normal_dir, normal_image))
img_pneumonia = Image.open(os.path.join(train_pneumonia_dir, pneumonia_image))

# Show images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_normal, cmap="gray")
plt.title("NORMAL")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_pneumonia, cmap="gray")
plt.title("PNEUMONIA")
plt.axis("off")

plt.show()

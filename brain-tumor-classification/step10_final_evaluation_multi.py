import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# CONFIG
# ======================
MODEL_PATH = "brain_tumor_resnet50_finetuned.keras"
TEST_DIR = "Testing"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ======================
# TEST DATA GENERATOR (MULTI-IMAGE)
# ======================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False   # VERY IMPORTANT
)

# ======================
# PREDICTIONS
# ======================
pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# ======================
# CLASSIFICATION REPORT
# ======================
print("\nFINAL CLASSIFICATION REPORT (MULTI-IMAGE TEST SET):\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Final Confusion Matrix – ResNet50 Fine-Tuned")
plt.tight_layout()
plt.show()

# ======================
# NORMALIZED CONFUSION MATRIX (OPTIONAL BUT STRONG)
# ======================
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix – ResNet50 Fine-Tuned")
plt.tight_layout()
plt.show()

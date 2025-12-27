import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("pneumonia_model.keras")

# Image settings
IMG_SIZE = (224, 224)
IMAGE_PATH = "test_xray.jpg"   # already copied

# Load and preprocess image
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output result
if prediction > 0.5:
    print(f"PREDICTION: PNEUMONIA ({prediction:.2f} confidence)")
else:
    print(f"PREDICTION: NORMAL ({1 - prediction:.2f} confidence)")

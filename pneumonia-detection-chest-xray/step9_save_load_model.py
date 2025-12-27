from tensorflow.keras.models import load_model

# Save model in NEW Keras format
model = load_model("pneumonia_mobilenetv2.h5")
model.save("pneumonia_model.keras")

print("Model saved in Keras format (.keras)")

# Load model again
loaded_model = load_model("pneumonia_model.keras")
print("Model loaded successfully")

# Print model summary to confirm
loaded_model.summary()

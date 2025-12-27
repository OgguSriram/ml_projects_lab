from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training data generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Validation data generator (NO augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    "chest_xray/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    "chest_xray/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Class indices:", train_generator.class_indices)

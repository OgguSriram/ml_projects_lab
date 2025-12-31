# Brain Tumor Classification – v2 (Improved ResNet50)

## Overview
This is the improved version of the brain tumor classification project.
It uses a larger, balanced MRI dataset and a fine-tuned ResNet50 model.

## Dataset
- BRISC2025
- 6,000 MRI images
- Axial / Coronal / Sagittal planes
- 4 classes: Glioma, Meningioma, Pituitary, No Tumor

## Model
- ResNet50 (ImageNet pretrained)
- Fine-tuned last 80 layers
- EarlyStopping & ReduceLROnPlateau

## Performance
- **Test Accuracy: 79%**
- High recall for No Tumor and Pituitary classes

## How to Run
```bash
pip install -r requirements.txt
python src/train_resnet50.py
python src/test_resnet50.py

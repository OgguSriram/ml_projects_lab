# 🧠 Brain Tumor Classification Using Deep Learning

This project implements an end-to-end deep learning pipeline for classifying brain MRI images into multiple tumor categories using Convolutional Neural Networks and transfer learning.

---

## 📌 Problem Statement
Manual diagnosis of brain tumors from MRI scans is time-consuming and error-prone. This project aims to automate brain tumor classification using deep learning techniques.

---

## 📂 Dataset
- Brain MRI image dataset
- 4 classes:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- Total images: ~3,200
- Dataset not uploaded due to size constraints

---

## 🧠 Models Implemented
- Custom CNN (baseline)
- ResNet50 (Transfer Learning)
- ResNet50 Fine-Tuned (Final Model)
- EfficientNetB0 (Experimental – lower performance)

---

## 🏆 Final Model
**ResNet50 Fine-Tuned**

- Validation Accuracy: **46%**
- Best generalization among tested models

---

## 📊 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Normalized Confusion Matrix

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

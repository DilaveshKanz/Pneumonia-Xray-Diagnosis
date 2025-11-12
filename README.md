# 🩻 Pneumonia X-Ray Diagnosis (VGG16 CNN)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-yellow?style=for-the-badge&logo=gradio)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-89.90%25-brightgreen?style=for-the-badge)

A high-performance, VGG16-based Convolutional Neural Network built with TensorFlow to diagnose pneumonia from chest X-ray images. This model achieves **89.90% test accuracy** using a robust, trustworthy validation strategy.

This project was built as part of my portfolio for application to MBZUAI.

---

## 🏆 Final Results: 89.90% Accuracy

This model excels at its most critical task: **identifying pneumonia (96% Recall)**, while maintaining a strong balance in identifying normal cases (79% Recall).

| Training History | Confusion Matrix |
| :---: | :---: |
| ![Training History](training_history_VGG16.png) | ![Final Confusion Matrix](confusion_matrix_VGG16.png) |

---

## 🔬 The Strategy: Why This Model is Robust

Many public notebooks on this dataset achieve 92-95% accuracy by making a critical error: they validate against the tiny, 16-image `val` folder. These scores are "flukes" and are not reproducible.

My approach was to build a **scientifically sound and reliable model**.

1.  **A Robust Validation Set:** I ignored the 16-image `val` folder and created a proper **80/20 validation split** (1,043 images) from the main 5,216-image `train` set.
2.  **State-of-the-Art Architecture:** I used **Transfer Learning** with the proven VGG16 model, freezing the base and adding a custom classification head with `Dropout(0.5)` to prevent overfitting.
3.  **Handling Imbalance:** The model was trained using `class_weight` to teach it to pay closer attention to the minority "NORMAL" class.
4.  **Finding the Best Model:** `EarlyStopping` (monitoring `val_loss`) was used to automatically find the model's peak performance and prevent overfitting.

This **89.90%** score is a trustworthy, reproducible benchmark of what this architecture can achieve.

---

## 🚀 Quickstart

This repository is fully self-contained. The scripts will automatically download the dataset and install dependencies.

### 1. Clone the Repository
```bash
# Remember to change "your-username" to your actual GitHub username
git clone [https://github.com/your-username/Pneumonia-Xray-Diagnosis-CNN.git](https://github.com/your-username/Pneumonia-Xray-Diagnosis-CNN.git)
cd Pneumonia-Xray-Diagnosis-CNN

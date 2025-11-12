# 🩻 X-Ray Pneumonia Diagnosis (90% Accuracy)

This is a deep learning project built as part of my portfolio for the MBZUAI application. The model is a **VGG16** Convolutional Neural Network (CNN) that can diagnose pneumonia from chest X-ray images with **89.90% accuracy**.

### 🏆 Final Results
* **Test Accuracy:** 89.90%
* **Pneumonia Recall:** 96% (The model correctly identifies 96% of all pneumonia cases)
* **Normal Recall:** 79%

![Final Confusion Matrix](confusion_matrix_VGG16.png)
![Training History](training_history_VGG16.png)

---

## 🔬 Methodology

This project was built using a robust, professional machine learning workflow to ensure the final score was trustworthy and not a "fluke."

1.  **Data Strategy:** The original dataset's `val` folder (16 images) was identified as statistically useless. To fix this, I created a robust **80/20 validation split** (1,043 images) from the `train` folder.
2.  **Model Architecture:** I used **Transfer Learning** with the VGG16 model, freezing the base and adding a custom classification head with `Dropout(0.5)` to prevent overfitting.
3.  **Training:** The model was trained using `class_weight` to handle the imbalanced dataset.
4.  **Validation:** `EarlyStopping` (monitoring `val_loss`) was used to automatically find the model's peak performance and prevent overfitting.

---

## 🚀 How to Run This Project

### 1. Prerequisites
* Python 3.8+
* TensorFlow
* Gradio
* Kaggle Account (for the dataset)

### 2. Set Up the Project
```bash
# Clone the repository
git clone [https://github.com/your-username/X-Ray-Diagnosis-Project.git](https://github.com/your-username/X-Ray-Diagnosis-Project.git)
cd X-Ray-Diagnosis-Project

# Install all required libraries
pip install -r requirements.txt

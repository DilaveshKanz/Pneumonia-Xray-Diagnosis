<div align="center">

# Pneumonia X-Ray Diagnosis (VGG16 CNN)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-FFA000?style=for-the-badge&logo=gradio&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-89.90%25-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A VGG16-based Convolutional Neural Network for automated pneumonia diagnosis from chest X-ray images.**

[🚀 Live Demo](#-live-demo) • [📖 Documentation](#-the-strategy-why-this-model-is-robust) • [⚡ Quick Start](#-quickstart) • [📊 Results](#-final-results-8990-accuracy)

---

</div>

## 🎯 Overview

A production-ready deep learning model built with TensorFlow that achieves **89.90% test accuracy** for pneumonia detection in chest X-rays. This project emphasizes scientific rigor and reproducibility, using proper validation techniques and transfer learning with VGG16.

### ✨ Key Highlights

- 🎯 **96% Recall** on pneumonia cases (critical for medical diagnosis)
- 🏗️ **Transfer Learning** with VGG16 architecture
- ⚖️ **Class-weighted Training** to handle dataset imbalance
- 🔬 **Scientifically Sound Validation** (avoiding common pitfalls)
- 🌐 **Interactive Web Interface** with Gradio
- 📦 **Fully Automated Pipeline** (dataset download → training → deployment)

---

## 🚀 Live Demo

**[View Live Demo on Hugging Face Spaces](#-https://huggingface.co/spaces/Dilavesh/Pneumonia-Xray-Diagnosis)**

---

## 📊 Final Results: 89.90% Accuracy

<div align="center">

| Metric | Normal | Pneumonia | Overall |
|--------|--------|-----------|---------|
| **Precision** | 94% | 88% | - |
| **Recall** | 79% | 96% | - |
| **F1-Score** | 86% | 92% | - |
| **Accuracy** | - | - | **89.90%** |

</div>

### 🎯 Why These Numbers Matter

The model excels at its **most critical task**: identifying pneumonia with **96% recall**, ensuring minimal false negatives in medical diagnosis. The strong 94% precision on normal cases minimizes unnecessary follow-up procedures.

---

## 🔬 The Strategy: Why This Model is Robust

> **Most public notebooks achieve 92-95% accuracy on this dataset. Why is mine different?**


#### 🔑 Key Design Decisions

1. **📊 Proper Validation Split**
   - Created an 80/20 split from the 5,216-image training set
   - 1,043 validation images ensure statistically significant results
   - Ignored the unreliable 16-image validation folder

2. **🏗️ State-of-the-Art Architecture**
   - Transfer learning with **VGG16** (ImageNet pre-trained weights)
   - Frozen base layers preserve learned features
   - Custom classification head with `Dropout(0.5)` prevents overfitting

3. **⚖️ Handling Class Imbalance**
   - Dataset contains 3:1 ratio of pneumonia to normal cases
   - Applied `class_weight` to emphasize minority class learning
   - Ensures balanced performance across both classes

4. **🎓 Smart Training Strategy**
   - `EarlyStopping` monitors `val_loss` to prevent overfitting
   - Automatically saves best model checkpoint
   - Reproducible results with controlled randomness

> **The 89.90% score is a trustworthy, reproducible benchmark of this architecture's real-world performance.**

---



## 📊 Dataset

**Chest X-Ray Images (Pneumonia)**

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,856 JPEG files
- **Classes**: NORMAL vs PNEUMONIA (bacterial & viral)
- **Resolution**: Variable (resized to 224×224 for VGG16)
- **Auto-download**: The `train_and_evaluate.py` script automatically fetches the dataset via KaggleHub

### Dataset Composition

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| **Training** (original) | 1,341 | 3,875 | 5,216 |
| **Testing** | 234 | 390 | 624 |
| **My Split (80/20)** | | | |
| └─ Train | 1,073 | 3,100 | 4,173 |
| └─ Validation | 268 | 775 | 1,043 |

---

## ⚡ Quickstart

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Kaggle API credentials for dataset download

### 🚀 Installation & Training

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Pneumonia-Xray-Diagnosis-CNN.git
cd Pneumonia-Xray-Diagnosis-CNN
```

#### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Train & Evaluate the Model

```bash
python train_and_evaluate.py
```

**What this does:**
- ⬇️ Automatically downloads the Kaggle dataset
- 🔄 Creates proper 80/20 train/validation split
- 🏋️ Trains VGG16 model with EarlyStopping
- 💾 Saves best model as `pneumonia_VGG16_best.keras`
- 📊 Generates accuracy reports and visualizations
- ⏱️ **Estimated time**: 30-45 minutes on GPU

#### 4️⃣ Launch Interactive Web App

```bash
python app.py
```

Then open your browser to `http://localhost:7860` to test the model with your own X-ray images!

---

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) |
| **Web Interface** | ![Gradio](https://img.shields.io/badge/Gradio-FFA000?style=flat-square&logo=gradio&logoColor=white) |
| **Data Science** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square) |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) |

</div>

### Core Dependencies

- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural networks API
- **Gradio** - Interactive ML web interface
- **Scikit-learn** - Class weights and evaluation metrics
- **KaggleHub** - Automatic dataset downloading
- **OpenCV** - Image preprocessing
- **Matplotlib & Seaborn** - Results visualization



</div>

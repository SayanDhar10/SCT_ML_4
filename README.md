# ✋ Hand Gesture Recognition using CNN + PyTorch

This repository implements a multi-class hand gesture classification system using deep learning. The model is trained on the **LeapGestRecog** dataset and leverages a **Convolutional Neural Network (CNN)** built using **PyTorch**. It also includes **feature-space visualization** using **UMAP**, evaluation metrics, and confidence-based predictions.

---

## 📌 Task Objective

> Classify grayscale hand gesture images into one of 10 categories using a custom CNN trained from scratch.

---

## 🔧 Features

- 🔹 Custom PyTorch Dataset class for LeapGestRecog  
- 🔹 CNN model trained on gesture images  
- 🔹 Train-test split using PyTorch DataLoader  
- 🔹 Real-time accuracy and loss tracking  
- 🔹 Evaluation with confusion matrix, classification report  
- 🔹 Visualization of:  
  - UMAP feature embeddings  
  - Per-class accuracy  
  - Misclassified images  
  - Prediction with confidence scores  

---

## 🛠️ Workflow

📌 Data Preprocessing 🧹
- Images resized to **64×64 grayscale**
- Normalized to **[-1, 1]**
- Labels extracted from **folder names**
- Train-test split using `train_test_split`

🧠 Model Architecture
**Custom CNN:**
- 3 Convolutional layers + ReLU + MaxPooling
- Fully connected classifier
- Output layer with **10 neurons** + Softmax

 🔁 Training & Evaluation
- **Optimizer**: Adam  
- **Loss Function**: CrossEntropy  
- **Epochs**: 15  
- Accuracy and loss **tracked per epoch**  
- **UMAP** applied on CNN features


 📊 Visualization & Interpretation
- Confusion Matrix (Heatmap)
- Loss & Accuracy curves
- UMAP 2D feature projection
- Per-class performance plots
- Predicted images with confidence scores

---

## 🧠 Methodology

🔸 Transformations
- Resize to **(64×64)**
- `ToTensor()`  
- Normalize with `mean=0.5`, `std=0.5`


🔸 Model Training
- Custom `train()` and `evaluate()` loops
- Accuracy and loss captured per epoch

🔸 Prediction Confidence
- `Softmax` used to extract **probabilities**
- Top predicted class shown with **confidence %**


🔸 Feature Visualization
- **umap-learn** applied on CNN features
- 2D embedding shows **clustering per gesture class**


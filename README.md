# Task-02-Tayyba-Yahya

## Iris Flower Classification using K-Nearest Neighbors

This project demonstrates a basic **Machine Learning classification pipeline** using the built-in **Iris dataset** from `scikit-learn`.

The model uses **K-Nearest Neighbors (KNN)** to classify Iris flowers into three species based on their sepal and petal measurements.

---

## 🌸 Project Overview

The Iris dataset contains measurements for three different Iris flower species:

- **Setosa**
- **Versicolor**
- **Virginica**

Each flower is described using four features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal |
| Sepal Width | Width of the sepal |
| Petal Length | Length of the petal |
| Petal Width | Width of the petal |

The target values are represented numerically:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

---

## 🛠️ Technologies Used

- **Python**
- **Scikit-learn**
- **NumPy**
- **K-Nearest Neighbors (KNN)**
- **StandardScaler**
- **Train/Test Split**
- **Classification Metrics**

---

## 📂 Project Structure

```text
Task-02-Tayyba-Yahya/
│
├── iris_project.py
└── README.md
```

---

## 🔄 Machine Learning Pipeline

The project follows the standard supervised machine learning workflow:

```text
🌸 Iris Dataset
       ↓
Load Dataset
       ↓
Separate Features (X) and Target (y)
       ↓
Train / Test Split
       ↓
Standardization
       ├── Fit on Training Data
       └── Transform Test Data
       ↓
KNN Classifier
       ↓
Train Model
       ↓
Make Predictions
       ↓
Evaluate Model
       ├── Accuracy
       ├── Confusion Matrix
       └── Classification Report
```

### Why split before scaling?

The scaler should learn the mean and standard deviation **only from the training data**. The same learned scaling parameters are then applied to the test data.

This prevents information from the test set from leaking into the training process.

---

## 📦 Libraries

The project imports the following tools:

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tayyba7/Task-02-Tayyba-Yahya.git
```

### 2. Open the project

```bash
cd Task-02-Tayyba-Yahya
```

### 3. Install dependencies

```bash
pip install scikit-learn numpy
```

### 4. Run the program

```bash
python iris_project.py
```

---

## 🤖 KNN Model

The project uses:

```python
KNeighborsClassifier(n_neighbors=5)
```

This means the classifier considers the **5 nearest training samples** when predicting the species of a new flower.

---

## 📊 Model Evaluation

The model evaluates its predictions using three main metrics.

### Accuracy

Accuracy measures the overall percentage of correct predictions.

```python
accuracy_score(y_test, predictions)
```

### Confusion Matrix

The confusion matrix shows how many samples from each class were correctly or incorrectly classified.

```python
confusion_matrix(y_test, predictions)
```

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

```python
classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
)
```

---

## 🎯 Learning Objectives

This task demonstrates the basic concepts of:

- Loading a dataset using Scikit-learn
- Understanding features and targets
- Splitting data into training and testing sets
- Standardizing numerical features
- Training a KNN classifier
- Making predictions
- Evaluating a classification model
- Understanding a complete machine learning pipeline

---

## 👩‍💻 Author

**Tayyba Yahya**

GitHub: [@tayyba7](https://github.com/tayyba7)

Repository: [Task-02-Tayyba-Yahya](https://github.com/tayyba7/Task-02-Tayyba-Yahya)

---

## 📄 License

This project is created for educational and learning purposes.

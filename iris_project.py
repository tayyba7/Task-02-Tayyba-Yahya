# imports - getting the tools we need
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

# 1. Load Dataset
# Iris dataset is already available in sklearn
iris = load_iris()

print("Dataset loaded successfully")

# 2. Separate Features (X) and Target (y)
# X contains the flower measurements
# y contains the flower species
X = iris.data
y = iris.target

print("Shape of data:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Class balance:", np.unique(y, return_counts=True))
print("First 5 rows:\n", X[:5])

# 3. Train / Test Split
# 80% of the data is used for training
# 20% is used for testing
# stratify keeps the same class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 4. Standardization
# Scale the data so all features are on a similar range
# We fit the scaler only on the training data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining data after scaling:\n", X_train[:5])
print("\nTest data after scaling:\n", X_test[:5])

# 5. KNN Classifier
# KNN checks the 5 closest data points to make a prediction
model = KNeighborsClassifier(n_neighbors=5)

# 6. Train Model
# Train the model using the training data
model.fit(X_train, y_train)

print("\nModel trained successfully")

# 7. Make Predictions
# Use the trained model to predict the test data
predictions = model.predict(X_test)

print("\nActual:   ", y_test)
print("Predicted:", predictions)

# 8. Evaluate Model
# Accuracy shows how many predictions were correct
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Confusion matrix shows which classes were predicted correctly
# and which ones were confused with each other
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification report gives precision, recall and F1-score
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)

# Testing different values of k
# This is an extra experiment to see which k gives better accuracy
print("\n--- Testing different k values ---")

for k in range(1, 15):
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train, y_train)
    accuracy = temp_model.score(X_test, y_test)
    print(f"k={k}, accuracy={accuracy:.3f}")

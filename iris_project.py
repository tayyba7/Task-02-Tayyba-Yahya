# imports - basically grabbing all the tools i need first
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np


# loading the iris data, this is built into sklearn so no download needed
iris = load_iris()

X = iris.data      # the measurements - sepal/petal length and width
y = iris.target    # the actual species, but as numbers (0,1,2)

# just checking what we're working with before doing anything else
print("Shape of data:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Class balance:", np.unique(y, return_counts=True))  # making sure it's 50/50/50
print("First 5 rows:\n", X[:5])


# scaling the data so no single measurement dominates just bc its numbers are bigger
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nAfter scaling:\n", X_scaled[:5])  # numbers should look smaller now, some negative


# splitting into train/test - basically study material vs surprise exam
# shuffle=True bc the data is sorted by species and we don't want a biased split
# stratify=y keeps the species ratio equal in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# building the knn model - it'll check the 5 closest flowers to make a guess
model = KNeighborsClassifier(n_neighbors=5)

# training it - basically showing it the answers so it can learn the pattern
model.fit(X_train, y_train)

# now the actual test - predicting species for flowers it hasn't seen
predictions = model.predict(X_test)

print("\nActual:   ", y_test)
print("Predicted:", predictions)


# time to check how good it actually did
print("\nAccuracy:", accuracy_score(y_test, predictions))

# this shows exactly what got mixed up with what, not just a percentage
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

# breaks it down per species so we can see if one type is harder to guess
print("\nClassification Report:\n",
      classification_report(y_test, predictions, target_names=iris.target_names))


# just messing around here - testing different k values to see which works best
# instead of blindly going with k=5
print("\n--- trying different k values ---")
for k in range(1, 15):
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train, y_train)
    acc = temp_model.score(X_test, y_test)
    print(f"k={k}, accuracy={acc:.3f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
# Create the absolute path to the CSV file relative to the script's location
csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataSets_Required', 'Car_dataset.csv'))
df = pd.read_csv(csv_file_path)
print(df.head())

# df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")

# df.head()

df.info()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df = df.sample(n=20,random_state=42)
df

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='AnnualSalary', hue='Purchased', data=df, palette='coolwarm', s=100)
plt.title('Age vs Annual Salary (Colored by Purchased)')
plt.xlabel('Age')
plt.ylabel('Annual Salary')
plt.grid(True)
plt.show()

X = df[['Age', 'AnnualSalary']].values
y = df['Purchased'].values

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM classifier (linear kernel)
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_scaled, y)

# Function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 6))
    
    # Scatter the original points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100, edgecolor='k')
    
    # Create grid to evaluate model
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.title('SVM Decision Boundary with Age and Annual Salary')
    plt.xlabel('Age (Scaled)')
    plt.ylabel('Annual Salary (Scaled)')
    plt.grid(True)
    plt.show()

# Plot the data and SVM decision boundary
plot_decision_boundary(X_scaled, y, svm_clf)


# # SVM code mathematically

C = 1.0  # Regularization parameter
eta = 0.000001  # Learning rate
n_iter = 1000  # Number of iterations

n_samples, n_features = X.shape
w = np.zeros(n_features)
b = 0

for _ in range(n_iter):
    for idx, x_i in enumerate(X):
        if y[idx] * (np.dot(x_i, w) + b) >= 1:
            w -= eta * (2 * 1/n_iter * w)  # Update only weights
        else:
            w -= eta * (2 * 1/n_iter * w - np.dot(x_i, y[idx]))
            b -= eta * y[idx]

print(f"Weights: {w}")
print(f"Bias: {b}")

def predict(X):
    return np.sign(np.dot(X, w) + b)

predictions = predict(X)
print(f"Predictions: {predictions}")
print(f"Actual: {y}")

from sklearn.metrics import accuracy_score
acc = accuracy_score(y, predictions)
print(f"Accuracy: {acc * 100:.2f}%")


# # SVM code using library

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
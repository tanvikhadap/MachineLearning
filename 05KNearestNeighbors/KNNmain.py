import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
# Create the absolute path to the CSV file relative to the script's location
csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataSets_Required', 'Car_dataset.csv'))
df = pd.read_csv(csv_file_path)

# df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")

df = df.drop(columns=['User ID'])
df.head()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='AnnualSalary', hue='Purchased', data=df, palette='deep')
plt.title('Scatter Plot of Standardized Age vs Annual Salary (Entire Dataset)')
plt.xlabel('Age (Standardized)')
plt.ylabel('Annual Salary (Standardized)')
plt.show()


# # KNN code mathematically 

from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        predictions = [self._predict_single_point(x) for x in X_test]
        return np.array(predictions)

    def _predict_single_point(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNN(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print("Predictions:", predictions)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # KNN code using library

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)

scores = []

for i in range(1,16):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    
    knn.fit(X_train,y_train)
    
    y_pred = knn.predict(X_test)

    scores.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(range(1,16),scores)
plt.ylim(0.900,0.920)
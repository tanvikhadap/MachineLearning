import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")

df = df.drop(columns=['User ID'])
df.head()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.shape

def sample_rows(df,percent):
  return df.sample(int(percent*df.shape[0]),replace=True)

df1 = sample_rows(df,0.5)

df1

df2 = sample_rows(df,0.5)
df3 = sample_rows(df,0.5)

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:3],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:3],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:3],df3.iloc[:,-1])

from sklearn.tree import plot_tree

plot_tree(clf1)

plot_tree(clf2)

plot_tree(clf3)

clf1.predict(np.array([1,28,91500]).reshape(1,3))

clf2.predict(np.array([1,28,91500]).reshape(1,3))

clf3.predict(np.array([1,28,91500]).reshape(1,3))

input_values = np.array([1, 28, 91500]).reshape(1, 3)

prediction1 = clf1.predict(input_values)
prediction2 = clf2.predict(input_values)
prediction3 = clf3.predict(input_values)

average_prediction = np.mean([prediction1, prediction2, prediction3])

print(f"Aggregated prediction (average): {average_prediction}")


# # For 100 Dataset 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import mode

def sample_rows(df, percent):
    return df.sample(int(percent * df.shape[0]), replace=True)

datasets = []

for i in range(100):
    df_sample = sample_rows(df, 0.5)
    datasets.append(df_sample)

X = df.drop(columns=['Purchased'])  # Replace 'Purchased' with your actual target column name
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = []

for i in range(100):
    clf = DecisionTreeClassifier(random_state=i)
    clf.fit(X_train, y_train)
    classifiers.append(clf)

input_values = np.array([1, 28, 91500]).reshape(1, 3)

all_predictions = []

for clf in classifiers:
    pred = clf.predict(input_values)
    all_predictions.append(pred)

all_predictions = np.array(all_predictions)

final_prediction, _ = mode(all_predictions)

print(f"Aggregated prediction (majority vote): {final_prediction[0]}")
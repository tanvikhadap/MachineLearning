import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")

df

df.info()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df = df.sample(n=100,random_state=42)
df

# Create scatter plot
plt.figure(figsize=(10, 6))

# Plot 'Age' vs 'Purchased'
sns.scatterplot(x='Age', y='Purchased', hue='Gender', size='AnnualSalary', sizes=(20, 200), data=df)

# Customize plot
plt.title('Age, Gender, and AnnualSalary vs Purchased')
plt.xlabel('Age')
plt.ylabel('Purchased')
plt.legend(title='Gender', loc='upper left', labels=['Female', 'Male'])
plt.grid(True)
plt.show()

# Split into features and target
X = df[['Gender', 'Age', 'AnnualSalary']].values
y = df['Purchased'].values

# Define the Gini index function
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Define the function to split the dataset
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Function to select the best split
def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Function to create a terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Recursive function to split nodes
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Function to build the decision tree
def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Function to make a prediction with the decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Function to calculate accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Function to evaluate the decision tree algorithm
def evaluate_algorithm(dataset, max_depth, min_size):
    # Train and test on the same dataset (can split if desired)
    train, test = dataset, dataset
    tree = build_tree(train, max_depth, min_size)
    predictions = [predict(tree, row) for row in test]
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy

# Parameters for decision tree
max_depth = 3
min_size = 1

# Run the decision tree and get accuracy
accuracy = evaluate_algorithm(dataset, max_depth, min_size)
print(f"Decision Tree Accuracy: {accuracy:.2f}%")

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")
df


# # Decison tree code using libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']

clf = DecisionTreeClassifier()

clf.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data: {:.2f}%".format(accuracy * 100))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased'])
print("\nClassification Report:")
print(report)

#entire dataset
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, feature_names=['Gender', 'Age', 'AnnualSalary'], 
               class_names=['Not Purchased', 'Purchased'], filled=True, rounded=True)
plt.title("Decision Tree for Purchase Prediction")
plt.show()

#One branch 
plt.figure(figsize=(12, 8))
plot_tree(tree_clf, feature_names=['Gender', 'Age', 'AnnualSalary'], class_names=['Not Purchased', 'Purchased'], filled=True)
plt.show()

# Extract rules of a specific branch
n_nodes = tree_clf.tree_.node_count
children_left = tree_clf.tree_.children_left
children_right = tree_clf.tree_.children_right
feature = tree_clf.tree_.feature
threshold = tree_clf.tree_.threshold

# Function to print rules of a specific node
def print_rule(node, depth=0):
    indent = "    " * depth
    if children_left[node] != children_right[node]:
        # Continue to left or right child
        print(f"{indent}if {X.columns[feature[node]]} <= {threshold[node]:.2f}:")
        print_rule(children_left[node], depth + 1)
        print(f"{indent}else:  # if {X.columns[feature[node]]} > {threshold[node]:.2f}")
        print_rule(children_right[node], depth + 1)
    else:
        # Leaf node
        print(f"{indent}Predict: {np.argmax(tree_clf.tree_.value[node][0])} (Class {tree_clf.classes_[np.argmax(tree_clf.tree_.value[node][0])]})")

# Print the rules for a specific branch
print("Decision rules for one branch:")
print_rule(0)  # Start at the root node (node 0)
# Machine Learning Algorithms: From Scratch and Using Libraries

This project explores the implementation of various machine learning algorithms both from scratch and with the use of standard libraries (like scikit-learn). By comparing the two approaches, this project highlights the inner workings of these algorithms and their efficiency when applied using well-established libraries.

## Project Structure

```
.
├── from_scratch/
│   ├── linear_regression.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── knn.py
│   └── ...
├── using_libraries/
│   ├── linear_regression.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── knn.py
│   └── ...
└── README.md
```

## Algorithms Implemented

### 1. Linear Regression
- **From Scratch**: Implemented using mathematical operations for gradient descent, loss calculation, and prediction.
- **Using Libraries**: Implemented using `scikit-learn`'s `LinearRegression` class.

### 2. Support Vector Machine (SVM)
- **From Scratch**: Implemented by solving the optimization problem manually.
- **Using Libraries**: Implemented using `scikit-learn`'s `SVC` class.

### 3. Decision Tree
- **From Scratch**: Built a decision tree by splitting the data based on the Gini index.
- **Using Libraries**: Implemented using `scikit-learn`'s `DecisionTreeClassifier`.

### 4. K-Nearest Neighbors (KNN)
- **From Scratch**: Manually calculated the Euclidean distance between data points to classify the nearest neighbors.
- **Using Libraries**: Implemented using `scikit-learn`'s `KNeighborsClassifier`.

## Datasets

The project uses the following datasets:
1. **Supply Chain Dataset**: Columns include 'temp,' 'out/in,' 'year,' 'month,' 'day,' 'hour,' and more.
2. **Classification Dataset**: Columns include 'User ID,' 'Gender,' 'Age,' 'AnnualSalary,' and 'Purchased.'

## Getting Started

### Prerequisites
- Python 3.x
- For running the algorithms implemented using libraries, install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:
- `scikit-learn`
- `numpy`
- `matplotlib` (for plotting, if applicable)

### Running the Code

#### From Scratch Implementations
To run the linear regression model from scratch:

```bash
python from_scratch/linear_regression.py
```

You can similarly run other algorithms from the `from_scratch` folder.

#### Using Libraries Implementations
To run the SVM using scikit-learn:

```bash
python using_libraries/svm.py
```

## Results and Comparison

Each algorithm implementation outputs the following:
- **From Scratch**: A detailed breakdown of how the algorithm processes the data step-by-step, including custom loss functions, gradient calculations, and final predictions.
- **Using Libraries**: A faster and more efficient solution that leverages optimized code.

You can compare the results in terms of:
- Training time
- Accuracy
- Prediction time

## Conclusion

This project demonstrates:
- The importance of understanding the mechanics of machine learning algorithms.
- How libraries abstract away complexity, providing efficient implementations suitable for large-scale projects.

---

Let me know if you'd like more specific content or further modifications!


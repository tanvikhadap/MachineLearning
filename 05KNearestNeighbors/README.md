---

# K-Nearest Neighbors (KNN) Implementation: From Scratch vs Using Libraries

## What is it?
This project demonstrates the implementation of a K-Nearest Neighbors (KNN) classifier in two ways
1. **From scratch:** Building a KNN classifier without using any external libraries.
2. **Using libraries:** Utilizing Python’s `scikit-learn` library to implement a KNN classifier.

The goal is to compare these approaches, showing how KNN works mathematically and how to use pre-built library methods for ease and efficiency.

## Why should I make it?
Understanding the inner workings of machine learning algorithms is critical for building a deeper knowledge of how they operate. By comparing a manual KNN implementation with a library-based one, this project helps clarify the algorithm's mechanics and its practical usage in real-world applications.

## Who should make it?
This project is designed for
- Machine learning practitioners
- Data science students or professionals
- Anyone interested in understanding the difference between custom implementations and using libraries for KNN.

## When should I make it?
You should create this README before sharing or publishing the project to ensure that collaborators and users can easily understand the purpose and functionality of the code.

## Where should I put it?
Place the README file in the top-level directory of your project. This ensures that anyone new to the project can quickly find out what it's about and how to use it. Git hosting platforms like GitHub, GitLab, and Bitbucket will display the README file by default when visiting the repository.

## How should I make it?
This README uses **Markdown**, which allows for easy formatting and readability. Below is a breakdown of how the project works

---

## Project Structure

- **knn_scratch.py**: Implements KNN from scratch using pure Python.
- **knn_libraries.py**: Implements KNN using the `scikit-learn` library.
- **car_data.csv**: The dataset used for KNN classification.

---

## How to Install and Run

### Prerequisites
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/knn-comparison.git
   ```

2. Navigate to the project directory:
   ```bash
   cd knn-comparison
   ```

3. Run the scratch implementation:
   ```bash
   python knn_scratch.py
   ```

4. Run the library-based implementation:
   ```bash
   python knn_libraries.py
   ```

---

## Code Explanation

### KNN from Scratch
The `knn_scratch.py` file demonstrates:
- **Euclidean Distance Calculation**: Measures the distance between data points in a multi-dimensional space.
- **Prediction**: Finds the `k` nearest neighbors and assigns the most common class label among those neighbors.
- **Accuracy Calculation**: Compares predicted vs actual values and computes the accuracy of the model.

### KNN Using `scikit-learn`
The `knn_libraries.py` file shows:
- **Model Training**: Using `KNeighborsClassifier` from `scikit-learn`.
- **Model Evaluation**: The `accuracy_score` function is used to evaluate the accuracy.
- **Hyperparameter Tuning**: A loop is used to test various values of `k` (the number of neighbors) and plot the resulting accuracy scores.

---

## Results

- **Accuracy**: 
  - Scratch Implementation: 91.00%
  - Library Implementation: 91.50%

The accuracy values can differ based on the dataset and the number of neighbors (`k`) chosen for the model.

---

## Conclusion

This project highlights the differences between implementing KNN manually and using the `scikit-learn` library. While implementing the algorithm from scratch gives a solid understanding of how it works, using libraries saves time and computational resources, making it a more practical choice for real-world applications.

---

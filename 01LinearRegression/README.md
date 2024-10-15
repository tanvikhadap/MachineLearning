# Linear Regression Demonstration: From Scratch vs. Using Libraries

## What is it?

This project demonstrates how to implement **Linear Regression** in Python using two different approaches:
1. **Using Libraries:** like scikit-learn to handle the regression modeling process.
2. **From Scratch:** manually implementing the mathematical principles of linear regression without using any external libraries.

The dataset used in this project contains temperature readings recorded at different times and locations, with features such as date, time, and room information.

## Why should I make it?

This README file explains how to run the project and gives insights into the code. By making a README, users and collaborators can easily understand:
- The purpose of the project.
- How to set up and run the code.
- What results to expect and how to interpret them.

A README helps in maintaining clarity and makes it easy for others to contribute to or use the project.

## Who should make it?

Anyone working on a programming project, especially if it's meant to be shared publicly or with other team members. A README is useful whether you're the sole developer or working in a collaborative environment, as it ensures anyone using or modifying the code understands the project.

## When should I make it?

A README should be created at the beginning of the project, ideally as the first file in the repository. It should definitely be ready before sharing the project with others or making it public. You can continue updating it as the project evolves.

## Where should I put it?

The README file should be placed in the **top-level directory** of the project. This is where most users will look for instructions. Code hosting platforms like **GitHub** and **GitLab** will automatically display the README when viewing the project directory.

## How should I make it?

READMEs are typically written in **Markdown** due to its lightweight formatting capabilities and widespread use in project documentation. Markdown allows easy formatting of headings, lists, and code snippets, while remaining simple to write and read.

You can write your README in any text editor or a dedicated Markdown editor such as **Typora**, **Visual Studio Code**, or use online Markdown editors like **Dillinger** or **StackEdit**. 

---

## Project Overview

This project includes two implementations of **Linear Regression** on a dataset of IoT-based temperature readings:
1. **Using Libraries**: The scikit-learn library is used to create and evaluate a linear regression model.
2. **From Scratch**: The linear regression model is implemented using NumPy to perform matrix operations based on the normal equation.

The dataset includes time-series data with features extracted from the `noted_date` field, such as year, month, day of the week, hour, and minute. Temperature readings are filtered to include only "In" room temperatures.

---

## Setup Instructions

### Requirements

- **Python 3.x**
- Required libraries:
    - numpy
    - pandas
    - seaborn
    - matplotlib
    - scikit-learn

You can install the required libraries using the following command:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Running the Code

1. Clone or download the repository and navigate to the project directory:
    ```bash
    git clone <repository-url>
    cd linear-regression-demo
    ```

2. Ensure that the dataset (`IOT-temp.csv`) is present in the project folder.

3. Run the script:
    ```bash
    python linear_regression.py
    ```

### Outputs

- **Visualizations**: Scatter plots and histograms to explore relationships between temperature and various features (e.g., year, month, day, hour).
- **Model Performance**: The code outputs **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** for both the library-based and from-scratch implementations.
  
---

## Features

1. **Data Preprocessing**:
   - Parsing `noted_date` into relevant components such as year, month, day of the week, hour, and minute.
   - Filtering the data to include only "In" room temperature readings.
   
2. **Data Visualization**:
   - Scatter plots of temperature vs. features like year, month, and day.
   - Histogram to display temperature distribution across months.

3. **Linear Regression Using Libraries**:
   - Uses **scikit-learn** to split the data into training and testing sets.
   - Builds a linear regression model and evaluates it on the test set using MSE and RMSE.

4. **Linear Regression From Scratch**:
   - Implements the normal equation to compute linear regression coefficients manually.
   - Predicts the target variable and calculates the error (MSE, RMSE) without using libraries.

---

## Conclusion

This project provides a hands-on demonstration of linear regression, showing both how it's done using established machine learning libraries, and how you can implement the model manually using basic Python and NumPy. The goal is to deepen understanding of the mathematical concepts behind linear regression while appreciating the convenience of using libraries like scikit-learn.

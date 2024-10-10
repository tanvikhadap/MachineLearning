import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\IOT-temp.csv")
df.head()

df.shape
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df["noted_date"] = pd.to_datetime(df["noted_date"], format="%d-%m-%Y %H:%M")
df["year"] = df["noted_date"].apply(lambda date: date.year)
df["month"] = df["noted_date"].apply(lambda date: date.month)
df["day"] = df["noted_date"].apply(lambda date: date.day_name())
df["week_of_year"] = df["noted_date"].apply(lambda date: date.weekofyear)
df["hour"] = df["noted_date"].apply(lambda date: date.hour)
df["minute"] = df["noted_date"].apply(lambda date: date.minute)

df["room_id/id"].value_counts()
df.drop(["id", "room_id/id"], axis=1)

df['temp'] = df['temp'].astype(int)
df['week_of_year'] = df['week_of_year'].astype(int)
df['hour'] = df['hour'].astype(int)
df['minute'] = df['minute'].astype(int)
print(df.dtypes)

day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

df['day'] = df['day'].map(day_mapping)

df.drop(["noted_date"], axis=1, inplace=True)

df.head()

df = df[df['out/in'] == 'In']
print(df.head())

df.value_counts()

df.shape

# Data Visualization
fig, axes = plt.subplots(nrows=2,ncols=4,figsize=(16,6))
axes = axes.flatten()
axes[0].plot(df['out/in'],df['temp'],'o')
axes[0].set_ylabel('temp')
axes[0].set_title('out/in')

axes[1].plot(df['year'],df['temp'],'o')
axes[1].set_ylabel('temp')
axes[1].set_title('year')

axes[2].plot(df['month'],df['temp'],'o')
axes[2].set_ylabel('temp')
axes[2].set_title('month')

axes[3].plot(df['day'],df['temp'],'o')
axes[3].set_ylabel('temp')
axes[3].set_title('day')

axes[4].plot(df['week_of_year'],df['temp'],'o')
axes[4].set_ylabel('temp')
axes[4].set_title('week_of_year')

axes[5].plot(df['hour'],df['temp'],'o')
axes[5].set_ylabel('temp')
axes[5].set_title('hour')

axes[6].plot(df['minute'],df['temp'],'o')
axes[6].set_ylabel('temp')
axes[6].set_title('minute')

fig.delaxes(axes[7])

plt.tight_layout()
plt.show()

# Histogram Plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='temp', hue='month', multiple='stack', palette='viridis')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Average Temperature by Month')
plt.show()


# # Using Libraries 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df['out/in'] = df['out/in'].map({'In': 0, 'Out': 1})

X = df.drop(columns=['temp'])
y = df['temp']

X_matrix = X.values
y_vector = y.values

print("Features matrix (X):")
print(X_matrix)

print("\nTarget vector (y):")
print(y_vector)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Square Error (RMSE):", rmse)


# # From scratch 

# Number of training examples
m = len(y_vector)

theta = (X^T * X)^(-1) * X^T * y
theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

print("Coefficients:", theta)

# Predict values using the calculated coefficients
def predict(X, theta):
    return np.dot(X, theta)

# Predict on the same dataset (for demonstration purposes)
predictions = predict(X, theta)
print("Predictions:", predictions)

# To calculate the error (Mean Squared Error)
m = len(y)
error = (1/(2*m)) * np.sum((predictions - y) ** 2)
print("Mean Squared Error:", error)

rmse = np.sqrt(error)

print("Root Mean Square Error (RMSE):", rmse)



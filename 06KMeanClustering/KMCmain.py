import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\car_data.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df

correlation_matrix = df[['Gender','Age', 'AnnualSalary', 'Purchased']].corr()
print(correlation_matrix)

df = df.drop(columns=['User ID','Gender'])

df1 = df.sample(n=100, random_state=42)
df.head()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df1, x='Age', y='AnnualSalary')
plt.title('Scatter Plot of Age vs AnnualSalary')
plt.xlabel('Age')
plt.ylabel('Annual Salary')
plt.show()


# # K-mean Code from scratch 

X = df1[['Age', 'AnnualSalary']]

from sklearn.preprocessing import StandardScaler
# Scale the data (K-Means is sensitive to scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


import random

#decide clusters
#select random centriods
#assign clusters
#move centriods
#check finish

class Kmeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def fit_predict(self,X):
        X = np.array(X)
        random_index = random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids = X[random_index]
        print(self.centroids)
        
        for i in range(self.max_iter):
            #assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X,cluster_group)
            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_group
    def assign_clusters(self,X):
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self,X,cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)
km = Kmeans(n_clusters=4)
y_means = km.fit_predict(X)

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, y_means)

print(f"Silhouette Score: {silhouette_avg}")


# # KMean code using libraries

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df1)
    wcss.append(km.inertia_)

wcss

plt.plot(range(1,11),wcss)

X = df1.iloc[:,:].values
km = KMeans(n_clusters=4)
y_means = km.fit_predict(X)

y_means

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, y_means)

print(f"Silhouette Score: {silhouette_avg}")

X[y_means == 3,1]

plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='blue')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='red')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')
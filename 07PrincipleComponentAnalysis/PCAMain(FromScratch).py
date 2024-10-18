import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
sImageFilePath=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'DataSets_Required','Images','IMAGE2' + '.' + 'jpeg'))

#
image_raw = imread(sImageFilePath)
#
#Code to get folder path
# imageFileName = 


#
print(image_raw.shape)

# Displaying the image
plt.figure(figsize=[12,8])
plt.imshow(image_raw)

# Calculate the storage size of the image_bw array
image_size_bytes = image_raw.nbytes
print(f"Storage size of imge_raw: {image_size_bytes} bytes")

image_sum = image_raw.sum(axis=2)
print(image_sum.shape)

image_bw = image_sum/image_sum.max()
print(image_bw.max())

plt.figure(figsize=[12,8])
plt.imshow(image_bw, cmap=plt.cm.gray)

height, width = image_raw.shape[:2]
print(f"Image Dimensions: {height}x{width} pixels")

# Calculate the storage size of the image_bw array
image_size_bytes = image_bw.nbytes
print(f"Storage size of image_bw: {image_size_bytes} bytes")

# PCA from scratch function
def pca_scratch(X, n_components=None):
    # Mean center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select a subset of the eigenvectors (top n_components)
    if n_components is not None:
        components = sorted_eigenvectors[:, :n_components]
    else:
        components = sorted_eigenvectors
    
    return components, mean, sorted_eigenvalues


# Flatten the grayscale image for PCA
X = image_bw.reshape(-1, image_bw.shape[1])

# Perform PCA without limiting components to get all eigenvalues
components, mean, eigenvalues = pca_scratch(X)

# Print the eigenvalues
print("Eigenvalues:\n", eigenvalues)

# Calculate the explained variance ratio (normalized eigenvalues)
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Variance spread",explained_variance_ratio)

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio) * 100

# Plot cumulative explained variance
plt.figure(figsize=[10, 5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance (%)')
plt.xlabel('Number of principal components')
plt.plot(cumulative_explained_variance, lw=2)
plt.axhline(y=95, color='r', linestyle='--')  # 95% variance line
plt.axvline(x=np.argmax(cumulative_explained_variance > 95), color='k', linestyle='--')  # Point where 95% is reached
plt.grid(True)
plt.show()

# Find the number of components that explain 95% of the variance
k_95 = np.argmax(cumulative_explained_variance > 95) + 1  # +1 because indices start at 0
print(f"Number of components explaining 95% variance: {k_95}")

# Function to plot PCA-reduced images with size in KB
def plot_at_k(k):
    components, mean, _ = pca_scratch(X, k)
    
    # Project data onto the selected components
    X_centered = X - mean
    X_reduced = X_centered @ components
    
    # Reconstruct the image
    X_reconstructed = X_reduced @ components.T + mean
    X_reconstructed = X_reconstructed.reshape(image_bw.shape)
    
    # Calculate size in KB
    size_in_bytes = X_reconstructed.nbytes
    size_in_kb = size_in_bytes / 1024
    
    # Plot the reconstructed image
    plt.imshow(X_reconstructed, cmap=plt.cm.gray)
    plt.axis('off')
    
    return size_in_kb

# Values of k to visualize
ks = [10, 25, 50, 100, 150, 180]

plt.figure(figsize=[15, 9])

# Plot the PCA-reduced images for each value of k with sizes
for i in range(len(ks)):
    plt.subplot(2, 3, i + 1)
    size_kb = plot_at_k(ks[i])
    plt.title(f"Components: {ks[i]}, Size: {size_kb:.2f} KB")

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

import sys

# Values of k to visualize
ks = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 50, 70, 90, 110, 130, 150, 170, 175, 180]

plt.figure(figsize=[20, 16])

# Display the original image first
plt.subplot(5, 5, 1)
plt.imshow(image_bw, cmap=plt.cm.gray)
plt.title(f"Original Image, Size: {sys.getsizeof(image_bw) / 1024:.2f} KB")
plt.axis('off')

# Plot the PCA-reduced images for each value of k with sizes
for i in range(len(ks)):
    plt.subplot(5, 5, i + 2)
    size_kb = plot_at_k(ks[i])
    plt.title(f"Components: {ks[i]}, Size: {size_kb:.2f} KB")

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
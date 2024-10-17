---

# PCA Image Compression

## Introduction
This project demonstrates the use of Principal Component Analysis (PCA) and Incremental PCA (IPCA) for image compression. The image is first converted to grayscale, and then PCA is applied to reduce the dimensionality while retaining a significant portion of the original variance. The reduced image is reconstructed using the principal components.

## Requirements
To run this code, the following libraries are required:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the necessary packages using the following commands:
```bash
pip install numpy matplotlib scikit-learn
```

## Dataset
In this project, the image used for compression is loaded from a JPEG file. The image is read using `matplotlib.image.imread()` and processed in grayscale.

## Code Overview
### 1. Importing Libraries and Loading Image
The image is loaded using the `imread()` function from the `matplotlib` library and then displayed. The color image is converted to grayscale by summing across the color channels and normalizing the values.
```python
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Load the image
image_raw = imread("path_to_your_image.jpeg")

# Convert image to grayscale
image_sum = image_raw.sum(axis=2)
image_bw = image_sum/image_sum.max()

# Display the grayscale image
plt.imshow(image_bw, cmap=plt.cm.gray)
```

### 2. Applying PCA
PCA is applied to the grayscale image, and we compute how much variance is explained by each principal component. The cumulative variance is then plotted to understand how many components are necessary to retain 95% of the variance.

```python
from sklearn.decomposition import PCA, IncrementalPCA

# Fit PCA to the grayscale image
pca = PCA()
pca.fit(image_bw)

# Cumulative variance explained by components
var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
k = np.argmax(var_cumu>95)

# Plot cumulative variance
plt.plot(var_cumu)
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
plt.show()
```

### 3. Reconstructing the Image
We use Incremental PCA to transform and reconstruct the image using a selected number of principal components (in this case, the number of components that explains 95% of the variance). The reconstructed image is then displayed alongside images compressed at various component levels.

```python
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

# Display the reconstructed image
plt.imshow(image_recon, cmap=plt.cm.gray)
```

### 4. Visualizing Compression with Different Component Counts
The code includes a function to plot the reconstructed image with varying numbers of principal components to visualize the effect of compression.

```python
def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
    plt.imshow(image_recon, cmap=plt.cm.gray)

# Display images for different component counts
ks = [10, 25, 50, 100, 150, 180]
for i in range(6):
    plt.subplot(2,3,i+1)
    plot_at_k(ks[i])
    plt.title(f"Components: {ks[i]}")
plt.show()
```

### 5. Results
The reconstructed images demonstrate the balance between the number of components used in PCA and the quality of the reconstructed image. With fewer components, the image quality decreases, but the amount of data needed for storage is significantly reduced.

## Conclusion
This project highlights the effectiveness of PCA for image compression, showing how it can retain a large portion of the image’s information with fewer components, reducing the storage size while maintaining quality.

---

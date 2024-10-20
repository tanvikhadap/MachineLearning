import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import manifold

get_ipython().run_line_magic('matplotlib', 'inline')

data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True  # This must be inside the function call
)
pixel_values, targets = data
targets = targets.astype(int)

pixel_values_array = pixel_values.to_numpy()

pixel_values_array.shape

# Use .iloc to access the second image in the DataFrame
single_image = pixel_values.iloc[1, :].values.reshape(28, 28)

# Display the image
plt.imshow(single_image, cmap='gray')
plt.show()


# # t-SNE code using libraries

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values_array[:3000, :])

tsne_df = pd.DataFrame(
np.column_stack((transformed_data,targets[:3000])),
columns=["x","y","targets"]
)
tsne_df.loc[:,"targets"] = tsne_df.targets.astype(int)

print(tsne_df.head(11))

grid = sns.FacetGrid(tsne_df, hue="targets", height=8, aspect=1.5)
grid.map(plt.scatter, "x","y").add_legend()
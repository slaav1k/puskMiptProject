import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


np.random.seed(42)
X = np.random.rand(100, 2)

k_means = KMeans(n_clusters=3)
k_means.fit(X)

labels = k_means.labels_
center_coords = k_means.cluster_centers_


print(center_coords)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(center_coords[:, 0], center_coords[:, 1], marker='*', s=300, c='r')
plt.show()

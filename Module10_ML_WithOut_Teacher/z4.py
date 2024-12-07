import numpy as np
from sklearn.decomposition import PCA
np.random.seed(42)
X = np.random.rand(100, 5)

# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)

# explained_variance_ratio = pca.explained_variance_ratio_
# print(explained_variance_ratio)
# loadings = pca.components_
loadings = pca.explained_variance_ratio_
print(loadings)
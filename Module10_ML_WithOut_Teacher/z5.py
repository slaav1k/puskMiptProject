import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds

np.random.seed(42)
X = np.random.rand(100, 5)

U, S, Vt = svds(X, k=2)
S_diag = np.diag(S)
predicted_ratings = np.dot(np.dot(U, S_diag), Vt)

user_id = 2
item_id = 4
# print(predicted_ratings)
result = predicted_ratings[2, 4]
result = round(result, 1)

print(result)
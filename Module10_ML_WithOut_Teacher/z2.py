from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

X, y = make_blobs(n_samples=100, n_features=4, centers=4, random_state=42)

res = []

for i in range(10):
    X, y = make_blobs(n_samples=100, n_features=4, centers=4, random_state=i)

    Z = linkage(X, 'ward')
    labels = fcluster(Z, t=4, criterion='maxclust')
    silhouette = silhouette_score(X, labels)
    print(silhouette)
    res.append(silhouette)
    fig, ax = plt.subplots(figsize=(8, 6))
    dendrogram(Z)
    plt.title("Dendrogram")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.show()
silhouette_avg = sum(res) / len(res)
print("avg =", silhouette_avg)


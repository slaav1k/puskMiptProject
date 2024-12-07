from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

X, y = make_blobs(n_samples=1000, centers=3, random_state=42)


def dbscan_silhouette(eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_
    # labels = dbscan.fit_predict(X)
    # silhouette = silhouette_score(X, labels)
    # print(silhouette)
    if len(set(labels)) > 1:
        # Расчет силуэтных коэффициентов для каждой точки
        silhouette_values = silhouette_samples(X, labels)
        # print(silhouette_values)
        # Возврат среднего силуэтного коэффициента
        return np.mean(silhouette_values)
    else:
        # Если кластеров меньше 2 (например, все точки в одном кластере или выбросы), вернуть -1 или другое значение
        return -1


best_eps, best_min_samples, best_silhouette = None, None, -1
for eps in [0.1, 0.5, 1]:
    for min_samples in [5, 10, 20]:
        silhouette = dbscan_silhouette(eps, min_samples)
        if silhouette > best_silhouette:
            best_eps, best_min_samples, best_silhouette = eps, min_samples, silhouette

print([best_eps, best_min_samples, best_silhouette])

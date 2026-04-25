import numpy as np

def kmeans(X, k=2, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# TEST
X = np.array([[1,2],[1,4],[5,6],[6,7]])
labels, centroids = kmeans(X, k=2)

print("Labels:", labels)
print("Centroids:", centroids)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
plt.title("KMeans Clustering")
plt.show()

# Hierarchical
hc = AgglomerativeClustering(n_clusters=3)
labels_h = hc.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_h)
plt.title("Hierarchical Clustering")
plt.show()
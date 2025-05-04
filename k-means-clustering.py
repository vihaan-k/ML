import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

'''K-means method:
1. Take k random points and call them the centroids of the clusters. 
Each point that's closest in euclidean distance will belong to that clusters
2. Calculate the new centroid for each cluster.
Repeat until the centroids converge or until max iterations is reached.'''

plt.figure(1)
plt.plot(range(1,11), inertias, marker="o")
# We can see that in figure 1 the graph changes suddenly at K = 2 (called the elbow).

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.figure(2)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()
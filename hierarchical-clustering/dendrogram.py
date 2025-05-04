import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

'''Euclidean distance: Distance between 2 points on a plane (using pythagorean theorem).

Ward\'s linkage method: Each points starts off as its own cluster. Then, the 2 clusters with
the lowest increase in variance are merged.

Increase in variance:
1. Find the centroids of the 2 clusters (mean of x, mean of y)
2. Calculate the square of the euclidean distance between the centroids
3. Multiply the result by ab/(a+b) where a and b are the number of points in the clusters.
'''

plt.show()
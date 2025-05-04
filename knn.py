import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

data = list(zip(x, y))
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(data, classes)

new_point = (8, 21)
prediction1 = knn1.predict([new_point])
print(prediction1)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(data, classes)
prediction5 = knn5.predict([new_point])
print(prediction5)

'''KNN is a method for classification and regression. It works by finding the k nearest neighbors by euclidean 
distance of a point and predicting the class or value based on those neighbors. For classification, 
the class with the most votes is chosen and for regression, the average of the values is chosen.
See KNeighborsRegressor for regression.'''

plt.figure(1)
plt.title("KNN with K = 1")
plt.scatter(x+[8], y+[21], c=classes+[prediction1[0]])
plt.text(7, 20, "New point")

plt.figure(2)
plt.title("KNN with K = 5")
plt.scatter(x+[8], y+[21], c=classes+[prediction5[0]])
plt.text(7, 20, "New point")

plt.show()
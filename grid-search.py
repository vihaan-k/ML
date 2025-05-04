from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]

logr = LogisticRegression(max_iter = 10000)

'''C-value = How well the model tries to fit the data. 
High C-value: Model tries to fit the data as best as possible (could lead to overfitting)
Low C-value: Model tries to find a simple solution (could lead to underfitting)'''

C = [i/4 for i in range(1, 13)]
scores = {}

for choice in C:
    logr.set_params(C=choice)
    logr.fit(X, y)
    scores[choice] = (logr.score(X, y))

print(scores) # 1.75 is the best C-value because we're trying to prevent overfitting.
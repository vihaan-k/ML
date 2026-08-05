from csv import DictReader
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

filepath = "decision-tree\comedian-data.csv"

'''with open(filepath, "r", encoding="utf-8") as csv_file:
    data = [_ for _ in DictReader(csv_file)]

for show in data:
    if show["Go"] == "YES": show["Go"] = 1
    if show["Go"] == "NO": show["Go"] = 0
    if show["Nationality"] == "UK": show["Nationality"] = 0
    if show["Nationality"] == "USA": show["Nationality"] = 1
    if show["Nationality"] == "N": show["Nationality"] = 2

age = [person["Age"] for person in data]
experience = [person["Experience"] for person in data]
rank = [person["Rank"] for person in data]
nationality = [person["Nationality"] for person in data]
go = [person["Go"] for person in data]

X = np.array([age, experience, rank, nationality]).T
y = np.array(go)'''

data = pd.read_csv(filepath)

nat = {"UK": 0, "USA": 1, "N": 2}
data["Nationality"] = data["Nationality"].map(nat)

go = {"YES": 1, "NO": 0}
data["Go"] = data["Go"].map(go)

X = data[["Age", "Experience", "Rank", "Nationality"]]
y = data["Go"]

dtree = tree.DecisionTreeClassifier().fit(X, y)

tree.plot_tree(dtree, feature_names=["Age", "Experience", "Rank", "Nationality"])

'''Gini score is between 0 and 0.5 and it tells us how the y is split. 0 means y only has one value
and 0.5 means both values of y are evenly split
Formula: gini = 1 - (x/n)^2 - (y/n)^2. x = negative y, y = positive y and n = total samples.'''

print("Prediction for [40, 10, 7, 1]:", dtree.predict([[40, 10, 7, 1]]))
plt.show()
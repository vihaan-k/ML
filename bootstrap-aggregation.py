from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

# Bagging = Bootstrap aggregating

data = datasets.load_wine(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print("Train data accuracy of base classifier:", accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)))
print("Test data accuracy of base classifier:", accuracy_score(y_true=y_test, y_pred=y_pred))

'''A bagging classifier trains multiple trees on a bootstrap sample and predicts using a vote.
A random 1/e of the total samples are used on each tree.'''
scores = []
estimator_range = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30] # Grid search
for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

print("\nAccuracy score for every n estimators:\n", list(zip(estimator_range, scores))) 
# >=16 estimators has the highest accuracy


final_clf = BaggingClassifier(n_estimators=16, oob_score=True, random_state=0)
final_clf.fit(X_train, y_train)
print("\nOOB-score of the bagging model:", final_clf.oob_score_)
print("Test score of the bagging model:", accuracy_score(y_test, final_clf.predict(X_test)))
# OOB-score: How well the model performs on out of bag (unseen) training data (the remaining 1-1/e samples)

# All the trees trained
for i in range(16):
    plt.figure(i+1, (30, 20))
    plot_tree(final_clf.estimators_[i], feature_names=X.columns)
plt.show()
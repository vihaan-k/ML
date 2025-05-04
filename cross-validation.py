from sklearn import datasets
from sklearn.model_selection import (
    KFold, cross_val_score, StratifiedKFold, 
    LeaveOneOut, LeavePOut, ShuffleSplit
)
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=0)

k_folds = KFold(n_splits=5)
scores_kf = cross_val_score(clf, X, y, cv=k_folds)
print("K-folds cross validation scores:", scores_kf)
print("Mean k-folds cross validation score:", scores_kf.mean())
'''Data is split into 5 folds, 4 for training and 1 for testing
The model is trained on 4 folds and tested on the 5th fold, and the testing fold is rotated
The process is repeated 5 times, each time with a different fold as the test set'''

sk_folds = StratifiedKFold(n_splits=5)
scores_skf = cross_val_score(clf, X, y, cv=sk_folds)
print("\nStratified cross validation scores:", scores_skf)
print("Mean stratified cross validation score:", scores_skf.mean())
'''Stratified K-fold CV splits the classes into each fold evenly so that 
the class distribution is preserved in each fold e.g. 90% of class 0 and 10% of class 1 in each fold.
Straified K-fold is better for imbalanced datasets where the class distribution is not equal.'''

loo = LeaveOneOut()
scores_loo = cross_val_score(clf, X, y, cv = loo)
print("\nLeave-one-out CV scores: ", scores_loo)
print("Mean CV Score: ", scores_loo.mean())
'''Leave-one-out CV trains the model on all but one sample and tests on the left out sample.
The process is repeated for each sample. It's best for small datasets but can be slow for large datasets.'''

lpo = LeavePOut(p=2)
scores_lpo = cross_val_score(clf, X, y, cv = lpo)
print("\nLeave-P-Out CV scores: ", scores_lpo)
print("Mean CV Score: ", scores_lpo.mean())
'''Just like leave-one-out CV, but leaves out P samples instead of 1.
This works for larger datasets because it's much faster than LOO.
A higher P is less accurate but faster. A lower P is more accurate but slower.'''

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)
scores_ss = cross_val_score(clf, X, y, cv = ss)
print("\nShuffle split CV Scores: ", scores_ss)
print("Mean CV Score: ", scores_ss.mean())
'''Shuffle split CV randomly splits the data into the given train and test sizes.
Instead of rotating the folds, it randomly samples the data for each split.
This works well for large datasets and is very flexible'''
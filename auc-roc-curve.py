import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y = np.array([0]*500 + [1]*9500)

y_prob = np.array([1]*10000)
y_pred = y_prob > 0.5

print(f'Accuracy score 1.1: {accuracy_score(y, y_pred)}')
cf_mat1 = confusion_matrix(y, y_pred)
print('Confusion matrix 1.1')
print(cf_mat1)
print(f'class 0 accuracy: {cf_mat1[0][0]/500}')
print(f'class 1 accuracy: {cf_mat1[1][1]/9500}')

y_prob2 = np.array(
    np.random.uniform(0, 0.7, 500).tolist() + 
    np.random.uniform(0.3, 1, 9500).tolist()
)
y_pred2 = y_prob2 > 0.5

print(f'\n  Accuracy score 1.2: {accuracy_score(y, y_pred2)}')
cf_mat2 = confusion_matrix(y, y_pred2)
print('Confusion matrix 1.2')
print(cf_mat2)
print(f'class 0 accuracy: {cf_mat2[0][0]/500}')
print(f'class 1 accuracy: {cf_mat2[1][1]/9500}')

def plot_roc_curve(y_true, y_prob):
    # FPR = False Positive Rate: FP / (FP + TN), TPR = True Positive Rate (Recall)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr) # ROC curve is a plot of FPR (x) and TPR (y)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

'''
1. Each predicted value by the model is a threshold.
2. A predicted array is created at each threshold. 1 is >= the threshold, 0 is < the threshold.
3. The TPR and FPR are calculated for each threshold.
4. Theses FPR and TPR values are plotted to create the ROC curve.
'''

print("AUC score 1.1:", roc_auc_score(y, y_prob))
print("AUC score 1.2:", roc_auc_score(y, y_prob2))
# AUC score = Area under ROC curve. < 0.5 = bad model, 0.5 = random model, 1 = perfect model.

y2 = np.array([0]*10000 + [1]*10000)
y2_prob1 = np.array(
    np.random.uniform(.25, .5, 5000).tolist() +
    np.random.uniform(.3, .7, 10000).tolist() +
    np.random.uniform(.5, .75, 5000).tolist()
)
y2_prob2 = np.array(
    np.random.uniform(0, .4, 5000).tolist() +
    np.random.uniform(.3, .7, 10000).tolist() +
    np.random.uniform(.6, 1, 5000).tolist()
)

print("\nModel 2.1 accuracy score:", accuracy_score(y2, y2_prob1>.5))
print("Model 2.2 accuracy score:", accuracy_score(y2, y2_prob2>.5))
print("Model 2.1 AUC score:", roc_auc_score(y2, y2_prob1))
print("Model 2.2 AUC score:", roc_auc_score(y2, y2_prob2))

'''Models 2.1 and 2.2 have similar accuracy scores, but different AUC scores.
The model with the higher AUC score is better at distinguishing between the two classes.'''

plt.figure(1)
plt.title('ROC curve 1.1')
plot_roc_curve(y, y_prob)

plt.figure(2)
plt.title('ROC curve 1.2')
plot_roc_curve(y, y_prob2)


plt.figure(3)
plt.title('ROC curve 2.1')
plot_roc_curve(y2, y2_prob1)

plt.figure(4)
plt.title('ROC curve 2.2')
plot_roc_curve(y2, y2_prob2)

plt.show()
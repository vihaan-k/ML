import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

actual = np.random.binomial(1, 0.9, 10)
predicted = np.random.binomial(1, 0.9, 10)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

'''
Top-Left: True Negative (TN)
Top-Right: False Positive (FP)
Bottom-Left: False Negative (FN)
Bottom-Right: True Positive (TP)
'''

accuracy = metrics.accuracy_score(actual, predicted)
print("Accuracy:", accuracy) # Percentage that was correctly predicted
# (TN + TP) / total

precision = metrics.precision_score(actual, predicted)
print("Precision:", precision) # Percentage of predicted positives that were predicted correctly
# TP / (TP + FP)

recall = metrics.recall_score(actual, predicted)
print("Recall:", recall) 
# Recall / Sensitivity = Percentage of actual positives that were correctly predicted
# TP / (TP + FN)

specificity = metrics.recall_score(actual, predicted, pos_label = 0)
print("Specificity:", specificity) # Percentage of actual negatives that were correctly predicted
# TN / (TN + FP)

f_score = metrics.f1_score(actual, predicted)
print("F-score:", f_score) 
# Score of how well the predictions were made. Used for imbalanced data where precision and recall are important
# Harmonic mean of precision and sensitivity
# (2 * Precision * Sensitivity) / (Precision + Sensitivity)

print(actual)
print(predicted)

cm_display.plot()
plt.show()
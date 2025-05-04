from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
# Size of tumor in mm

y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # Tumor cancerous?

logr = LogisticRegression()
logr.fit(X, y)

print("Prediction with tumor = 3.46mm:", logr.predict(np.array([3.46]).reshape(-1, 1))[0])

print("Log-odds:", np.exp(logr.coef_)[0][0])
# If the tumor increases by 1mm, the odds of it being cancerous are 4.04x

def prob(logr: LogisticRegression, X: np.ndarray[tuple[int, int]]):
    odds = np.exp(logr.coef_ * X + logr.intercept_)
    return odds / (1 + odds)

# logr.coef_ = a, logr.intercept_ = b. The logistic function has the form e^(ax+b)/[1+e^(ax+b)]

print(prob(logr, X)) # Probability of each tumor being cancerous

y = [i[0] for i in prob(logr, X)]

plt.scatter(X, y)
line_x = np.linspace(0, 7, 500)
line_y = prob(logr, line_x)
plt.plot(line_x, line_y[0])

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

x = np.random.normal(3, 1, 1000)

for i, num in enumerate(x):
    if num < 0.1: x[i] = 0.1

y = np.random.normal(150, 40, 1000) / x

train_x = x[:800]
train_y = y[:800]

test_x = x[800:]
test_y = y[800:]

model = np.poly1d(np.polyfit(train_x, train_y, 5))
line = np.linspace(0, 7, 1000)

plt.scatter(train_x, train_y)
plt.plot(line, model(line))

print("Train R^2:", pearsonr(train_y, model(train_x)).statistic**2)
print("Test R^2:", pearsonr(test_y, model(test_x)).statistic**2)
print("Prediction of input 5:" , model(5))

plt.show()
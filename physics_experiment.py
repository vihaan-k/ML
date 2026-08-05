import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Temperature of air in °C
y = [1013, 1050, 1087, 1115, 1152, 1190, 1227, 1265, 1300, 1341, 1380] # Air pressure in hPa

slope, y_intercept, r, p, std_err = linregress(x, y)

def func(x):
    return slope * x + y_intercept

line_x = np.linspace(0, 100, 1000)

plt.scatter(x, y)
plt.plot(line_x, func(line_x))
plt.xlabel("θ (°C)")
plt.ylabel("p (hPa)")



print("R:", r)
print("R^2:", r*r)

plt.show()
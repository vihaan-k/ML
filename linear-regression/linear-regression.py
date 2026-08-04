import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from numpy import linspace

# Data x-values and y-values
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

'''f(x) = ax + b, a = slope, b = y-intercept, 
r = Pearson R-value, p = P-value, std_err = Standard error'''
slope, y_intercept, r, p, std_err = linregress(x, y)

# Function of best possible linear regression
def line_func(x):
    return slope * x + y_intercept

# 500 evenly spaced numbers between and including 2 and 17
line_x = linspace(2, 17, 500)

# Scatter plot of data and line of linear regression
plt.scatter(x, y)
plt.plot(line_x, line_func(line_x)) # plt.plot(x, y) plots a line graph

# Pearson-R value and R-squared value
print("R:", r)
print("R^2:", r*r)

plt.show()

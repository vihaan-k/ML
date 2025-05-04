import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

'''Best function of degree 3 for the values x and y. Polyfit finds the best coefficients 
and Poly1d converts the coefficients into a polynomial function'''
func = np.poly1d(np.polyfit(x, y, 3)) 

# Generates an array of 500 evenly spaced floats between 1 and 22. These are the x values of the trendline
x_graph = np.linspace(1, 22, 500) 


plt.scatter(x, y) # Scatter plot of dataset
plt.plot(x_graph, func(x_graph)) # Trendline of function

print("R:", pearsonr(y, func(x)).statistic) # R-value of the y points and the y values of the function
print("R^2:", pearsonr(y, func(x)).statistic**2) # R^2 value

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Array of 500 random floats between 0 and 500
a = np.random.uniform(0, 5, 500) 

'''Array of 100000 random floats with an approximate normal distribution (bell curve) 
with a mean of 5 and a standard deviation of 1'''
b = np.random.normal(5, 1, 100000)

plt.figure(1)
plt.hist(a, 5) # Plot a histogram of dataset a with 5 bars
plt.figure(2)
plt.hist(b, 100) # Plot a histogram of dataset b with 100 bars
plt.show()
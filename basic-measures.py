import numpy as np
from scipy.stats import mode

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
print("Mean:", np.mean(speed))
print("Median:", np.median(speed))
print("Mode:", mode(speed).mode)
print("Frequency of mode:", mode(speed).count)
print("Standard deviation:", np.std(speed))
print("Variance:", np.std(speed)**2)
print("75th percentile:", np.percentile(speed, 75))
print("90th percentile:", np.percentile(speed, 90))
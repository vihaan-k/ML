# Basic Measures

## Overview
Basic statistical measures summarize key characteristics of a dataset, such as its central tendency, spread, and relative position of values. The primary measures include the mean, median, mode, frequency of the mode, standard deviation, variance, range, and percentiles.

---

## Mathematical Formulation

### 1. Mean ($\mu$)
The arithmetic average of a dataset $X = \{ x_1, x_2, \ldots, x_n \}$:

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$

### 2. Median ($\tilde{x}$)
The middle value when the dataset is ordered $x_{1} \le x_{2} \le \ldots \le x_{n}$:

- $\tilde{x} = x_{\frac{n+1}{2}}$ if $n$ is odd
- $\tilde{x} = \frac{1}{2}(x_{\frac{n}{2}} + x_{\frac{n}{2}+1})$ if $n$ is even

### 3. Mode ($M_o$) and Frequency
- **Mode ($M_o$):** The value that appears most frequently in the dataset:
  $$M_o = \arg\max_{x} \text{Frequency}(x)$$
- **Frequency of Mode ($f_{M_o}$):** The count of occurrences of the modal value:
  $$f_{M_o} = \max_x \text{Frequency}(x)$$

### 4. Standard Deviation ($\sigma$)
A measure of the dispersion or spread of the data relative to the mean:

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2}$$

### 5. Variance ($\sigma^2$)
The average of the squared differences from the mean:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2$$

### 6. Range ($R$)
The difference between the maximum and minimum values in the dataset:

$$R = \max(X) - \min(X)$$

### 7. Percentile ($P_p$)
The value below which a given percentage ($p\%$) of observations fall. For a given percentile $p \in [0, 100]$:

$$P_p = \text{value at rank } k = 1 + \frac{p}{100}(n - 1)$$

*(Interpolation is used if $k$ is not an integer).*

---

## Code Example

```python
import numpy as np
from scipy import stats

# Dataset
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Calculate statistics
mean_val = np.mean(speed)
median_val = np.median(speed)
mode_res = stats.mode(speed)
std_val = np.std(speed)
var_val = np.var(speed)  # or np.std(speed)**2
range_val = np.ptp(speed)  # Peak-to-peak (max - min)
p75 = np.percentile(speed, 75)
p90 = np.percentile(speed, 90)

# Display results
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val}")
print(f"Mode: {mode_res.mode}")
print(f"Frequency of mode: {mode_res.count}")
print(f"Standard deviation: {std_val:.2f}")
print(f"Variance: {var_val:.2f}")
print(f"Range: {range_val}")
print(f"75th percentile: {p75}")
print(f"90th percentile: {p90}")
```

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python basic_measures.py
   ```

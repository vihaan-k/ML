# Data Distribution

A guide and reference implementation for generating and visualizing dummy datasets using standard probability distributions in Python.

## Overview
When generating synthetic or dummy data, the values can be distributed across a space in various ways depending on the underlying statistical assumptions. This project demonstrates how to generate and plot both **Uniform** and **Normal** distributions using NumPy and Matplotlib.

---

## Mathematical Formulation

### 1. Uniform (Random) Distribution
In a continuous uniform distribution, every value within a specified interval $[a, b]$ has an equal probability of occurring. 

For an array of 500 floats bounded between $0$ and $5$:
```python
numpy.random.uniform(0, 5, 500)
```

### 2. Normal Distribution (Bell Curve)
A normal distribution is symmetric about its mean ($\mu$), meaning data near the mean occurs more frequently than data far from it. Consequently, the mean, median, and mode of the dataset are approximately equal.

The probability density function (PDF) of a normal distribution is defined as:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

The probability $P$ that an observation $X$ falls within $k$ standard deviations ($\sigma$) of the mean ($\mu$) is given by the integral:

$$P(X \in [\mu - k\sigma, \mu + k\sigma]) = \frac{1}{\sigma\sqrt{2\pi}} \int_{\mu - k\sigma}^{\mu + k\sigma} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} dx$$

*(Note: Numerical integration methods, such as Riemann Sums, can be used to approximate this integral.)*

#### Empirical Rule Highlights
* **$\sim 68.3\%$** of observations fall within $1\sigma$ of the mean ($\mu \pm 1\sigma$).
* **$\sim 95.4\%$** of observations fall within $2\sigma$ of the mean ($\mu \pm 2\sigma$).
* **$\sim 99.7\%$** of observations fall within $3\sigma$ of the mean ($\mu \pm 3\sigma$).

To generate $100,000$ values drawn from a normal distribution with $\mu = 5$ and $\sigma = 1$:
```python
numpy.random.normal(5, 1, 100000)
```

Histograms can be visualized using Matplotlib:
```python
matplotlib.pyplot.hist(array, 10)
```

---

## Code Example
The complete Python implementation can be found in [`data_distribution.py`](./data_distribution.py).

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the script
```bash
python data_distribution/data_distribution.py
```

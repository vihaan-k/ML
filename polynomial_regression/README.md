# Polynomial Regression

A Python implementation of Polynomial Regression using SciPy and Matplotlib to analyze and model non-linear relationships between a single input feature and a target variable.

---

## 📌 Overview

Polynomial Regression models the relationship between an independent variable $x$ and a dependent variable $y$ as a $k$-th degree polynomial. Given $n$ sample observations, the algorithm calculates the optimal coefficients to fit a smooth curve through the dataset.

---

## 🧮 Mathematical Formulation

### 1. Model Representation
The predicted output $y$ is modeled as a polynomial function of degree $k$:

$$y = a_0 + a_1 x + a_2 x^2 + \dots + a_k x^k = \sum_{i=0}^{k} a_i x^i$$

For a complete derivation of how the coefficients $a_i$ are solved using linear algebra, see [`MATH.md`](./MATH.md).

---

## 📊 Evaluation Metrics

### Adjusted Coefficient of Determination ($\text{Adjusted } R^2$)
While standard $R^2$ automatically increases when higher-degree terms ($x^2, x^3, \dots$) are added, **Adjusted $R^2$** penalizes unnecessary model complexity, making it ideal for evaluating polynomial regression and preventing overfitting:

$$\bar{R}^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$$

Where standard $R^2$ is defined as:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

* **$n$**: Number of sample observations.
* **$k$**: Degree of the polynomial.
* **$y_i$**: Actual target value.
* **$\hat{y}_i$**: Model predicted value.
* **$\bar{y}$**: Mean of actual target values.

#### Interpretation
* **$\bar{R}^2 \approx 1.0$**: The polynomial model fits the data well without unnecessary complexity.
* **$\bar{R}^2 \le 0.0$**: The model performs no better than predicting the dataset mean $\bar{y}$.
* **Decreasing $\bar{R}^2$ with higher $k$**: Indicates that adding extra polynomial terms is overfitting the noise rather than capturing a real underlying trend.

---

## 💻 Source Code

The complete Python implementation can be found in [`polynomial_regression.py`](./polynomial_regression.py).

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python polynomial_regression/polynomial_regression.py
```
# Linear Regression

An implementation of simple linear regression using Python, SciPy, and Matplotlib to analyze and model the relationship between a single input feature and a target output.

---

## 📌 Overview

Simple Linear Regression is a supervised learning technique used to predict a continuous output ($Y$) based on a single input feature ($X$). Given $n$ sample observations, the model estimates a straight line that best describes the linear relationship within the dataset.

---

## 🧮 Mathematical Formulation

### 1. Model Representation
The predicted output $Y$ is modeled as a linear function of the input feature $X$:

$$Y = aX + b$$

Where:
* **$a$ (Slope):** The rate of change in $Y$ for every unit change in $X$.
* **$b$ (Y-Intercept):** The value of $Y$ when $X = 0$.

> 💡 **Code Connection:** In `linear_regression.py`, this line function is defined as:
> ```python
> def line_func(x):
>     return slope * x + y_intercept
> ```

---

### 2. Parameter Estimation

The optimal values for the slope ($a$) and intercept ($b$) are derived by minimizing the sum of squared errors across all $n$ sample data points:

$$a = \frac{n\sum xy - \sum x\sum y}{n\sum x^2 - (\sum x)^2}$$

$$b = \frac{\sum y - a\sum x}{n}$$

> 💡 **Code Connection:** These values are calculated directly using SciPy's `linregress` function:
> ```python
> slope, y_intercept, r, p, std_err = linregress(x, y)
> ```

---

## 📊 Evaluation Metrics

### Pearson Correlation Coefficient ($r$)
The strength and direction of the linear relationship between $X$ and $Y$ are measured by the Pearson $r$-value, ranging from $-1$ to $1$:

$$r = \frac{n\sum xy - \sum x\sum y}{\sqrt{\left[n\sum x^2 - (\sum x)^2\right] \left[n\sum y^2 - (\sum y)^2\right]}}$$

* $r = 1$: Perfect positive linear correlation.
* $r = -1$: Perfect negative linear correlation.
* $r = 0$: No linear correlation.

### Coefficient of Determination ($R^2$)
The metric $R^2$ (computed as $r^2$) represents the proportion of total variance in target $Y$ explained by the linear model:

$$R^2 = r^2$$

* $R^2 = 1.0$: The model perfectly explains all variability in the training data.
* $R^2 = 0.0$: The model fails to explain any variability in the target output.

> 💡 **Code Connection:** Printed directly to the console:
> ```python
> print("R:", r)
> print("R^2:", r*r)
> ```

---

## 🚀 How to Run

### 1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 2. Run the script
   ```bash
   python linear_regression/linear_regression.py
   ```

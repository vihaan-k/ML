# Multiple Linear Regression

A Python implementation of Multiple Linear Regression using Scikit-learn and Pandas to analyze and model linear relationships between multiple input features and a target variable.

---

## 📌 Overview

Multiple Linear Regression models the relationship between $k$ independent variables $x_j$ (where $j \in \{1, \dots, k\}$) and a dependent variable $y$ as a linear combination of the inputs. Given $n$ sample observations, the algorithm calculates the optimal coefficients for each input feature to minimize prediction error on the target variable.

---

## 🧮 Mathematical Formulation

### 1. Model Representation
The predicted target output $\hat{y}$ is modeled as a linear combination of $k$ feature inputs:

$$\hat{y} = a_0 + a_1 x_1 + a_2 x_2 + \dots + a_k x_k = a_0 + \sum_{j=1}^{k} a_j x_j$$

<!-- For a complete derivation of how the coefficient vector $A$ is solved via matrix operations, see [`MATH.md`](./MATH.md). -->

---

### 2. Feature Scaling (Standardization)
Because input features often differ significantly in scale and unit of measurement (e.g., vehicle weight in kilograms vs. engine volume in cubic centimeters), features are standardized prior to fitting:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

* **$x_{ij}$**: Raw value of feature $j$ for sample $i$.
* **$\mu_j$**: Mean of feature $j$ across all training samples.
* **$\sigma_j$**: Standard deviation of feature $j$ across all training samples.
* **$z_{ij}$**: Standardized $z$-score used in model training and inference.

---

## 📊 Evaluation Metrics

### Adjusted Coefficient of Determination ($\text{Adjusted } R^2$)
While standard $R^2$ automatically increases or stays the same whenever additional predictor variables are added, **Adjusted $R^2$** penalizes unnecessary model complexity, making it ideal for evaluating multiple feature sets and preventing overfitting:

$$\bar{R}^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$$

Where standard $R^2$ is defined as:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

* **$n$**: Number of sample observations.
* **$k$**: Number of predictor features.
* **$y_i$**: Actual target value.
* **$\hat{y}_i$**: Model predicted value.
* **$\bar{y}$**: Mean of actual target values.

#### Interpretation
* **$\bar{R}^2 \approx 1.0$**: The multiple regression model explains a large proportion of variance without unnecessary predictors.
* **$\bar{R}^2 \le 0.0$**: The model performs no better than predicting the target dataset mean $\bar{y}$.
* **Decreasing $\bar{R}^2$ with extra features**: Indicates that adding new input features is adding noise rather than meaningful predictive signal.

---

## 💻 Source Code

The complete Python implementation can be found in [`multiple_regression.py`](./multiple_regression.py), utilizing dataset inputs from [`cars_data.csv`](./cars_data.csv).

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python multiple_regression/multiple_regression.py
```
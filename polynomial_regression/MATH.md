# Math Behind Polynomial Regression

Polynomial regression fits a $k$-degree polynomial curve $f(x)$ to a dataset of $n$ observations:

$$f(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_k x^k = \sum_{j=0}^{k} a_j x^j$$

Given $n$ input values $x_1, x_2, \dots, x_n$ and their corresponding true target values $y_1, y_2, \dots, y_n$, our goal is to find the coefficient vector $A$ that minimizes the discrepancy between the predicted outputs $\hat{Y}$ and the true outputs $Y$.

---

## 1. Matrix Form Formulation

We structure our dataset, predictions, and target values into matrix form:

* **Design Matrix ($X \in \mathbb{R}^{n \times (k+1)}$):**  
  $$X = \begin{bmatrix}
  1 & x_1 & x_1^2 & \dots & x_1^k \\
  1 & x_2 & x_2^2 & \dots & x_2^k \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  1 & x_n & x_n^2 & \dots & x_n^k
  \end{bmatrix} \quad \text{where } x_{ij} = x_i^{j-1} \text{ for } j \in \{1, \dots, k+1\}$$

* **Coefficient Vector ($A \in \mathbb{R}^{(k+1) \times 1}$):**  
  $$A = \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ \vdots \\ a_k \end{bmatrix}$$

* **Target Vector ($Y \in \mathbb{R}^{n \times 1}$) and Predictions ($\hat{Y} \in \mathbb{R}^{n \times 1}$):**  
  $$Y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \quad \hat{Y} = XA = \begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n \end{bmatrix}$$

---

## 2. Defining the Cost Function (SSE)

We quantify overall model error using the Sum of Squared Errors (SSE). First, define the residual error vector $E = Y - \hat{Y} = Y - XA$. 

The single-value cost function $C(A) \in \mathbb{R}^{1 \times 1}$ is expressed as the inner product of the error vector with itself:

$$\begin{align*}
C(A) &= E^T E \\
&= (Y - XA)^T (Y - XA) \\
&= (Y^T - A^T X^T)(Y - XA) \\
&= Y^T Y - Y^T XA - A^T X^T Y + A^T X^T X A
\end{align*}$$

Because $Y^T XA$ evaluates to a scalar ($1 \times 1$ matrix), it equals its own transpose $(Y^T XA)^T = A^T X^T Y$. Combining these terms yields:

$$C(A) = Y^T Y - 2 Y^T XA + A^T X^T X A$$

---

## 3. Finding the Critical Points (First Derivative)

To find the vector $A$ that minimizes the cost function $C(A)$, we compute the gradient with respect to $A$ using matrix calculus rules:
$$\nabla_A (A^T M) = M \quad \text{and} \quad \nabla_A (A^T M A) = 2 M A \quad \text{(for symmetric } M\text{)}$$

Applying these identities to $C(A)$:

$$\nabla_A C(A) = -2 X^T Y + 2 X^T X A$$

Setting the gradient to zero to locate the critical values:

$$2 X^T X A - 2 X^T Y = 0 \iff X^T X A = X^T Y$$

These are the classical **Normal Equations**.

* **If $X^T X$ is invertible:**  
  $$A = (X^T X)^{-1} X^T Y$$

* **If $X^T X$ is singular (non-invertible):**  
  We compute the Moore-Penrose Pseudoinverse $X^+$ via Singular Value Decomposition (SVD):  
  $$A = X^+ Y$$

---

## 4. Confirming a Global Minimum (Second Derivative)

To confirm that our critical point is a global minimum, we examine the Hessian matrix of $C(A)$ with respect to $A$:

$$\nabla_A^2 C(A) = 2 X^T X$$

For any non-zero column vector $v \neq 0$:

$$v^T (2 X^T X) v = 2 (X v)^T (X v) = 2 \|X v\|^2 \ge 0$$

Assuming $X$ has full column rank, $Xv \neq 0$, making $2 X^T X$ strictly **positive definite**. 

Because the Hessian is positive definite everywhere, the cost function $C(A)$ is strictly convex. Therefore, the critical point $A$ found via the Normal Equations is the unique **global minimum**.
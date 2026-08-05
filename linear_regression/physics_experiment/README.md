# Air Pressure vs. Temperature Analysis

A practical application of linear regression using real-world physical data to demonstrate Gay-Lussac's Law (the linear relationship between air temperature and air pressure).

---

## Physical Background & Mathematical Model

When the volume of a gas is held constant, its pressure ($p$) varies linearly with its temperature ($\theta$).

### 1. Linear Regression Model
The relationship between temperature in Celsius ($\theta$) and pressure in hectopascals ($p$) is modeled using the line of best fit equation:

$$p = a\theta + b$$

Where:
* $a$ is the **slope** ($\text{hPa}/^\circ\text{C}$), representing the rate of pressure change per degree Celsius.
* $b$ is the **y-intercept** ($\text{hPa}$), representing the pressure at $0^\circ\text{C}$.

### 2. Goodness of Fit ($R$ and $R^2$)
To evaluate how accurately the linear model represents the empirical data:
* **Pearson Correlation Coefficient ($R$):** Measures the strength and direction of the linear relationship between temperature and pressure.
* **Coefficient of Determination ($R^2$):** Represents the proportion of variance in air pressure predictable from temperature.

### 3. Experimental Data
Using `scipy.stats.linregress`, these parameters are computed from the following experimental measurements:
```python
x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Temperature of air in °C
y = [1013, 1050, 1087, 1115, 1152, 1190, 1227, 1265, 1300, 1341, 1380] # Air pressure in hPa
```

---

## Expected Output & Results

Running the regression script yields the following physical parameters:

* **Linear Equation:** $p = 3.664 \cdot \theta + 1013.8$
* **Slope ($a$):** $\approx 3.664 \text{ hPa}/^\circ\text{C}$
* **Y-intercept ($b$):** $\approx 1013.8 \text{ hPa}$
* **Correlation ($R$):** $\approx 0.9998$
* **Goodness of Fit ($R^2$):** $\approx 0.9996$

An $R^2$ value of $0.9996$ confirms an extremely strong linear relationship between temperature and pressure, matching theoretical expectations for Gay-Lussac's Law.

---

## Source Code

The complete Python implementation can be found in [`physics_experiment.py`](./physics_experiment.py).

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the script
```bash
python physics_experiment/physics_experiment.py
```
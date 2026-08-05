import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

filepath = "categorical-data\cars-data.csv"
data = pd.read_csv(filepath)

# Columns are made for every car brand, and values are 0 or 1. The first column is omitted.
ohe_data = pd.get_dummies(data[["Car"]], drop_first=True) 

print(ohe_data.to_string())

X = pd.concat([data[["Volume", "Weight"]], ohe_data], axis=1)
# Axis determines the direction of the concatenation. 0 is vertical, 1 is horizontal.

scale = StandardScaler()
scaledX = scale.fit_transform(X)

y = data["CO2"]

regr = LinearRegression()
regr.fit(X, y)

scaled_inputs = scale.transform([[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
# Predict the CO2 emission of a VW with weight = 2300 and volume = 1300

print(regr.predict(scaled_inputs))
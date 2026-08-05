from csv import DictReader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

filepath = "multiple-regression\cars-data.csv"

'''
with open(filepath, "r", encoding="utf-8") as csv_file:
    data = [_ for _ in DictReader(csv_file)]

weight = [float(car["Weight"]) for car in data]
volume = [float(car["Volume"]) for car in data]
co2 = [float(car["CO2"]) for car in data]

X = np.array([weight, volume]).T
y = np.array(co2)'''

data = pd.read_csv(filepath)

X = data[["Weight", "Volume"]]
y = data["CO2"]

scale = StandardScaler()
scaledX = scale.fit_transform(X) 
# fit_transform is used when storing the mean and sd within the scale object

regr = LinearRegression()
regr.fit(scaledX, y)

scaled_inputs = scale.transform([[2300, 1300]])
# transform is used when using the stored mean and sd to scale

print("Coefficients:", regr.coef_)
print("Example Prediction (Input: [2300, 1300]):", regr.predict(scaled_inputs)[0])
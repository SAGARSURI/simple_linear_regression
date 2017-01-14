import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

with open('challenge_dataset.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('challenge_dataset.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('title', 'intro'))
        writer.writerows(lines)

#read data
dataframe = pd.read_csv('challenge_dataset.csv')

#splitting the data into training/testing sets 
x_total_values = dataframe[['title']]
x_train_values = x_total_values[:-20]
x_test_values = x_total_values[-20:]


y_total_values = dataframe[['intro']]
y_train_values = y_total_values[:-20]
y_test_values = y_total_values[-20:]

print(y_test_values)

#train model on data
reg = linear_model.LinearRegression()
reg.fit(x_train_values,y_train_values)

print("Mean squared error {}".format(np.mean((reg.predict(x_test_values) - y_test_values) ** 2)))

#visualise data
plt.scatter(x_test_values, y_test_values)
plt.plot(x_test_values, reg.predict(x_test_values))
plt.show()
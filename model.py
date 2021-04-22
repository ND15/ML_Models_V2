# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:01:46 2021

@author: Nikhil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("Fish.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state=0)
print(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[4]])) 
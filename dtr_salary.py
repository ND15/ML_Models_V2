# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:28:13 2021

@author: User
"""

import pandas as pd
import numpy as np
import pickle

dataset = pd.read_csv("Fish.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

pickle.dump(regressor, open('dtr_pkl.pkl', 'wb'))
model_dtr = pickle.load(open('dtr_pkl.pkl', 'rb'))

print(model_dtr.predict([[4.2]]))
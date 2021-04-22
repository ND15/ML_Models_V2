# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:46:27 2021

@author: User
"""

import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('Fish.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

pickle.dump(sc_x, open('svr_pkl_sx.pkl','wb'))
s_x = pickle.load(open('svr_pkl_sx.pkl','rb'))

pickle.dump(sc_y, open('svr_pkl_sy.pkl','wb'))
s_y = pickle.load(open('svr_pkl_sy.pkl','rb'))


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

pickle.dump(regressor,open('svr_pkl.pkl','wb'))

model_svr = pickle.load(open('svr_pkl.pkl', 'rb'))

print(s_y.inverse_transform(model_svr.predict(s_x.transform([[4]]))))
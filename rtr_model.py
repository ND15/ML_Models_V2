import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv("Fish.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x,y)

pickle.dump(regressor, open('rtr_pkl.pkl', 'wb'))

rtr_model = pickle.load(open('rtr_pkl.pkl', 'rb'))

print(rtr_model.predict([[8]]))
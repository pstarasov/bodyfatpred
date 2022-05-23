# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:09:16 2022

@author: root
"""
import pandas as pd    # Data processing, Input & Output load    
import numpy as np
from sklearn.ensemble import AdaBoostRegressor    # Ada Boost Regressor

import pickle
from sklearn.model_selection import train_test_split as tts


data = pd.read_csv(r"C:\Users\root\datasets\bodyfat.csv")

data_new = data.copy()
data_new = data_new.drop('Density', axis = 1)
data_new = data_new.loc[data_new['BodyFat'] != 0.0 ]
data_new = data_new.loc[data_new['Height'] != 29.50]
data_new = data_new.loc[data_new['BodyFat'] != 0.7 ]
data_new = data_new.loc[data_new['BodyFat'] != 47.5]

data_new['Height'] = data_new['Height'] * 2.54
data_new['Weight'] = data_new['Weight'] * 0.454

X = data_new.drop(['BodyFat'], axis = 1)
y = data_new['BodyFat']
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 42) 


model = AdaBoostRegressor(learning_rate = 0.1, loss = 'square', n_estimators = 80, random_state = 42)
model.fit(X_train, y_train)
#pred=model.predict([np.array([23,70,172,36,93,85,94,59,37,22,32,27,17]).astype(float)])
#print(pred)
pickle.dump(model, open("model.pkl", 'wb'))
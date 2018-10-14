# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:39:00 2018

@author: guna_
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
x = df.iloc[:,1:25].values
y = df.iloc[:,25].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

import time

time_start = time.clock()
#run your code
time_elapsed = (time.clock() - time_start)
import sklearn
import numpy as np
import pandas as py
from sklearn import linear_model
x=np.array([[5],[15],[25],[35],[45],[55]])
y=np.array([5,20,14,32,22,38])

model= linear_model.LinearRegression()
model.fit(x,y)
x_new = np.array([150]).reshape((-1,1))
y_new = model.predict(x_new)
print(y_new)
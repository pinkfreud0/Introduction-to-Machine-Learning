import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Load the data
df = pd.read_csv("mlr05.csv",)
arr=df.to_numpy()

X=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
X_pred=[19, 20, 21, 22, 23, 24, 25, 26]
y=arr[0:20,0]
y_real=arr[19:,0]
plt.ylabel('sales')
plt.plot(X,df.X1[0:20])

reg= linear_model.LinearRegression()
reg.fit(df[0:20],df.X1[0:20])
y_pred=reg.predict(df[19:])
plt.plot(X_pred, y_pred)

plt.plot(X_pred,y_real)
plt.title('real values')
Diff=y_real-y_pred
print("Difference between prediction and real value is: ")
print(Diff)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import glob
import os


# Importing data from one path with multiple training files.
path = r'C:\Users\Michel\Desktop\UMD\Fall22\ENPM808A\Final project\Training\Training_data' # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None,low_memory=False)
    X=df.iloc[:,:-14]
    y=df.iloc[:,-14:]
    youssef=X.groupby(np.arange(len(X.columns))//4,axis=1).mean()
    karam=pd.concat([youssef,y],axis="columns")
    li.append(karam)
    # frame = pd.concat(karam, axis=0, ignore_index=True)
df=pd.concat(li)    
df_=df.to_numpy()
X=df_[:,:-2]
Y=df_[:,-2:]


min_max_scaler=preprocessing.MinMaxScaler()
X_scaled=min_max_scaler.fit_transform(X)

Y_scaled=min_max_scaler.fit_transform(Y)
in_rows,in_features=X.shape
_,out_features=Y.shape

# #Importing and preprocessing test set
testdf= pd.read_csv(r"C:\Users\Michel\Desktop\UMD\Fall22\ENPM808A\Final project\Aug14_Box_g1.csv")        
Xtst=testdf.iloc[:,:-14]
ytst=testdf.iloc[:,-14:]
yousseftst=Xtst.groupby(np.arange(len(Xtst.columns))//4,axis=1).mean()
karamtst=pd.concat([yousseftst,ytst],axis="columns")
testdf_=karamtst.to_numpy()
X_t=testdf_[:,:-2]
Y_t=testdf_[:,-2:]
Xtest=min_max_scaler.fit_transform(X_t)
Ytest=min_max_scaler.fit_transform(Y_t)



#Defining the first NN Network with relu activation
model1 = Sequential()
model1.add(Dense(100, input_dim=in_features, activation="relu"))
model1.add(Dense(64, activation="relu"))
model1.add(Dense(16, activation="relu"))
model1.add(Dense(out_features))
model1.compile(loss="mse", optimizer="adam")
#Fitting first model
history= model1.fit(X_scaled, Y_scaled, validation_split=0.1, epochs=100, batch_size=100, verbose=0)
# Calculating Loss in Model1
# min_Ein1_4=min(history.history["loss"])


# ########################################################
#Defining the second model with sigmoid activation
model2= Sequential()
model2.add(Dense(100, input_dim=in_features, activation="sigmoid"))
model2.add(Dense(64, activation="sigmoid"))
model2.add(Dense(16, activation='sigmoid'))
model2.add(Dense(out_features))
model2.compile(loss="mse", optimizer="adam")
#Fitting the second model, then predicting training model
history2=model2.fit(X_scaled, Y_scaled,validation_split=0.1, epochs=100, batch_size=100, verbose=0)
# Calculating loss in Model2
# # min_Ein2=min(history2.history["loss"])

#Predicting test dataset with model 1 and 2
Y_test_predicted1=model1.predict(Xtest)
Y_test_predicted2=model2.predict(Xtest)

#calculating out-of-sample error using Mean-Squared Error
mse = tf.keras.losses.MeanSquaredError()
E_out1=mse(Ytest,Y_test_predicted1).numpy()
E_out2=mse(Ytest,Y_test_predicted2).numpy()

# # Visualizing E_in and E_val for the two models
# fig, ax = plt.subplots(2)
# ax=plt.subplot(1,2,1)
# plt.plot(history.history['loss'], 'r')
# plt.plot(history.history['val_loss'],'b')
# ax.set_ylim([0, 0.15])
# ax.set_title('Relu activation')
# ax=plt.subplot(1,2,2)
# plt.plot(history2.history['loss'], 'r')
# plt.plot(history2.history['val_loss'],'b')
# ax.set_ylim([0, 0.15])
# ax.set_title('Sigmoid activation')
# plt.legend(['in-sample error', 'validation error'])#,'error in validation sample'])
# plt.xlabel('Epoch')
# plt.show()


# #Visualizing Model1 Predictions in a plot
# Ypred1=model1.predict(X_scaled[:100,:])
# fig, ax = plt.subplots(2)
# fig.suptitle('Model1: Relu activation func, validation set=0%, 3 Layers')
# ax=plt.subplot(2,1,1)
# plt.plot(X_scaled[:100,1],Y_scaled[:100,1], 'ro')
# plt.plot(X_scaled[:100,1],Ypred1[:100,1],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('w')
# ax=plt.subplot(2,1,2)
# plt.plot(X_scaled[:100,0],Y_scaled[:100,0], 'ro')
# plt.plot(X_scaled[:100,0],Ypred1[:100,0],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('v')
# plt.legend(['Real value','Predited value'])
# plt.show()


# #Visualizing Model2 Predictions in a plot
# fig, ax = plt.subplots(2)
# Ypred2=model2.predict(X_scaled)
# fig.suptitle('Model2: Sigmoid activation func, validation set=0%, 3 Layers')
# ax=plt.subplot(2,1,1)
# plt.plot(X_scaled[:100,1],Y_scaled[:100,1], 'ro')
# plt.plot(X_scaled[:100,1],Ypred2[:100,1],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('w')
# ax=plt.subplot(2,1,2)
# plt.plot(X_scaled[:100,0],Y_scaled[:100,0], 'ro')
# plt.plot(X_scaled[:100,0],Ypred2[:100,0],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('v')
# plt.legend(['Real value','Predited value'])
# plt.show()

# # #Visualizing prediction of test set
# fig, ax = plt.subplots(2)
# # fig.suptitle('Model1: Relu activation func, validation set=0%, 3 Layers')
# ax=plt.subplot(2,1,1)
# plt.plot(Xtest[:100,1],Ytest[:100,1], 'ro')
# plt.plot(Xtest[:100,1],Y_test_predicted1[:100,1],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('w')
# ax=plt.subplot(2,1,2)
# plt.plot(Xtest[:100,0],Ytest[:100,0], 'ro')
# plt.plot(Xtest[:100,0],Y_test_predicted1[:100,0],'bo')
# ax.set_ylim([0,1.2])
# ax.set_title('v')
# plt.legend(['Real Test Value','Predicted  Test Value'])
# plt.show()


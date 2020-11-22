from sklearn import neighbors
import numpy as np
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:6]
y=test[:,-1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled)
#X_train_scaled = scaler.fit_transform(X_train)
#X_train = pd.DataFrame(X_train_scaled)
#X_test_scaled = scaler.fit_transform(X_test)
#X_test = pd.DataFrame(X_test_scaled)

rmse_val = [] #to store rmse values for different k
#for K in range(20):
#    K = K+1
#    model = neighbors.KNeighborsRegressor(n_neighbors = K)

#    model.fit(X_train, y_train)  #fit the model
#    pred=model.predict(X_test) #make prediction on test set
#    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
#    rmse_val.append(error) #store rmse values
#    print('RMSE value for k= ' , K , 'is:', error)

crossvalidation = KFold(n_splits=10, random_state=7)
for K in range(20):
    K=K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X,y)
    scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
    RMSE=np.sqrt(-scores.mean())
    rmse_val.append(RMSE) #store rmse values
    print('RMSE value for k= ' , K , 'is:', RMSE)
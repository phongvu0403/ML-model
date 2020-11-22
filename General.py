import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from math import sqrt
from sklearn import ensemble

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:5]
y=test[:,6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

RMSE={'RMSE_mlg':[],'RMSE_pr':[],'RMSE_dr':[],'RMSE_KNN':[],'RMSE_xgr':[],'RMSE_rfr':[]}
#multiple linear regression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
RMSE_mlg = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE_mlg)
RMSE['RMSE_mlg'].append(RMSE_mlg)

#polinomial regression
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
x_test_=PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
model = LinearRegression().fit(x_, y_train)
y_pred = model.predict(x_test_)
RMSE_pr = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE_pr)
RMSE['RMSE_pr'].append(RMSE_pr)

#Decisiontree regressor
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, y_train)
y_pred = regr_1.predict(X_test)
RMSE_dr = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE_dr)
RMSE['RMSE_dr'].append(RMSE_dr)

#KNN Regressor
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
#    print('RMSE value for k= ' , K , 'is:', error)

RMSE_minKNN = min(rmse_val)
RMSE['RMSE_KNN'].append(RMSE_minKNN)

#GradientBoosting
params={'n_estimators':100,'max_depth':5,'learning_rate':0.1,'criterion':'mse'}
gradient_boosting_regressor_model=ensemble.GradientBoostingRegressor(**params)
gradient_boosting_regressor_model.fit(X_train,y_train)
y_pred = gradient_boosting_regressor_model.predict(X_test)
RMSE_xgr = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE_xgr)
RMSE['RMSE_xgr'].append(RMSE_xgr)

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
RMSE_rfr = np.sqrt(mean_squared_error(y_test,y_pred))
#print(RMSE_rfr)
RMSE['RMSE_rfr'].append(RMSE_rfr)


#Extremely Gradient Boosting Regression














print(RMSE)
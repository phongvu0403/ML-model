import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd 
#import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#import plot_learning_curve

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:6]
y=test[:,-1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

params={'n_estimators':100,'max_depth':5,'learning_rate':0.1,'criterion':'mse'}
gradient_boosting_regressor_model=ensemble.GradientBoostingRegressor(**params)
crossvalidation = KFold(n_splits=10, random_state=7)
model=gradient_boosting_regressor_model.fit(X,y)
scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)

#y_pred = gradient_boosting_regressor_model.predict(X_test)
#RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

RMSE=np.sqrt(-scores.mean()) 
print(RMSE)

#plt.figure(figsize=(12,6))
#plt.titile('Gradient Boosting model')
#plt.scatter(X_train, y_train)
#plt.plot(X_train,gradient_boosting_regressor_model.predict(X_test),color='black')
#plt.show()

#print(gradient_boosting_regressor_model.score(X_train,y_train))
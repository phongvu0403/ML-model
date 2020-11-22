import xgboost as xgb
import pandas as pd
import numpy as np
#import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:6]
y=test[:,-1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
crossvalidation = KFold(n_splits=10, random_state=7, shuffle=False)
#DM_train = xgb.DMatrix(data = X_train, label = y_train)  
#DM_test =  xgb.DMatrix(data = X_test,label = y_test)
DM=xgb.DMatrix(data=X,label=y)

gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[100, 200],
     'max_depth': [10, 15, 20, 25]
}

#gbm = xgb.XGBRegressor()
#grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1)
#grid_mse.fit(X_train, y_train)
#print("Best parameters found: ",grid_mse.best_params_)
#print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = crossvalidation, verbose = 1)
model=grid_mse.fit(X, y)
scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
RMSE=np.sqrt(-scores.mean())
print(RMSE)
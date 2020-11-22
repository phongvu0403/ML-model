import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from pprint import pprint
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:5]
y=test[:,6]

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [ 70, 80, 90],
    'min_samples_leaf': [2, 3],
    'min_samples_split': [2, 3],
    'n_estimators': [1800, 1850, 1900]
}

# Create a based model
rf = RandomForestRegressor()

crossvalidation = KFold(n_splits=10, random_state=7)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = crossvalidation, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)
print(grid_search.best_params_)
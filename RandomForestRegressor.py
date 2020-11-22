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
csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:6]
y=test[:,-1]
#feature_list = list(features.columns)
#features = np.array(features)
#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)

#baseline_preds = test_features[:, feature_list.index('average')]
#baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))
crossvalidation = KFold(n_splits=10, random_state=7, shuffle=False)
rf = RandomForestRegressor( max_depth = 90, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 1800)
#rf=RandomForestRegressor()
model=rf.fit(X, y)
scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
#predictions = rf.predict(test_features)
#RMSE = np.sqrt(mean_squared_error(test_labels,predictions))
RMSE=np.sqrt(-scores.mean()) 
print(RMSE) 
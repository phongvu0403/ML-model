import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib as plt
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
#model = LinearRegression().fit(X_train, y_train)
crossvalidation = KFold(n_splits=10, random_state=7)
model=LinearRegression().fit(X,y)
scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
#cv_results.mean()
#y_pred = model.predict(X_test)
#RMSE = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE)
#print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
RMSE=np.sqrt(-scores.mean())
print(RMSE)
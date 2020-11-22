import numpy as np
import matplotlib as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
X=test[:,0:6]
y=test[:,-1]
y=np.reshape(y,(-1,1))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#print(y_test)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#X_test = sc_X.fit_transform(X_test)
#y_test = sc_y.fit_transform(y_test)
crossvalidation = KFold(n_splits=10, random_state=7)
regressor = SVR(kernel='rbf')
model=regressor.fit(X,y)
scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
RMSE=np.sqrt(-scores.mean())
#y_pred = regressor.predict(X)
#RMSE = np.sqrt(mean_squared_error(y_test,y_pred)) 
print(RMSE)
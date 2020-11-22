import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
#x_test_=PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
#model = LinearRegression().fit(x_, y_train)
#y_pred = model.predict(x_test_)
#RMSE = np.sqrt(mean_squared_error(y_test,y_pred)) 
#print(RMSE) 

lm = skl_lm.LinearRegression()

crossvalidation = KFold(n_splits=10, random_state=7, shuffle=False)

for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=crossvalidation,
 n_jobs=1)
    
    #print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
    RMSE=np.sqrt(-scores.mean())
    print(RMSE)
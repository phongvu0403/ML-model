import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib as plt
from sklearn.tree import DecisionTreeRegressor
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
crossvalidation = KFold(n_splits=10, random_state=7)
regr_1 = DecisionTreeRegressor(max_depth=5)
model=regr_1.fit(X, y)
#y_pred = regr_1.predict(X_test)
#RMSE = np.sqrt(mean_squared_error(y_test,y_pred)) 

scores=cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation)
RMSE=np.sqrt(-scores.mean())
print(RMSE)


# Plot the results
#plt.figure()
#plt.scatter(X, y, s=20, edgecolor="black",
#            c="darkorange", label="data")
#plt.plot(X_test, y_1, color="cornflowerblue",
#         label="max_depth=2", linewidth=2)
#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
# Regression Example With Boston Dataset: Standardized
#from pandas import read_csv
from google.colab import drive
drive.mount('/content/gdrive')

import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from sklearn import metrics

csv_file = '/content/gdrive/My Drive/listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
x=test[:,0:6]
y=test[:,-1]
y=np.reshape(y,(-1,1))
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.25, random_state=42)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
RMSE=[]
for train, test in kfold.split(inputs, targets):
  #print(inputs[train])
  # Define the model architecture
  model = Sequential()
  model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
  model.add(Dense(27, activation='relu')) # Hidden 2
  model.add(Dense(1)) # Output

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  history = model.fit(inputs[train], targets[train],batch_size=150,epochs=150,verbose=1)

  # Predict
  pred = model.predict(x_test)

  # Measure MSE error.  
  score = metrics.mean_squared_error(pred,y_test)
  print("Final score (MSE): {}".format(score))

  # Measure RMSE error.  RMSE is common for regression.
  score = np.sqrt(metrics.mean_squared_error(pred,y_test))
  print("Final score (RMSE): {}".format(score))
  fold_no = fold_no + 1
  RMSE.append(score)

print(RMSE)



#result
result=0
for score in RMSE:
  result+=score
result=result/10
print(result)

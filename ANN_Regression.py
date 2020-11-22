# Regression Example With Boston Dataset: Standardized
#from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# load dataset
csv_file = 'listStoreValue-long.csv'
test = np.array(pd.read_csv(csv_file))
# split into input (X) and output (Y) variables
X=test[:,0:6]
Y=test[:,-1]
Y=np.reshape(Y,(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

num_features = len(X_train[1,:])


model = Sequential()
model.add(Dense(13, input_dim=num_features, kernel_initializer='normal', activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(X_train, y_train, epochs=30, batch_size=150, verbose=1, cv=10)

score_mse_test = model.evaluate(X_test, y_test)
print('Test Score:', score_mse_test)

score_mse_train = model.evaluate(X_train, y_train)
print('Train Score:', score_mse_train)
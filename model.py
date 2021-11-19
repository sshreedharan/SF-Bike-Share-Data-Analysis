# ECE 143 Final Project - Model
# Date: 11/18/2021
# Author: Haoyu Wang

# Description:
# This file builds three models to predict the number of daily rentals using
# weather and date. The three models are a linear regression model, a random
# forest model and a neural network model.
# To run this file, make sure this file, 'trip.csv' and 'weather.csv' are under
# the same folder.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers,callbacks

# Count the number of daily rentals using trip.csv
df_trip = pd.read_csv('trip.csv')
df_trip.drop(df_trip.columns.difference(['start_date']), 1, inplace=True)
df_trip['start_date'] = df_trip['start_date']\
.apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))

# Y as pd.Series with date as index and number of rentals as values
Y = df_trip.groupby(['start_date']).size()
Y.name = 'rentals'

# Build a DateFrame with weather data.
df_weather = pd.read_csv('weather.csv')
df_weather.drop(['events','zip_code'],axis=1,inplace=True)

# X_full = Mean daily weather data + weekend(1 for weekends 0 for weekdays)
df_weather['date'] = df_weather['date']\
.apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
X_full = df_weather.groupby(['date']).mean()
X_full['weekend'] = X_full.apply(lambda x:\
 pd.Timestamp(x.name).weekday() in [5,6], 1, 0)

# Fillin max_gust_speed_mph na with its mean
X_full = X_full.fillna(int(X_full.max_gust_speed_mph.mean()))

# Mutual Information
colnames = ['max_TEMP', 'mean_TEMP', 'min_TEMP',
       'max_DWPNT', 'mean_DWPNT', 'min_DWPNT',
       'max_HUM', 'mean_HUM', 'min_HUM',
       'max_SLP', 'mean_SLP',
       'min_SLP', 'max_VIS',
       'mean_VIS', 'min_VIS', 'max_wind_speed',
       'mean_wind_speed', 'max_gust_speed', 'cloud_cover',
       'wind_dir', 'weekend']
mi_scores = mutual_info_regression(X_full, Y)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=colnames)
mi_scores = mi_scores.sort_values(ascending=True)

# Visualize MI scores
plt.figure()
plt.rcParams.update({'font.size': 18})
mi_scores.plot.barh(figsize=(18,10),title='Mutual Information Scores')
plt.show()

# Split the train set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_full,Y,
    test_size=0.2,random_state=1)

# Scaling using Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns,
    index=X_train.index)


# Linear regression model and coefficients:
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled,y_train)
coeff = [(int(i*100))/100 for i in lin_reg.coef_]
df_lincoef = pd.DataFrame({'coef':coeff},index=X_train.columns)
df_lincoef = df_lincoef.sort_values(by=['coef'],key=abs,ascending=False)
print('Linear Regression Model:\nCoefficients:')
print(df_lincoef,'\n')

# Linear regression MSE on training set and validation set
X_val_scaled = scaler.transform(X_val)
lin_err = mean_squared_error(y_train,lin_reg.predict(X_train_scaled))
lin_val_err = mean_squared_error(y_val,lin_reg.predict(X_val_scaled))
print('Training MSE: ',lin_err,'\nValidation MSE: ',lin_val_err)


# Random forest model
# Parameter tuning with grid search
est_list = [70,75,80,85,90,95,100]
mxd_list = [7,8,9,10,11,12]
min_val_err = None

# Choose the best params for rf model based on validation error
for i in est_list:
    for j in mxd_list:
        rf_reg = RandomForestRegressor(n_estimators=i,max_depth=j,random_state=1)
        rf_reg.fit(X_train_scaled,y_train)
        rf_err = mean_squared_error(y_val,rf_reg.predict(X_val_scaled))
        if not min_val_err:
            min_val_err = rf_err
            best_param = (i,j)
            continue
        if rf_err < min_val_err:
            min_val_err = rf_err
            best_param = (i,j)

# Print the MSE errors
rf_reg = RandomForestRegressor(n_estimators=best_param[0],
    max_depth=best_param[1],random_state=1)
rf_reg.fit(X_train_scaled,y_train)
rf_err = mean_squared_error(y_train,rf_reg.predict(X_train_scaled))
print('Random Forest Model:')
print('Training MSE: ',rf_err,'\nValidation MSE: ',min_val_err)


# Neural network model:
# Early stopping callback
early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 20,
    restore_best_weights = True,
)

# Build the model
model = keras.Sequential([
    layers.Dense(units=12, activation='relu'),
    layers.Dense(units=12, activation='relu'),
    layers.Dense(units=12, activation='relu'),
    layers.Dense(units=12, activation='relu'),
    layers.Dense(units=1, activation='linear'),
])

model.compile(
    #optimizer='adam',
    loss='mse',
)

# Fit the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data = (X_val_scaled, y_val),
    batch_size = 20,
    epochs = 500,
    callbacks=[early_stopping],
    verbose = 0,
)

# Compute the errors
nn_err = mean_squared_error(y_train,model.predict(X_train_scaled))
nn_val_err = mean_squared_error(y_val,model.predict(X_val_scaled))

print('Neural Network Model:')
print('Training MSE: ',nn_err,'\nValidation MSE: ',nn_val_err)

# Predict new cases with rf model
# Generate a random row, assume it to be new case
np.random.seed(seed=42)
idx = int(np.random.rand()*len(X_full))
rand_row = X_full.iloc[idx,]
rand_pred = rf_reg.predict(scaler.transform(np.array(rand_row).reshape(1,-1)))

# Plot the prediction
df_newcase = pd.DataFrame({'Max':[67.0,51.2,86.0,29.98,10.0,21.2,26],\
'Mean':[59.8,48.8,69.0,29.95,9.6,9.0,0],'Min':[52.2,46.2,51.2,29.92,7.8,0,0]},
    index=['TEMP','DWPNT','HUM','SLP','VIS','Wind_S','Gust_S'])
plt.figure()
plt.rcParams.update({'font.size': 18})
ax = df_newcase.plot.barh(figsize=(18,10))
plt.text(y=4,x=65,s='cloud_cover=2.6\nwind_dir_degree=309.8\nweekend=False')
plt.show()
print('\nThe predicted number of daily rentals is',int(rand_pred[0]))

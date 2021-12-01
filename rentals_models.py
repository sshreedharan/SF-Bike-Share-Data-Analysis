# ECE 143 Final Project - Model
# Date: 11/18/2021
# Author: Haoyu Wang

# Description:
# This file builds three models to predict the number of daily rentals using
# weather and date. The three models are a linear regression model, a random
# forest model and a neural network model. After calcaulating the RMSE of these
# models, we can find that the random forest model has the lowest training RMSE.
# Then we predict the number of daily rentals for a given day using the random
# forest model.

# Functions:
#   prepare_data(df_trip,df_weather)
#   mutual_info_scores(X,Y)
#   split_train_valid_sets(X,Y)
#   scale_data(X_train)
#   linear_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
#   random_forest_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
#   neural_network_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
#   predict_data(reg,X,scaler,visual)
#   full_model(df_trip,df_weather,visual,less_features)

# To run all parts of this file, call
#   full_model(df_trip,df_weather,visual,less_features),
#   where df_trip is trip pd.DataFrame and df_weather is weather pd.DataFrame.
#   It is recommended to set visual=True and less_features=False.


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

def prepare_data(df_trip,df_weather):
    '''
    This function prepare the data for model.
    Y = number of daily rentals.
    X = mean weather data + binary variable "weekend"

    :param df_trip: trip data           :type: pd.DataFrame
    :param df_weather: weather data     :type: pd.DataFrame
    :rtype: tuple (X,Y) - (pd.DataFrame,pd.Series)
    '''

    # Check the input:
    assert(type(df_trip)==pd.DataFrame)
    assert(type(df_weather)==pd.DataFrame)

    # Get Y, the number of daily rentals, using df_trip
    df_trip.drop(df_trip.columns.difference(['start_date']),1,inplace=True)
    df_trip['start_date'] = df_trip['start_date']\
    .apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))

    # Y: pd.Series, index=data, values=daily rentals
    Y = df_trip.groupby(['start_date']).size()
    Y.name = 'rentals'


    # Get X, mean weather data + a binary variable 'weekend'
    # 'weekend': 1 for weekends, 0 for weekdays
    df_weather.drop(['events','zip_code'],axis=1,inplace=True)
    df_weather['date'] = df_weather['date']\
    .apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))

    X = df_weather.groupby(['date']).mean()
    X['weekend'] = X.apply(lambda x:\
     pd.Timestamp(x.name).weekday() in [5,6], 1, 0)

    # Fillin max_gust_speed_mph na with its mean
    X = X.fillna(int(X.max_gust_speed_mph.mean()))

    return X,Y


def mutual_info_scores(X,Y):
    '''
    This function visualizes the mutual information scores of X and Y
    using bar chart.
    :param X: weather data and 'weekend'    :type: pd.DataFrame
    :param Y: number of daily rentals       :type: pd.Series
    :rtype: None
    '''

    # Check the input:
    assert(type(X)==pd.DataFrame)
    assert(type(Y)==pd.Series)

    # Short representations for colunms' names in X:
    #   TEMP -> temperature
    #   DWPNT -> Dew Point
    #   HUM -> Humidity
    #   SLP -> Sea Level Pressure
    #   VIS -> Visibility
    colnames = ['max_TEMP', 'mean_TEMP', 'min_TEMP',
           'max_DWPNT', 'mean_DWPNT', 'min_DWPNT',
           'max_HUM', 'mean_HUM', 'min_HUM',
           'max_SLP', 'mean_SLP',
           'min_SLP', 'max_VIS',
           'mean_VIS', 'min_VIS', 'max_wind_speed',
           'mean_wind_speed', 'max_gust_speed', 'cloud_cover',
           'wind_dir', 'weekend']

    # Compute MI scores using sklearn module
    mi_scores = mutual_info_regression(X, Y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=colnames)
    mi_scores = mi_scores.sort_values(ascending=True)

    # Visualize MI scores
    plt.figure()
    mi_scores.plot.barh(title='Mutual Information Scores',figsize=(10,8))
    plt.xlabel('Scores',fontsize=16)
    plt.ylabel('Features',fontsize=16)
    plt.show()


def split_train_valid_sets(X,Y):
    '''
    This function splits the X and Y into training sets and validation sets.
    Set test_size ratio = 0.2, random_state = 1
    :param X: weather data and 'weekend'    :type: pd.DataFrame
    :param Y: number of daily rentals       :type: pd.Series
    :return: (X_train, X_val, y_train, y_val)
    :rtype: (pd.DataFrame,pd.DataFrame,pd.Series,pd.Series)
    '''

    # Check the input:
    assert(type(X)==pd.DataFrame)
    assert(type(Y)==pd.Series)

    # Split the train set and validation set
    return train_test_split(X,Y,test_size=0.2,random_state=1)


def scale_data(X_train):
    '''
    This function scales X_train data with standard scaler.
    It returns the scaled data and the scaler.
    :param X_train: training set X          :type: pd.DataFrame
    :rtype: (pd.DataFrame, scaler)
    '''

    # Check the input:
    assert(type(X_train)==pd.DataFrame)

    # Scaling using Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns,
        index=X_train.index)

    return X_train_scaled, scaler


def linear_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler):
    '''
    This function builds a linear regression model on X_train_scaled and
    y_train. Then it calculates the root-mean-squared-error of training
    set and validation set.
    Return the regressor.
    :param X_train_scaled: training set X   :type: pd.DataFrame
    :param y_train: training set y          :type: pd.Series
    :param X_val: validation set X          :type: pd.DataFrame
    :param y_val: validation set y          :type: pd.Series
    :param scaler: standard scaler
    '''

    # Check the input:
    assert(type(X_train_scaled)==pd.DataFrame)
    assert(type(y_train)==pd.Series)
    assert(type(X_val)==pd.DataFrame)
    assert(type(y_val)==pd.Series)

    # Linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled,y_train)

    # Linear regression MSE on training set and validation set
    X_val_scaled = scaler.transform(X_val)
    lin_err = mean_squared_error(y_train,lin_reg.predict(X_train_scaled))
    lin_val_err = mean_squared_error(y_val,lin_reg.predict(X_val_scaled))
    print('Linear Regression Model')
    print('Training RMSE:',f'{np.sqrt(lin_err):.2f}')
    print('Validation RMSE',f'{np.sqrt(lin_val_err):.2f}')
    return lin_reg

def random_forest_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler):
    '''
    This function builds a random forest regression model on X_train_scaled and
    y_train. Then it calculates the root-mean-squared-error of training
    set and validation set.
    Return the regressor.
    :param X_train_scaled: training set X   :type: pd.DataFrame
    :param y_train: training set y          :type: pd.Series
    :param X_val: validation set X          :type: pd.DataFrame
    :param y_val: validation set y          :type: pd.Series
    :param scaler: standard scaler
    '''

    # Check the input:
    assert(type(X_train_scaled)==pd.DataFrame)
    assert(type(y_train)==pd.Series)
    assert(type(X_val)==pd.DataFrame)
    assert(type(y_val)==pd.Series)

    # Simple parameter tuning with grid search
    est_list = [70,75,80,85,90,95,100]
    mxd_list = [7,8,9,10,11,12]
    min_val_err = None
    X_val_scaled = scaler.transform(X_val)

    # Choose the best params for rf model based on validation error
    for i in est_list:
        for j in mxd_list:
            rf_reg = RandomForestRegressor(n_estimators=i,max_depth=j,
                                            random_state=1)
            rf_reg.fit(X_train_scaled,y_train)
            rf_err = mean_squared_error(y_val,rf_reg.predict(X_val_scaled))
            if not min_val_err:
                min_val_err = rf_err
                best_param = (i,j)
                continue
            if rf_err < min_val_err:
                min_val_err = rf_err
                best_param = (i,j)

    # Print the RMSEs
    rf_reg = RandomForestRegressor(n_estimators=best_param[0],
        max_depth=best_param[1],random_state=1)
    rf_reg.fit(X_train_scaled,y_train)
    rf_err = mean_squared_error(y_train,rf_reg.predict(X_train_scaled))
    print('\nRandom Forest Regression Model')
    print('Training RMSE:',f'{np.sqrt(rf_err):.2f}')
    print('Validation MSE: ',f'{np.sqrt(min_val_err):.2f}')
    return rf_reg

def neural_network_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler):
    '''
    This function builds a neural network regression model on X_train_scaled
    and y_train. Then it calculates the root-mean-squared-error of training
    set and validation set.
    Return the model.
    :param X_train_scaled: training set X   :type: pd.DataFrame
    :param y_train: training set y          :type: pd.Series
    :param X_val: validation set X          :type: pd.DataFrame
    :param y_val: validation set y          :type: pd.Series
    :param scaler: standard scaler
    '''

    # Check the input:
    assert(type(X_train_scaled)==pd.DataFrame)
    assert(type(y_train)==pd.Series)
    assert(type(X_val)==pd.DataFrame)
    assert(type(y_val)==pd.Series)

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
    X_val_scaled = scaler.transform(X_val)
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

    print('\nNeural Network Model')
    print('Training RMSE:',f'{np.sqrt(nn_err):.2f}')
    print('Validation RMSE: ',f'{np.sqrt(nn_val_err):.2f}')
    return model

def predict_data(reg,X,scaler,visual):
    '''
    This function first select a random day and then predicts the number of
    rentals for this date using the given model. If visual flag is true,
    visualize the weather of this day.
    :param reg: model regressor
    :param X: DataFrame
    :param visual: visualization flag
    '''

    # Generate a random row
    np.random.seed(seed=42)
    idx = int(np.random.rand()*len(X))
    rand_row = X.iloc[idx,]
    # rand_row[-1] = 1

    # Prediction
    rand_pred = reg.predict(scaler.transform(np.array(rand_row).reshape(1,-1)))

    # Visualize the weather data and print the prediction
    if visual:
        df_newcase = pd.DataFrame({'Max':[67.0,51.2,86.0,29.98,10.0,21.2,26],
                    'Mean':[59.8,48.8,69.0,29,9.6,9.0,26],
                    'Min':[52.2,46.2,51.2,28,7.8,0,26]},
                    index=['TEMP','DWPNT','HUM','SLP','VIS','Wind_S','Gust_S'])
        dif1 = df_newcase['Max']-df_newcase['Mean']
        dif2 = df_newcase['Mean']-df_newcase['Min']
        df_try = pd.DataFrame({'Min':df_newcase['Min'],'Mean':dif2,'Max':dif1},
                    index=['TEMP','DWPNT','HUM','SLP','VIS','Wind_S','Gust_S'])
        print('If the weather of a weekday is')
        df_try.plot.bar(stacked=True,figsize=(10,8),
                    color=['lightsteelblue','cornflowerblue','royalblue'],
                    rot=0, title='Weather Information')

        plt.text(y=55,x=3,
            s='cloud_cover=2.6\nwind_dir_degree=309.8',
            fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Weather',fontsize=16)
        plt.ylabel('Values',fontsize=16)
        plt.legend(prop={'size': 16})
        plt.show()
    print('then the predicted number of daily rentals is',
        (str(int(rand_pred[0]))+'.'))


def full_model(df_trip, df_weather, visual, less_features):
    '''
    This function calls all other functions in this file.
    Visualize the mutual information scores and prediction data if the visual
    flag is true. Print the RMSE of each model.
    It also builds models with only the features that has mi score > 0.1 if
    the less_features flag is true. However, the RMSE does not improve.
    :param df_trip: trip dataset        :type: pd.DataFrame
    :param df_weather: weather dataset  :type: pd.DataFrame
    :param visual: visualization flag   :type: bool
    :param less_features: feature flag  :type: bool
    '''

    # Data preprocessing
    X, Y = prepare_data(df_trip,df_weather)
    if visual:
        mutual_info_scores(X,Y)
    X_train, X_val, y_train, y_val = split_train_valid_sets(X,Y)
    X_train_scaled, scaler = scale_data(X_train)

    # Model fitting
    lin_reg = linear_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
    rf_reg = random_forest_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
    nn_model = neural_network_model_RMSE(X_train_scaled,y_train,
                                            X_val,y_val,scaler)
    # Prediction
    predict_data(rf_reg,X,scaler,visual)

    if not less_features:
        return

    # Repeat with less features
    print('\nRepeat modeling with the following 4 features:')
    selected_features = ['max_temperature_f','mean_temperature_f',
                        'mean_humidity','weekend']
    print(selected_features,'\n')
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_train_scaled, scaler = scale_data(X_train)

    lin_reg = linear_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
    rf_reg = random_forest_model_RMSE(X_train_scaled,y_train,X_val,y_val,scaler)
    nn_model = neural_network_model_RMSE(X_train_scaled,y_train,
                                            X_val,y_val,scaler)


### EXAMPLE ###
'''
plt.style.use('bmh')
df_trip = pd.read_csv('dataset_bike_share/trip.csv')
df_weather = pd.read_csv('dataset_bike_share/weather.csv')
full_model(df_trip,df_weather,visual=True,less_features=False)
'''

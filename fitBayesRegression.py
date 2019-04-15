import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time

def fit_BR(Train_X, Train_Y, Test_X, Test_Y, Predict_Y, Predefined_Split):

    #  1. Fitting

    Fit_BR = GridSearchCV(\
        BayesianRidge(), cv = Predefined_Split, \
        param_grid = {})
    Time0 = time.time()
    Fit_BR.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0

    #  2. Prediction and Error

    Time0 = time.time()
    Predict_Y = Fit_BR.predict(Test_X)

    #obtain MAE and MSE
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0

    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test, Predict_Y)

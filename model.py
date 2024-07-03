from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_X(X, temp_scaler, p_scaler, train=True):
    pressure = X[:, :27]
    temperature = X[:, 27:54]
    time = X[:, 54:]
    if train:
        temp_scaled = temp_scaler.fit_transform(temperature)
        p_scaled = p_scaler.fit_transform(pressure)
    else:
        temp_scaled = temp_scaler.transform(temperature)
        p_scaled = p_scaler.transform(pressure)
    X_scaled = np.hstack((p_scaled, temp_scaled, time))
    return X_scaled

def apply_scaling(X_train, X_test):
    p_scaler = StandardScaler()
    temp_scaler = StandardScaler()
    X_train = scale_X(X_train, temp_scaler, p_scaler, train=True)
    X_test = scale_X(X_test, temp_scaler, p_scaler, train=False)
    return X_train, X_test

def predict_with_model(X_train, Y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions
from preproc import preprocess
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import sys
from model import predict_with_model

def evaluate(infile, freq=15, test_size=0.2):
    clean_data = preprocess(infile, time_features=True)
    _, batch_size = clean_data.shape
    train, test = train_test_split(clean_data, test_size=test_size, random_state=42)
    errors = np.full((batch_size), -1.0)
    std_devs = np.full((batch_size), -1.0)
    for i in range(54):
        print("Evaluating sensor", i)
        X_train = np.delete(train, i, axis=1)
        Y_train = train[:, i].astype(float)
        X_test = np.delete(test, i, axis=1)
        Y_test = test[:, i].astype(float)

        predictions = predict_with_model(X_train, Y_train, X_test)
        error = root_mean_squared_error(Y_test, predictions)
        residuals = Y_test - predictions
        std_dev = np.std(np.abs(residuals))
        std_devs[i] = std_dev
        errors[i] = error
    return errors, std_devs

def main():
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    infile = sys.argv[1]
    errors, std_devs = evaluate(infile)
    pressure_errors = errors[0:27]
    temp_errors = errors[27:54]
    p_std_devs = std_devs[:27]
    t_std_devs = std_devs[27:]
    print("Pressure sensor errors:")
    print(pressure_errors)
    print("Average pressure sensor error:")
    print(np.mean(pressure_errors))
    print("Average pressure residual standard deviation:")
    print(np.mean(p_std_devs))
    print("Temperature sensor errors:")
    print(temp_errors)
    print("Average temperature sensor error:")
    print(np.mean(temp_errors))
    print("Average temperature residual standard deviation:")
    print(np.mean(t_std_devs))

if __name__ == "__main__":
    main()

from preproc import preprocess
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import sys

def evaluate(infile, freq=15, test_size=0.2):
    clean_data = preprocess(infile)
    size, batch_size = clean_data.shape
    train, test = train_test_split(clean_data, test_size=test_size, random_state=42)
    errors = np.ndarray((batch_size))
    for i in range(batch_size):
        X_train = np.delete(train, i, axis=1)
        Y_train = train[:, i].astype(float)
        X_test = np.delete(test, i, axis=1)
        Y_test = test[:, i].astype(float)

        ### INITIALISE, TRAIN THE MODEL AND MAKE PREDICTIONS ###

        # This could involve linear regression, RNN, LSTM, transformer etc.
        # This could be a binary classification task using the ground truth as labels
        # If binary classification, Y should be replaced by the 'ground truth' dataset
        # For now, a basic linear regression will be used, with distance thresholding
        # used to generate binary classification labels

        model = LinearRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        error = root_mean_squared_error(Y_test, predictions)
        errors[i] = error
    return errors

def main():
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    infile = sys.argv[1]
    errors = evaluate(infile)
    pressure_errors = errors[0:27]
    temp_errors = errors[27:]
    print("Pressure sensor errors:")
    print(pressure_errors)
    print("Temperature sensor errors:")
    print(temp_errors)

if __name__ == "__main__":
    main()

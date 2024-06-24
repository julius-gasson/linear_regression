from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preproc import preprocess
import numpy as np
import sys


def get_weights(infile):
    full_data = preprocess(infile)
    for i in range(26):
        print(f"i = {i}")
        X = np.delete(full_data, i, axis=1)
        Y = full_data[:, i].astype(float)  # Ensure labels are float for regression

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        print("Mean Squared Error:", mse)

        # Print model weights (coefficients) and biases (intercept)
        weights = model.coef_
        bias = model.intercept_
        print("Weights:", weights)
        print("Bias:", bias)

def main():
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})
    infile = sys.argv[1]
    get_weights(infile)


if __name__ == '__main__':
    main()
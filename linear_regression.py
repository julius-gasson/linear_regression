from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preproc import preprocess
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


NUM_SENSORS = 27


def get_weights(infile, log=False):
    all_weights = np.empty((NUM_SENSORS, NUM_SENSORS))
    all_biases = np.empty(NUM_SENSORS)
    full_data = preprocess(infile)
    for i in range(NUM_SENSORS):
        X = np.delete(full_data[:, :NUM_SENSORS], i, axis=1)
        Y = full_data[:, i].astype(float)

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        weights = model.coef_
        bias = model.intercept_
        for j in range(NUM_SENSORS):
            if j < i:
                all_weights[i, j] = weights[j]
            elif j > i:
                all_weights[i, j] = weights[j - 1]
            else:
                all_weights[i, j] = np.nan
        all_biases[i] = bias

        if log:
            print(f"i = {i}")
            print("X_TRAIN_AVG:", np.mean(x_train))
            print("Y_TRAIN_AVG:", np.mean(y_train))
            print("Mean Squared Error:", mse)
            print("Weights:", weights)
            print("Bias:", bias)
    np.savetxt("parameters/weights.csv", all_weights, delimiter=",")
    np.savetxt("parameters/biases.csv", all_biases, delimiter=",")
    return all_weights, all_biases

def visualise(all_weights):
    plt.figure(figsize=(14, 10))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad(color="grey")
    sns.heatmap(
        all_weights,
        annot=True,
        cmap=cmap,
        center=0,
        cbar=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        xticklabels=np.arange(1, NUM_SENSORS + 1),
        yticklabels=np.arange(1, NUM_SENSORS + 1),
    )
    plt.title("Heatmap of Linear Regression Weights")
    plt.xlabel("Contributing Sensors")
    plt.ylabel("Affected Sensors")
    plt.show()

def main():
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    infile = sys.argv[1]
    weights, _ = get_weights(infile, log=True)
    visualise(weights)

if __name__ == "__main__":
    main()

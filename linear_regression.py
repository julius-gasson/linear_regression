from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from preproc import preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def get_weights(infile, log=False, pressure=True, temperature=False, num_sensors=27, length=-1):
    all_weights = np.empty((num_sensors, num_sensors))
    all_biases = np.empty(num_sensors)
    errors = np.empty(num_sensors)
    full_data = preprocess(infile, length=length)
    if not pressure:
        full_data = full_data[:, num_sensors:]
    elif not temperature:
        full_data = full_data[:, :num_sensors]
    for i in range(num_sensors):
        X = np.delete(full_data[:, :num_sensors], i, axis=1)
        Y = full_data[:, i].astype(float)

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, predictions)
        weights = model.coef_
        bias = model.intercept_
        for j in range(num_sensors):
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
            print("Root Mean Squared Error:", rmse)
            print("Weights:", weights)
            print("Bias:", bias)
    np.savetxt("parameters/weights.csv", all_weights, delimiter=",")
    np.savetxt("parameters/biases.csv", all_biases, delimiter=",")
    if log:
        print("Average RMSE:", np.mean(errors))
    return all_weights, all_biases


def visualise(all_weights, num_sensors=27):
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
        xticklabels=np.arange(1, num_sensors + 1),
        yticklabels=np.arange(1, num_sensors + 1),
    )
    plt.title("Heatmap of Linear Regression Weights")
    plt.xlabel("Contributing Sensors")
    plt.ylabel("Affected Sensors")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Linear regression for pressure sensors."
    )
    parser.add_argument(
        "infile", type=str, help="The CSV file containing pressure data."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Log the discovered parameters for each ID.",
    )
    parser.add_argument(
        "--length",
        "-l",
        type=int,
        default=-1,
        help="The number of sets of readings to read from the top of the CSV file.",
    )
    parser.add_argument(
        "--visualise",
        "-v",
        action="store_true",
        default=False,
        help="Visualise the weights in a heatmap.",
    )
    parser.add_argument(
        "--pressure",
        "-p",
        action="store_true",
        default=False,
        help="Only evaluate for pressure sensors.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        action="store_true",
        default=False,
        help="Only evaluate for temperature sensors.",
    )
    parser.add_argument(
        "--num-sensors",
        "-n",
        type=int,
        default=27,
        help="Number of sensors to be evaluated.",
    )
    args = parser.parse_args()
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    weights, _ = get_weights(
        args.infile,
        log=args.log,
        pressure=not args.temperature,
        temperature=not args.pressure,
        num_sensors=args.num_sensors,
        length=args.length,
    )
    if args.visualise:
        visualise(weights, num_sensors=args.num_sensors)


if __name__ == "__main__":
    main()

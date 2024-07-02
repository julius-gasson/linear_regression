from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from preproc import preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd


def get_weights(infile, log=False, pressure=True, temperature=False, num_sensors=27, length=-1):
    batch_size = num_sensors * 2
    all_weights = np.empty((batch_size, batch_size))
    all_biases = np.empty(batch_size)
    stud_res = []
    full_data = preprocess(infile, length=length)
    for i in range(batch_size):
        print(f"Evaluation for Sensor {i+1}")
        X = np.delete(full_data, i, axis=1)
        Y = full_data[:, i].astype(float)
        data = pd.DataFrame({"X": X, "Y": Y})
        model = ols("X ~ Y", data=data).fit()
        stud_res = model.outlier_test()
    stud_res = np.array(stud_res)
    np.save("studentised_residuals.npy", stud_res)
    return all_weights, all_biases


def visualise(all_weights, num_sensors=27, pressure=True):
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
    if pressure:
        plt.savefig("pressure_heatmap.png")
    else:
        plt.savefig("temperature_heatmap.png")
    plt.close()

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
        visualise(weights, num_sensors=args.num_sensors, pressure=arg.s)


if __name__ == "__main__":
    main()

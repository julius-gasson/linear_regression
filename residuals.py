from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from preproc import preprocess
import numpy as np
from numpy.linalg import pinv
import argparse

def count_lines(filename):
    count = 0
    with open(filename, 'r') as file:
        for _ in file:
            count += 1
    return count

def add_new_batch(infile, index=0):
    index = index + 1 #  Skip the header
    with open("csv/reversed.csv", 'r') as file:
        lines = file.readlines()
        new_data = lines[index:index+54]
        with open(infile, 'w') as f:
            f.writelines(new_data)

def get_studentised_residuals(x_test, y_test, predictions):
    n, p = x_test.shape
    mse = mean_squared_error(y_test, predictions) * (n / (n-p-1))
    # H = X⋅(X^T⋅X)^(-1)⋅X^T
    hat_matrix = x_test @ pinv(x_test.T @ x_test) @ x_test.T
    leverage = np.diag(hat_matrix)
    residuals = y_test - predictions
    stud_res = residuals / np.sqrt(mse * (1 - leverage))
    return stud_res

def evaluate(infile, models, num_sensors=27, index=0, min_batches=5):
    ######## WAIT FOR NEW BATCH TO BE RECEIVED ########
    while True:
        response = input("Press Enter to add the next batch, or Q to exit")
        if response.lower == "q":
            return
        elif response == "":
            break
    add_new_batch(infile, index=index)

    ######## STOCHASTIC GRADIENT DESCENT ########
    new_data = preprocess(infile, length=2)
    new_data = new_data[3, :] #  Just the value is required
    for i in range(num_sensors * 2):
        X = np.delete(new_data, i)
        Y = new_data[i].astype(float)
        model = models[i]
        model.partial_fit(X, Y)
        if index > min_batches:
            continue

    ########## ANOMALY DETECTION ##########
        predictions = model.predict(X)
        stud_res = get_studentised_residuals(X, Y, predictions)
        print(f"Studentised residuals: {stud_res}")

    ########## RETURN TO START ##########
    evaluate(infile, models, num_sensors=num_sensors, index=index+1)

def main():
    parser = argparse.ArgumentParser(description="Stochastic regression model for pressure sensors.")
    parser.add_argument("infile", type=str, help="The empty CSV file containing pressure and temperature data.")
    parser.add_argument("--num-sensors", "-n", type=int, default=27, help="Number of sensors to be evaluated.")
    args = parser.parse_args()
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    models = [SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1, tol=1e-3, random_state=42, warm_start=True) for _ in range(args.num_sensors * 2)]
    evaluate(args.infile, models, num_sensors=args.num_sensors)

if __name__ == "__main__":
    main()

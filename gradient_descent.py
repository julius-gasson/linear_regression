from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
from runstats import Statistics
from preproc import preprocess
import numpy as np
import argparse

def count_lines(filename):
    count = 0
    with open(filename, 'r') as file:
        for _ in file:
            count += 1
    return count

def add_new_batch(infile, index=0, batch_size=54):
    with open("csv/reversed.csv", 'r') as file:
        lines = file.readlines()
        i = index * batch_size + 1
        new_data = lines[i:i+batch_size]
        with open(infile, 'a') as f:
            f.writelines(new_data)
    # with open(infile) as file:
    #     count = 0
    #     for _ in file:
    #         count += 1 
    #     print(count)

def evaluate(infile, num_sensors=27, index=0, min_batches=50, threshold=3.0, log=False):
    print(("\n"+"="*50).center(50))
    print("SGD Anomaly Detector for Pressure and Temperature Sensors".center(50))
    print("Instructions: press Enter to add the next batch, or Q to exit.")
    print(("\n"+"="*50).center(50))
    batch_size = num_sensors * 2
    models = [SGDRegressor(learning_rate='constant', eta0=0.1e-6, max_iter=1, tol=1e-3, random_state=42, warm_start=True) for _ in range(batch_size)]
    stats = [Statistics() for _ in range(batch_size)]
    ######## WAIT FOR NEW BATCH TO BE RECEIVED ########
    while True:
        response = input()
        if response.lower() == "q":
            print("Exiting monitor...")
            return
        add_new_batch(infile, index=index, batch_size=batch_size)

        ######## STOCHASTIC GRADIENT DESCENT ########
        new_data = preprocess(infile, length=2)
        new_data = new_data[:, 3] #  Just the value is required
        for i in range(num_sensors * 2):
            X = np.delete(new_data, i).reshape(1, -1).astype(float)
            Y = np.array([new_data[i]]).astype(float)
            model = models[i]
            model.partial_fit(X, Y)
        
        ########## EVALUATION ##########
            prediction = model.predict(X)
            residual = abs(Y - prediction)
            if log and i == 0:
                print(f"Iteration {index}: Prediction for sensor 1: {prediction}, actual value: {Y}, residual: {residual}" )
        ######### ANOMALY DETECTION ##########
            if index < min_batches // 2: #  A convergence metric could be used here
                continue
            s = stats[i]
            s.push(residual)
            if log and i == 0:
                if index == min_batches // 2:
                    print(("\n"+"="*50).center(50))
                    print("Data collection started...".center(50))
                    print(("="*50).center(50))
                    continue
                means = [s.mean() for s in stats]
                mean: float = sum(means) / len(means)
                print(f"Mean residual for all sensors: {mean}")
            distance = residual - s.mean()
            if i == 0 and log:
                if index == min_batches:
                    print(("\n"+"="*50).center(50))
                    print("Monitoring started...".center(50))
                    print(("\n"+"="*50).center(50))
            if index > min_batches and distance > threshold * s.stddev():
                print(f"Anomaly detected for sensor {i+1} at index {index} with residual {residual}.")
                print(f"Prediction was: {prediction}, actual value was: {Y}")
                print(f"Distance was: {distance}, max allowed distance was: {threshold * s.stddev()}")
        index += 1

def main():
    parser = argparse.ArgumentParser(description="Stochastic regression model for pressure and temperature sensors.")
    parser.add_argument("infile", type=str, help="The empty CSV file containing pressure and temperature data.")
    parser.add_argument("--num-sensors", "-n", type=int, default=27, help="Number of sensors to be evaluated.")
    parser.add_argument("--threshold", "-t", type=float, default=3.0, help="Threshold for anomaly detection: number of std. dev. of residuals from the mean")
    parser.add_argument("--log", "-l", action="store_true", default=False, help="Log the discovered parameters for ID 1.")
    parser.add_argument("--min-batches", "-b", type=int, default=50, help="Minimum number of batches to collect before anomaly detection.")
    args = parser.parse_args()
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    evaluate(args.infile, num_sensors=args.num_sensors, min_batches=args.min_batches, threshold=args.threshold, log=args.log)

if __name__ == "__main__":
    main()

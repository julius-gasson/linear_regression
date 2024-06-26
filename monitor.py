import pandas as pd
import numpy as np
import argparse
from pycontract.pycontract_core import Monitor, error
from pycontract.extra_csv import CSVSource

class AnomalyDetector(Monitor):
    weights: np.ndarray
    biases: np.ndarray
    max_residual: float
    num_sensors: int
    values: np.ndarray
    last_id: int
    def __init__(self, weights, biases, max_residual, num_sensors):
        super().__init__()
        self.weights = np.nan_to_num(weights.to_numpy()).T
        self.biases = biases.to_numpy().T.flatten()
        self.max_residual = max_residual
        self.num_sensors = num_sensors
        self.values = np.empty(self.num_sensors)
        self.last_id = 0
    def transition(self, event):
        if event['Tipo Grandezza'] != "Pressione a valle":
            return
        event["ID"] = int(event["ID"].replace("PDM", ""))
        match event:
            case {"ID": 1, "Valore": _, 'Tipo Grandezza': "Pressione a valle"}:
                self.values = np.empty_like(self.values)
        match event:
            case {"ID": pdm_id, "Valore": _, 'Tipo Grandezza': "Pressione a valle"} \
                if (pdm_id - self.last_id) % self.num_sensors != 1:
                    self.last_id = pdm_id
                    return error(f"Missing data for at least one sensor. Cannot calculate anomaly statistics.")
        match event:
            case {"ID": pdm_id, "Valore": value, 'Tipo Grandezza': "Pressione a valle"}:
                self.last_id = pdm_id
                self.values[pdm_id-1] = value
        match event:
            case {"ID": self.num_sensors, "Valore": _, 'Tipo Grandezza': "Pressione a valle"}:
                self.last_id = pdm_id
                predictions = np.matmul(self.values, self.weights)
                predictions += self.biases
                residuals = np.abs(predictions - self.values)
                if np.any(residuals > self.max_residual):
                    error_msg = f"Unexpected value! \n\nPredicted: {predictions} \n\nActual values:\n\n{self.values} \
                    \n\nDifferences were: {residuals}\n\nMax allowed difference is: {self.max_residual}"
                    return error(error_msg)

def main():
    parser = argparse.ArgumentParser(description="Pressure anomaly detector.")
    parser.add_argument("infile", type=str, help="The CSV file containing pressure data.")
    parser.add_argument("--max-residual", "-d", type=float, help="The max difference between predictions and actual values.")
    parser.add_argument("--num-sensors", "-n", type=int, default=27, help="The number of sensors in the network.")
    parser.add_argument("--weights", "-w", type=str, default="parameters/weights.csv", help="The CSV file containing the weights.")
    parser.add_argument("--biases", "-b", type=str, default="parameters/biases.csv", help="The CSV file containing the biases.")
    args = parser.parse_args()

    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.5f}".format})
    weights = pd.read_csv(args.weights, header=None)
    biases = pd.read_csv(args.biases, header=None)
    monitor = AnomalyDetector(weights=weights, biases=biases, max_residual=args.max_residual, num_sensors=args.num_sensors)
    with CSVSource(args.infile) as csv_reader:
        for event in csv_reader:
            if event is not None:
                monitor.eval(event)
        monitor.end()

if __name__ == "__main__":
    main()
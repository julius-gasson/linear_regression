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

    def __init__(
        self, weights, biases, max_residual, num_sensors, event_type, training_file=""
    ):
        super().__init__()
        self.weights = np.nan_to_num(weights.to_numpy()).T
        self.biases = biases.to_numpy().T.flatten()
        self.max_residual = max_residual
        self.num_sensors = num_sensors
        self.values = np.empty(self.num_sensors)
        self.last_id = 0
        self.event_type = event_type
        self.training_file = training_file
    def transition(self, event):
        if event["Tipo Grandezza"] != self.event_type:
            return
        event["ID"] = int(event["ID"].replace("PDM", ""))
        match event:
            case {"ID": 1, "Valore": _, "Tipo Grandezza": self.event_type}:
                self.values = np.empty_like(self.values)
        match event:
            case {"ID": pdm_id, "Valore": _, "Tipo Grandezza": self.event_type} if (
                pdm_id - self.last_id
            ) % self.num_sensors != 1:
                self.last_id = pdm_id
                return error(
                    f"Missing data for at least one sensor. Cannot calculate anomaly statistics."
                )
        match event:
            case {"ID": pdm_id, "Valore": value, "Tipo Grandezza": self.event_type}:
                self.last_id = pdm_id
                self.values[pdm_id - 1] = value
        match event:
            case {
                "ID": self.num_sensors,
                "Valore": _,
                "Tipo Grandezza": self.event_type,
            }:
                self.last_id = pdm_id
                predictions = self.values @ self.weights + self.biases
                residuals = np.abs(predictions - self.values)
                if self.training_file != "":
                    with open(self.training_file, "a") as f:
                        for i in range(self.num_sensors):
                            f.write(
                                f"PDM{i+1};{event['Data Campionamento']};{event['ORA Campionamento']};{self.values[i]};{self.event_type}\n"
                            )
                if np.any(residuals > self.max_residual):
                    error_msg = f"Unexpected value! \n\nPredicted: {predictions} \n\nActual values:\n\n{self.values} \
                    \n\nDifferences were: {residuals}\n\nMax allowed difference is: {self.max_residual}"
                    return error(error_msg)


def main():
    parser = argparse.ArgumentParser(description="Pressure anomaly detector.")
    parser.add_argument(
        "infile", type=str, help="The CSV file containing pressure data."
    )
    parser.add_argument(
        "--max-residual",
        "-d",
        type=float,
        default=0.005,
        help="The max difference between predictions and actual values.",
    )
    parser.add_argument(
        "--num-sensors",
        "-n",
        type=int,
        default=27,
        help="The number of sensors in the network.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="parameters/weights.csv",
        help="The CSV file containing the weights.",
    )
    parser.add_argument(
        "--biases",
        "-b",
        type=str,
        default="parameters/biases.csv",
        help="The CSV file containing the biases.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        action="store_true",
        help="Only evaluate for temperature sensors.",
    )
    parser.add_argument(
        "--online-learning",
        "-o",
        type=str,
        help="Add new values to a CSV file used as the training set.",
    )
    args = parser.parse_args()

    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.5f}".format})
    weights = pd.read_csv(args.weights, header=None)
    biases = pd.read_csv(args.biases, header=None)
    event_type = "Temperatura Ambiente" if args.temperature else "Pressione a valle"
    monitor = AnomalyDetector(
        weights=weights,
        biases=biases,
        max_residual=args.max_residual,
        num_sensors=args.num_sensors,
        event_type=event_type,
        training_file=args.online_learning,
    )
    with CSVSource(args.infile) as csv_reader:
        for event in csv_reader:
            if event is not None:
                monitor.eval(event)
        monitor.end()


if __name__ == "__main__":
    main()

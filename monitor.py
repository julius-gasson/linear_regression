import pandas as pd
import numpy as np
import sys
from pycontract.pycontract_core import Monitor, error
from pycontract.pycontract_plantuml import visualize
from pycontract.extra_csv import CSVSource

NUM_SENSORS = 27
MAX_RESIDUAL = 0.01

class AnomalyDetector(Monitor):
    weights: np.ndarray
    biases: np.ndarray
    values: np.ndarray
    def __init__(self, weights, biases):
        super().__init__()
        self.weights = weights.to_numpy().nan_to_num(nan=0.0)
        self.biases = biases.to_numpy()
        self.values = np.empty(NUM_SENSORS)
        self.last_id = 0
    def transition(self, event):
        if event['Tipo Grandezza'] != "Pressione a valle":
            return
        match event:
            case {"ID": "PDM1", "Valore": _, 'Tipo Grandezza': "Pressione a valle"}:
                self.values = np.empty_like(self.values)
            case {"ID": pdm_id, "Valore": _, 'Tipo Grandezza': "Pressione a valle"} \
                if (pdm_id - last_id) % NUM_SENSORS != 1:
                    last_id = pdm_id
                    return error(f"Missing data for at least one sensor. Cannot calculate anomaly statistics.")
            case {"ID": pdm_id, "Valore": value, 'Tipo Grandezza': "Pressione a valle"}:
                self.values[pdm_id] = value
            case {"ID": "PDM27", "Valore": _, 'Tipo Grandezza': "Pressione a valle"}:
                last_id = pdm_id
                predictions = np.matmul(self.weights, self.values) + self.biases
                if np.any(np.abs(predictions - self.values) > MAX_RESIDUAL):
                    return error(f"Unexpected value! {self.values} -> {predictions}")
        return

def main():
    infile = sys.argv[1]
    weights = pd.read_csv("parameters/weights.csv", header=None)
    biases = pd.read_csv("parameters/biases.csv", header=None)
    monitor = AnomalyDetector(weights=weights, biases=biases)
    with CSVSource(infile) as csv_reader:
        for event in csv_reader:
            if event is not None:
                monitor.eval(event)
    
#!/bin/bash

TRAINING_FILE="csv/safe.csv"

echo "Running linear regression on $TRAINING_FILE..."
python linear_regression.py $TRAINING_FILE
echo "Running monitor..."
python monitor.py $1 -d 0.005 -n 27 -w parameters/weights.csv -b parameters/biases.csv
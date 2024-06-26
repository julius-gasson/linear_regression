head -1 csv/reversed.csv > csv/batches.csv
python gradient_descent.py csv/batches.csv -l -t 2.0
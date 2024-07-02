import sys
from sklearn.model_selection import train_test_split
import numpy as np
from preproc import preprocess
from anomalies import add_anomalies
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate(infile, freq=15, test_size=0.2, daily=True):
    ### CREATE FULL DATASET WITH SYNTHETIC ANOMALIES ###

    clean_data = preprocess(infile)
    # ground_truth must be a binary array for each day in the dataset

    ### SEPARATE THE DATASET ###

    size, batch_size = clean_data.shape  #  Make a loop here
    anomaly_responses = np.ndarray((batch_size, int(np.ceil(size * test_size))))
    train, test = train_test_split(clean_data, test_size=test_size, random_state=42)
    test_copy = test.copy()
    # Anomalies could be added to the dataset before X and Y split
    test_anom, ground_truth = add_anomalies(
        dataset=test,
        anomaly_type="step",
        anoms_per_day=0.4,
        daily=daily,
        pressure_step_size=0.005,
        temp_step_size=2,
        debug=False,
    )
    for i in range(batch_size):
        X_train = np.delete(train, i, axis=1)
        Y_train = train[:, i].astype(float)
        X_test = np.delete(test_anom, i, axis=1)
        Y_test = test_anom[:, i].astype(float)

        ### INITIALISE, TRAIN THE MODEL AND MAKE PREDICTIONS ###

        # This could involve linear regression, RNN, LSTM, transformer etc.
        # This could be a binary classification task using the ground truth as labels
        # If binary classification, Y should be replaced by the 'ground truth' dataset
        # For now, a basic linear regression will be used, with distance thresholding
        # used to generate binary classification labels

        model = LinearRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

        ### GET ANOMALY RESPONSE FOR EACH POINT USING DISTANCE METRIC ###

        residuals = Y_test - predictions
        med_res = np.median(residuals)
        # print(f"i = {i}, residuals shape = {residuals.shape}")
        anomaly_responses[i] = np.abs(residuals - med_res)
    ### PREDICT WINDOW LOCATIONS ###
    # For now, the window length will be fixed to 1 day, starting at 00:00:00
    # print("GT shape:", ground_truth.shape)
    # print("AR shape:", anomaly_responses.shape)
    window_size = int(24 * (60 / freq))
    pressure_threshold = 0.002
    temperature_threshold = 1.7
    window_positions = np.array(
        [
            np.arange(
                0, np.size(anomaly_responses, 1) - window_size + 1, window_size
            ).astype(int)
            for _ in range(batch_size)
        ]
    )
    window_predictions = np.zeros_like(ground_truth).T
    # assert (
    #     window_positions.shape == window_predictions.shape
    # ), f"Shape mismatch: WPos shape {window_positions.shape} != WPred shape {window_predictions.shape}"
    for i in range(batch_size):
        for p in range(np.size(window_positions, 1)):
            start_index = window_positions[i, p]
            window = anomaly_responses[i, start_index : start_index + window_size]
            mean = np.mean(window)
            print(p, i, mean)
            if (
                i < batch_size / 2
                and mean > pressure_threshold
                or i >= batch_size / 2
                and mean > temperature_threshold
            ):
                window_predictions[i, p] = 1
    ### EVALUATE PERFORMANCE BASED ON PROXIMITY TO ACTUAL WINDOWS ###
    print("       WP        GT  ")
    window_predictions = window_predictions.astype(int)
    for i in range(batch_size):
        print(i, window_predictions.astype(int)[i], end=" ")
        print(ground_truth.T[i])
    window_predictions = window_predictions.flatten()
    ground_truth = ground_truth.T.flatten()
    precision = precision_score(ground_truth, window_predictions)
    recall = recall_score(ground_truth, window_predictions)
    f1 = f1_score(ground_truth, window_predictions)
    accuracy = accuracy_score(ground_truth, window_predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")


def main():
    infile = sys.argv[1]
    evaluate(infile)


if __name__ == "__main__":
    main()

import sys
from sklearn.model_selection import train_test_split
import numpy as np
from preproc import preprocess
from anomalies import add_anomalies
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def plot_means(
    means: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    label: str,
):
    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(label, color=color)
    ax1.plot(means, color=color)
    ax1.axhline(y=threshold, color="r", linestyle="--", label="Prediction threshold")
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    gt_color = "tab:red"
    pred_color = "orange"
    ax2.set_ylabel("Ground Truth & Predictions", color=gt_color)
    gt_indices = [i for i, x in enumerate(ground_truth) if x == 1]
    print("GT indices:", gt_indices)
    ax2.scatter(
        gt_indices,
        [1 for _ in gt_indices],
        color=gt_color,
        marker="x",
        label="Ground Truth",
        s=50,
    )

    pred_indices = [i for i, x in enumerate(predictions) if x == 1]
    ax2.scatter(
        pred_indices,
        [1 for _ in pred_indices],
        color=pred_color,
        marker="+",
        label="Predictions",
        s=50,
    )

    ax2.tick_params(axis="y", labelcolor=gt_color)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])

    plt.legend(loc="upper right")
    title = label.replace("_", " ")
    plt.title(f"{title} and Ground Truth vs. Predictions")
    fig.tight_layout()
    plt.savefig(f"{label}.png")
    plt.close()


def get_window_predictions(
    anomaly_responses: np.ndarray,
    ground_truth: np.ndarray,
    pressure_threshold: float,
    temperature_threshold: float,
    freq=15,
    batch_size=54,
):
    window_size = int(24 * (60 / freq))
    window_positions = np.array(
        [
            np.arange(
                0, np.size(anomaly_responses, 1) - window_size + 1, window_size
            ).astype(int)
            for _ in range(batch_size)
        ]
    )
    window_predictions = np.zeros_like(ground_truth).T
    print(f"AR size: {anomaly_responses.shape[1]}")
    print(f"window size: {window_size}")
    print(f"size: {np.size(anomaly_responses, 1) - window_size + 1}")
    assert (
        window_positions.shape == window_predictions.shape
    ), f"Shape mismatch: WPos shape {window_positions.shape} != WPred shape {window_predictions.shape}"
    pressure_means = []
    temp_means = []
    for i in range(batch_size):
        for p in range(np.size(window_positions, 1)):
            start_index = window_positions[i, p]
            window = anomaly_responses[i, start_index : start_index + window_size]
            mean = np.mean(window)
            if i < 27:
                pressure_means.append(mean)
            else:
                temp_means.append(mean)
            # print(p, i, mean)
            if (
                i < batch_size / 2
                and mean > pressure_threshold
                or i >= batch_size / 2
                and mean > temperature_threshold
            ):
                window_predictions[i, p] = 1
    return window_predictions, pressure_means, temp_means


def get_stats(window_predictions: np.ndarray, ground_truth: np.ndarray):
    precision = precision_score(ground_truth, window_predictions)
    recall = recall_score(ground_truth, window_predictions)
    f1 = f1_score(ground_truth, window_predictions)
    accuracy = accuracy_score(ground_truth, window_predictions)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")


def evaluate(infile, freq=15, test_size=0.2, daily=True, distance="mad"):
    ### CREATE FULL DATASET WITH SYNTHETIC ANOMALIES ###

    clean_data = preprocess(infile)
    # ground_truth must be a binary array for each day in the dataset

    ### SEPARATE THE DATASET ###

    size, batch_size = clean_data.shape  #  Make a loop here
    anomaly_responses = np.ndarray((batch_size, int(np.ceil(size * test_size))))
    train, test = train_test_split(clean_data, test_size=test_size, random_state=42)
    print("Test shape: ", test.shape)
    # Anomalies could be added to the dataset before X and Y split
    # These are not actual days due to shuffling of dataset
    test_anom, ground_truth, indices = add_anomalies(
        dataset=test,
        anomaly_type="step",
        anoms_per_day=0.4,
        daily=daily,
        pressure_step_size=0.005,
        temp_step_size=2,
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
        if distance == "mad":
            med_res = np.median(residuals)
            anomaly_responses[i] = np.abs(residuals - med_res)
            pressure_threshold = 0.003
            temperature_threshold = 1.8
        elif distance == "zscore":
            pressure_threshold = 10
            temperature_threshold = 5
            train_residuals = Y_train - model.predict(X_train)
            z_score = (residuals - np.mean(train_residuals)) / np.std(train_residuals)
            anomaly_responses[i] = np.abs(z_score)
        else:
            sys.exit(f"Invalid distance metric: {distance}")
    ### PREDICT WINDOW LOCATIONS ###
    # For now, the window length will be fixed to 1 day, starting at 00:00:00
    # print("GT shape:", ground_truth.shape)
    # print("AR shape:", anomaly_responses.shape)
    window_predictions, pressure_means, temp_means = get_window_predictions(
        anomaly_responses=anomaly_responses,
        ground_truth=ground_truth,
        pressure_threshold=pressure_threshold,
        temperature_threshold=temperature_threshold,
        freq=freq,
        batch_size=batch_size,
    )
    gt_flattened = ground_truth.T.flatten()
    gt_pressure = gt_flattened[: len(gt_flattened) // 2]
    gt_temp = gt_flattened[len(gt_flattened) // 2 :]
    wp_flattened = window_predictions.flatten()
    wp_pressure = wp_flattened[: len(wp_flattened) // 2]
    wp_temp = wp_flattened[len(wp_flattened) // 2 :]
    pressure_means = np.array(pressure_means)
    temp_means = np.array(temp_means)
    plot_means(pressure_means, gt_pressure, wp_pressure, pressure_threshold, "pressure_means")
    plot_means(temp_means, gt_temp, wp_temp, temperature_threshold, "temperature_means")
    ### EVALUATE PERFORMANCE BASED ON PROXIMITY TO ACTUAL WINDOWS ###
    # print("       WP        GT  ")
    window_predictions = window_predictions.astype(int)
    # for i in range(batch_size):
    #     print(i, window_predictions.astype(int)[i], end=" ")
    #     print(ground_truth.T[i])
    # print("Anomalies at: ")
    # for i, j in zip(indices[1], indices[0]):
    #     print(f"({i}, {j})")
    # pred_indices = np.where(window_predictions == 1)
    # print("Predicted anomalies at:")
    # for i, j in zip(pred_indices[0], pred_indices[1]):
    #     print(f"({i}, {j})")
    window_predictions = window_predictions.flatten()
    ground_truth = ground_truth.T.flatten()
    get_stats(window_predictions, ground_truth)


def main():
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.8f}".format})
    infile = sys.argv[1]
    evaluate(infile)


if __name__ == "__main__":
    main()

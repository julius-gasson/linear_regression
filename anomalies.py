import sys
from preproc import preprocess
import numpy as np

def choose_targets(
    dataset: np.ndarray,
    anoms_per_day: float,
    window_size: float = 1.0,
    freq: int = 15,
    daily=True,
    all_sensors=False,
):
    size, batch_size = dataset.shape
    readings_per_day = int(24 * (60 / freq))
    num_days = int(size // readings_per_day)
    num_anomalies = anoms_per_day * num_days if daily else anoms_per_day * size
    readings_per_window = int(readings_per_day * window_size)
    num_choices = (
        int(num_anomalies) if daily else int(num_anomalies // readings_per_window)
    )
    choice_range = (
        np.arange(0, size, readings_per_day)
        if daily
        else np.arange(size - readings_per_window + 1)
    )
    chosen_windows = np.random.choice(
        choice_range, size=num_choices, replace=False
    ).astype(int)
    sensor_indices = (
        # We may want to allow the same sensor to malfunction twice, just not at the same time
        np.random.choice(batch_size, size=num_choices, replace=True).astype(int)
        if not all_sensors
        else np.arange(batch_size)
    )
    target_indices = np.array(
        [
            [
                np.arange(window, window + readings_per_window)
                for window in chosen_windows
            ],
            np.tile(sensor_indices, (readings_per_window, 1)).T
            if not all_sensors
            else np.tile(sensor_indices, (readings_per_window * num_choices,)),
        ]
    )
    target_indices = target_indices.reshape(2, -1)
    target_indices = np.unique(target_indices, axis=1)
    ground_truth = np.zeros((size, batch_size)) if not all_sensors else np.zeros(size)
    if daily:
        true_indices = chosen_windows
        if not all_sensors:
            for day, sensor in zip(true_indices, sensor_indices):
                ground_truth[day, sensor] = 1
        else:
            ground_truth[true_indices] = 1
    else:
        if not all_sensors:
            for idx, sensor in zip(target_indices[0], target_indices[1]):
                ground_truth[idx, sensor] = 1
        else:
            ground_truth[target_indices[0]] = 1
    return target_indices, ground_truth.astype(int)


def add_anomalies(
    dataset,
    anoms_per_day=0.2,
    anomaly_type="step",
    window_size=1.0,
    pressure_step_size=0.005,
    temp_step_size=0.5,
    scale_factor=1.05,
    debug=False,
    daily=True,
    all_sensors=False,
):
    target_indices, ground_truth = choose_targets(
        dataset,
        anoms_per_day,
        window_size=window_size,
        daily=daily,
        all_sensors=all_sensors,
    )
    clean_sample = dataset[target_indices[:3], :1].flatten()
    row_indices = target_indices[0, :]
    col_indices = target_indices[1, :]
    num_sensors = dataset.shape[1] / 2

    if anomaly_type == "step":
        for i, row_index in enumerate(row_indices):
            col_index = col_indices[i]
            if col_index < num_sensors:
                dataset[row_index, col_index] += pressure_step_size
            else:
                dataset[row_index, col_index] += temp_step_size
    elif anomaly_type == "ramp":
        dataset[row_indices, col_indices] *= scale_factor
    else:
        sys.exit(f"Invalid anomaly type: {anomaly_type}")
    if debug:
        new_sample = dataset[target_indices[:3], :1].flatten()
        print(f"Sample was {clean_sample},\nis now {new_sample}")
    np.set_printoptions(threshold = np.inf)
    gt_indices = np.arange(0, np.size(ground_truth, 0), 96)
    ground_truth = ground_truth[gt_indices, :]
    indices = np.where(ground_truth == 1)
    print("Anomalies at:")
    for i, j in zip(indices[0], indices[1]):
        print(f"({i}, {j})")
    return dataset, ground_truth


def main():
    infile = sys.argv[1]
    clean_dataset = preprocess(infile)
    cd_copy = clean_dataset.copy()
    add_anomalies(
        clean_dataset, anoms_per_day=0.8, anomaly_type="step", debug=True, daily=True, all_sensors=False
    )

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import argparse

def preprocess(infile, log=False, length=-1):
    # Load data
    df = pd.read_csv(infile, delimiter=';', decimal='.')
    if length == 1 or length == 2:
        df = df.tail(length * 27)
        df["ID"] = df["ID"].str.replace("PDM", "").astype(int)
        return df.to_numpy()
    # Normalize column names and convert IDs
    df["ID"] = df["ID"].str.replace("PDM", "").astype(int)

    # Convert datetime fields
    df['Data Campionamento'] = pd.to_datetime(df['Data Campionamento'], dayfirst=True)
    df['ORA Campionamento'] = pd.to_datetime(df['ORA Campionamento'], format='%H:%M:%S').dt.time
    
    # Create a multi-index for pivoting
    df_pivot = df.pivot_table(index=['Data Campionamento', 'ORA Campionamento', 'ID'],
                              columns='Tipo Grandezza',
                              values='Valore',
                              aggfunc='first').reset_index()
    df_pivot.sort_values(by=['Data Campionamento', 'ORA Campionamento', 'ID'], inplace=True)
    df_pivot.reset_index(drop=True, inplace=True)


    # Prepare data structure to collect entries per ID
    num_ids = df['ID'].max()
    data_by_id = [None] * num_ids

    # Process each group
    for (id_index, group) in df_pivot.groupby('ID'):
        if not group.empty:
            # Ensure each ID has a dedicated list
            if data_by_id[id_index - 1] is None:
                data_by_id[id_index - 1] = []
            data_by_id[id_index - 1].append(group[['Pressione a valle', 'Temperatura Ambiente']].to_numpy())

    # Determine the maximum number of timestamps
    max_length = max(len(data) for sublist in data_by_id for data in sublist if sublist is not None)

    # Initialize final data array with NaNs
    final_data = np.full((max_length, num_ids * 2), np.nan)  # Flattening the last two dimensions

    # Populate the final data array
    for time_step in range(max_length):
        for idx, sublist in enumerate(data_by_id):
            if sublist is not None and len(sublist[0]) > time_step:
                final_data[time_step, idx] = sublist[0][time_step, 0]  # Pressure
                final_data[time_step, idx + num_ids] = sublist[0][time_step, 1]  # Temperature
    avg_pressures = np.nanmean(final_data[:, :num_ids], axis=0)
    if log:
        for i in range(len(avg_pressures)):
            print(f"ID: {i+1}, Average Pressure: {avg_pressures[i]}")
        print(avg_pressures)
    if length > 2:
        df = df.tail(length * 27)
    return final_data

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("infile", type=str, help="The CSV file containing pressure data.")
    args = parser.parse_args()

    arr = preprocess(args.infile, log=True)
    print("Array shape:", arr.shape)

if __name__ == "__main__":
    main()
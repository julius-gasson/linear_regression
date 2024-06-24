import pandas as pd
import numpy as np
import sys

def preprocess(infile):
    # Load data
    df = pd.read_csv(infile, delimiter=';', decimal='.')
    
    # Normalize column names and convert IDs
    df["ID"] = df["ID"].str.replace("PDM", "").astype(int)
    
    # Normalize measurement type names
    df['Tipo Grandezza'] = df['Tipo Grandezza'].replace({
        'Pressione a valle': 'Pressure',
        'Temperatura Ambiente': 'Temperature'
    })

    # Convert datetime fields
    df['Data Campionamento'] = pd.to_datetime(df['Data Campionamento'], dayfirst=True)
    df['ORA Campionamento'] = pd.to_datetime(df['ORA Campionamento'], format='%H:%M:%S').dt.time
    
    # Create a multi-index for pivoting
    df_pivot = df.pivot_table(index=['Data Campionamento', 'ORA Campionamento', 'ID'],
                              columns='Tipo Grandezza',
                              values='Valore',
                              aggfunc='first').reset_index()

    # Sort data and reset index for consistency
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
            data_by_id[id_index - 1].append(group[['Pressure', 'Temperature']].to_numpy())

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

    return final_data

def main():
    infile = sys.argv[1]
    arr = preprocess(infile)
    print("Array shape:", arr.shape)

if __name__ == "__main__":
    main()

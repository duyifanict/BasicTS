import json
import os
import shutil

import numpy as np
from generate_adj_mx import generate_adj_jinan as generate_adj

# Hyperparameters
dataset_name = 'JiNan'
data_file_path = f'datasets/raw_data/{dataset_name}/{dataset_name}.npz'
graph_file_path = f'datasets/raw_data/{dataset_name}/adj_{dataset_name}.pkl'
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
steps_per_day = 288  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'traffic flow'
feature_description = [domain, 'time of day', 'day of week']
regular_settings = {
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'TRAIN_VAL_TEST_RATIO': [0.6, 0.2, 0.2],
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': 0.0
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    data = np.load(data_file_path)['data']
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data

def add_temporal_features(data):
    '''Add time of day and day of week as features to the data.'''
    l, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(l)])
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features

def save_data(data):
    '''Save the preprocessed data to a binary file.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f'Data saved to {file_path}')

def save_graph():
    '''Save the adjacency matrix to the output directory, generating it if necessary.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    if os.path.exists(graph_file_path):
        shutil.copyfile(graph_file_path, output_graph_path)
    else:
        generate_adj()
        shutil.copyfile(graph_file_path, output_graph_path)
    print(f'Adjacency matrix saved to {output_graph_path}')

def save_description(data):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'shape': data.shape,
        'num_time_steps': data.shape[0],
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'feature_description': feature_description,
        'has_graph': graph_file_path is not None,
        'frequency (minutes)': frequency,
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Add temporal features
    data_with_features = add_temporal_features(data)

    # Save processed data
    save_data(data_with_features)

    # Copy or generate and save adjacency matrix
    save_graph()

    # Save dataset description
    save_description(data_with_features)

if __name__ == '__main__':
    main()

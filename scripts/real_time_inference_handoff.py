""""
This script is intended to preprocess a single scenario file and its 
corresponding map file for real-time prediction using advisory congestion model. 

---
Only for inference purposes, not for training!
---
"""

import pandas as pd 
import numpy as np
import os
import ast
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def parse_scen_file(fp):
    scen_df = pd.DataFrame(columns=['unknown_var', 'map', 'map_height', 'map_width',
                           'start_location_x', 'start_location_y',
                           'goal_location_x', 'goal_location_y', 'unknown_heuristic'])
    with open(fp, 'r') as f:
        scen_file = f.readlines()[1:]
        
    for line in scen_file:
        line = line.strip().split('\t')
        scen_df.loc[len(scen_df)] = line
    
    return scen_df

def preprocess_scen(scen_df, scen_id):
    scen_df = scen_df.astype({
          'sif tart_location_x': int,
          'start_location_y': int,
          'goal_location_x': int,
          'goal_location_y': int,
          'unknown_heuristic': float
        })
    scen_df['start_location'] = scen_df.apply(lambda row: (row['start_location_x'], row['start_location_y']), axis=1)
    scen_df['goal_location'] = scen_df.apply(lambda row: (row['goal_location_x'], row['goal_location_y']), axis=1)

    scen_df['unique_id'] = [scen_id + '_agent_' + str(i+1)  for i in range(len(scen_df))]
    scen_df.drop(columns=['start_location_x',
       'start_location_y', 'goal_location_x', 'goal_location_y'], inplace=True)
    return scen_df

def parse_map(map_file):
    # Read the map file
    with open(map_file, 'r') as f:
        map_data = f.readlines()
    map_data = map_data[4:]  # Skip the map header

    width = len(map_data[0].strip())  # Infer width from the map
    height = len(map_data)

    # Initialize the grid representation
    grid = np.zeros((height, width))
    for row_idx in range(height):
        for col_idx in range(width):
            grid[row_idx, col_idx] = 0 if map_data[row_idx][col_idx] == '.' else 1


if __name__ == '__main__':
    scen_file = '../data/raw/random-32-32-20-even-1.scen' #This scene file can be given as input from bash
    # Bash command to use new scene file:
    # if len(sys.argv) > 1:
    #     scen_file = sys.argv[1]

    #This becomes static input for this map file -- Random 32x32-20 map
    random_32_32_df = pd.read_csv("../data/raw/random-32-32-20.csv")
    random_32_32_df['unique_id'] = random_32_32_df['unique_id'].astype(str)
    grid, wdith, height = parse_map("../data/raw/random-32-32-20/random-32-32-20.map") 
    
    scen_df = parse_scen_file(scen_file)
    scen_df = preprocess_scen(scen_df, os.path.basename(scen_file).split('.')[0])
    scen_df['unique_id'] = scen_df['unique_id'].astype(str)

    # Only keep rows from random_32_32_df that match scen_df['unique_id']
    merged_df = pd.merge(
        scen_df,
        random_32_32_df,
        on='unique_id',
        how='left'
    )

    ref_loc_df = merged_df[['scen_type', 'type_id', 'agents', 'start_location', 'goal_location']]
    ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

    # Model input data
    merged_df['makespan'] = [len(max(i.split('\n'), key=len)) for i in merged_df['solution_plan']]

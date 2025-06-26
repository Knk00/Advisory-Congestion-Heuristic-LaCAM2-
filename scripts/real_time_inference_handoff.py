""""
This script is intended to preprocess a single scenario file and its 
corresponding map file for real-time prediction using advisory congestion model. 

---
Only for inference purposes, not for training!
---
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
import ast
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt
import torch     



if __name__ == '__main__':
    # Bash command to use new scene file:
    # if len(sys.argv) > 1:
    #     scen_file = sys.argv[1]

    # Bash command to select a specific number of agents:
    # if len(sys.argv) > 1:
    #     agents = sys.argv[1]
    
    agents = 50 # This can be set to a specific agent number for testing, e.g., 1, 2, etc.
    
    scen_file = './data/raw/random-32-32-20/random-32-32-20-even-25.scen' #This scene file can be given as input from bash

    #This becomes static input for this map file -- Random 32x32-20 map
    random_32_32_df = pd.read_csv("./data/raw/random-32-32-20/random-32-32-20.csv")
    random_32_32_df['unique_id'] = random_32_32_df['scen_type'] + '_' + random_32_32_df['type_id'].astype(str) + '_agent_' + random_32_32_df['agents'].astype(str)
    random_32_32_df['unique_id'] = random_32_32_df['unique_id'].astype(str)
    
    grid, wdith, height = parse_map("./data/raw/random-32-32-20/random-32-32-20.map") 
    
    # Parse the scenario file
    # This function reads the scenario file and returns a DataFrame with relevant columns.
    scen_df = parse_scen_file(scen_file)
    scen_id = scen_file.split('random-32-32-20-')[-1].split('.scen')[0]
    scen_id = scen_id.replace('-', '_')
    scen_df = preprocess_scen(scen_df, scen_id)
    scen_df = scen_df.iloc[:agents] # Limit to the specified number of agents

    # Create a reference DataFrame for start and goal locations
    # This DataFrame will be used to match the agents in the scenario file with the agents
    ref_loc_df = scen_df[['unique_id', 'start_location', 'goal_location']].copy()
    ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str) 
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)

    ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)


    #Filter the random_32_32_df to only include unique_ids present in the scenario file
    # This ensures that we only work with the agents defined in the scenario file.
    random_32_32_df = random_32_32_df[random_32_32_df['unique_id'].isin(ref_loc_df['unique_id'])].iloc[-1]

    # Model input data
    makespan = [len(max(i.split('\n'), key=len)) for i in random_32_32_df['solution_plan']]

    # Transpose the grid for aligning with standard positional indexing
    # This is necessary because the grid is represented in a way that matches the scenario file.
    grid_1 = grid.T.copy()
    grid_2 = grid.T.copy()
    simulation_grid = grid.T.copy()

    # Extract solution plans from the DataFrame
    solution_plan = random_32_32_df['solution_plan']

    # def compute_frequency_heatmap(solution_plan, makespan, ref_loc_df):
    start_locations = []
    goal_locations = []

    query = ref_loc_df.iloc[agents - 1].copy()
    
    start_loc = tuple(query['start_location'])
    start_locations.append(start_loc)

    goal_loc = tuple(query['goal_location'])
    goal_locations.append(goal_loc) 

    for timestep in range(len(makespan)):
        curr_pos = calculate_curr_position(timestep, solution_plan[agents - 1], start_loc)

        # Ensure position is within bounds
        if 0 <= curr_pos[0] < simulation_grid.shape[0] and 0 <= curr_pos[1] < simulation_grid.shape[1]:
            if simulation_grid[curr_pos] >= 0:
                simulation_grid[curr_pos] += 1
            else:
                skip_counter += 1

        # Stop if agent reaches goal exactly at the end
        if timestep == len(solution_plan[agent - 1]):
            if tuple(query['goal_location'].values[0]) == curr_pos:
                break



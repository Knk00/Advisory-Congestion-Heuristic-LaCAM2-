""""
This file contains utility functions for processing advisory congestion input data.
"""

import pandas as pd
import numpy as np


def calculate_goal_location(df, plan_col='solution_plan', start_col='start_location'):
    calculated_goal_location = []
    for i in range(len(df)):
        row = df.iloc[i]
        x, y = row['start_location']
        plan = row['solution_plan']
        uid = row['unique_id']
        
        if plan == None:
            calculated_goal_location.append(None)
            continue
        x += plan.count('r') - plan.count('l')
        y += plan.count('u') - plan.count('d')
        
        calculated_goal_location.append((x, y))
        
    df['calculated_goal_location'] = calculated_goal_location
    
    return df


def calculate_curr_position(timestep, agent_solution_plan, start_loc):
    plan = agent_solution_plan[:timestep]
    x, y = start_loc
    x += plan.count('r') - plan.count('l')
    y += plan.count('u') - plan.count('d')
    return (x, y)

def min_max_scaling(grid, mask_value=-10):
    # Create a mask to identify non-obstacle values
    mask = (grid != mask_value)
    
    # Extract only valid values
    grid_mask = grid[mask]
    
    # Compute min and max from non-obstacle values
    grid_min = grid_mask.min()
    grid_max = grid_mask.max()
    
    # Create a float copy of the grid (so obstacle values remain unchanged)
    scaled_grid = grid.astype(float)

    # Apply Min-Max Scaling only on non-obstacle values
    scaled_grid[mask] = (grid_mask - grid_min) / (grid_max - grid_min)
    
    return scaled_grid


def create_direction_field(start, goal, map_shape):
    """
    Creates a direction field showing the movement intention
    from start to goal across the entire map.
    
    Args:
        start: tuple (x, y) of start position
        goal: tuple (x, y) of goal position
        map_shape: tuple (height, width) of the map
    
    Returns:
        Tuple of (field_x, field_y) 2D numpy arrays with direction components
    """
    height, width = map_shape
    
    # Calculate the direction vector from start to goal
    dir_vector = np.array([goal[0] - start[0], goal[1] - start[1]])
    
    # Normalize if not zero
    norm = np.linalg.norm(dir_vector)
    if norm > 0:
        dir_vector = dir_vector / norm
    
    # Create coordinate matrices for vectorized operations
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distances from each cell to the start position (vectorized)
    distances = np.sqrt((x_coords - start[0])**2 + (y_coords - start[1])**2)
    
    # Add a small constant to avoid division by zero
    distances = np.maximum(distances, 0.1)
    
    # Calculate influence (inversely proportional to distance squared)
    influence = 1.0 / (distances**2)
    
    # Compute field components using broadcasting
    field_x = dir_vector[0] * influence
    field_y = dir_vector[1] * influence  # Fixed: was using dir_vector[0] before
    
    # Normalize the fields
    x_max = np.max(np.abs(field_x))
    if x_max > 0:
        field_x = field_x / x_max
        
    y_max = np.max(np.abs(field_y))
    if y_max > 0:
        field_y = field_y / y_max
        
    return field_x, field_y

def create_aggregate_direction_fields(starts, goals, map_shape):
    """
    Creates aggregate direction fields for multiple agents.
    
    Args:
        starts: List of (x, y) tuples for start positions
        goals: List of (x, y) tuples for goal positions
        map_shape: tuple (height, width) of the map
        
    Returns:
        Tuple of (x_field, y_field) containing the aggregate directional information
    """
    height, width = map_shape
    x_field = np.zeros((height, width))
    y_field = np.zeros((height, width))
    
    # Process all agents in one go if there are many
    if len(starts) > 10:
        # Pre-allocate arrays for all agent fields
        all_x_fields = np.zeros((len(starts), height, width))
        all_y_fields = np.zeros((len(starts), height, width))
        
        # Calculate individual fields
        for i, (start, goal) in enumerate(zip(starts, goals)):
            all_x_fields[i], all_y_fields[i] = create_direction_field(start, goal, map_shape)
            
        # Sum across all agents
        x_field = np.sum(all_x_fields, axis=0)
        y_field = np.sum(all_y_fields, axis=0)
    else:
        # Original approach for smaller numbers of agents
        for start, goal in zip(starts, goals):
            field_x, field_y = create_direction_field(start, goal, map_shape)
            x_field += field_x
            y_field += field_y
    
    # Normalize the combined fields
    x_max = np.max(np.abs(x_field))
    if x_max > 0:
        x_field = x_field / x_max
        
    y_max = np.max(np.abs(y_field))
    if y_max > 0:
        y_field = y_field / y_max
        
    return x_field, y_field
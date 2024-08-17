"""
This file contains the custom filters used to improve the performanz of the CNN.
They are created as numpy-arrays and laters transformed to tensors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def prewitt_filter(direction="both"):
    prewitt_x = np.array(([-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]), dtype=np.float32)

    prewitt_y = np.array(([1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]), dtype=np.float32)
    
    if direction == "x":
        return prewitt_x
    elif direction == 'y':
        return prewitt_y
    elif direction == 'both':
        return prewitt_x, prewitt_y
    else:
        raise ValueError("Invalid direction. Use 'x', 'y', or 'both'.")

def sobel_filter(direction="both"):
    sobel_x = np.array(([-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]), dtype=np.float32)

    sobel_y = np.array(([ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]), dtype=np.float32)
    
    if direction == "x":
        return sobel_x
    elif direction == 'y':
        return sobel_y
    elif direction == 'both':
        return sobel_x, sobel_y
    else:
        raise ValueError("Invalid direction. Use 'x', 'y', or 'both'.")

def sharr_filter(direction="both"):
    sharr_x = np.array(([-3,  0,  3],
                        [-10,  0,  10],
                        [-3,  0,  3]), dtype=np.float32)

    sharr_y = np.array(([ 3,  10,  3],
                        [ 0,  0,  0],
                        [-3, -10, -3]), dtype=np.float32)
    
    if direction == "x":
        return sharr_x
    elif direction == 'y':
        return sharr_y
    elif direction == 'both':
        return sharr_x, sharr_y
    else:
        raise ValueError("Invalid direction. Use 'x', 'y', or 'both'.")

def kirsch_filter(directions="all"):
    filters = {
        'N': np.array([[-3, -3,  5],
                       [-3,  0,  5],
                       [-3, -3,  5]], dtype=np.float32),

        'NE': np.array([[-3, -3, -3],
                        [-3,  0,  5],
                        [-3,  5,  5]], dtype=np.float32),

        'E': np.array([[-3, -3, -3],
                       [-3,  0, -3],
                       [ 5,  5,  5]], dtype=np.float32),

        'SE': np.array([[-3, -3, -3],
                        [ 5,  0, -3],
                        [ 5,  5, -3]], dtype=np.float32),

        'S': np.array([[ 5, -3, -3],
                       [ 5,  0, -3],
                       [ 5, -3, -3]], dtype=np.float32),

        'SW': np.array([[ 5,  5, -3],
                        [ 5,  0, -3],
                        [-3, -3, -3]], dtype=np.float32),

        'W': np.array([[ 5,  5,  5],
                       [-3,  0, -3],
                       [-3, -3, -3]], dtype=np.float32),

        'NW': np.array([[-3,  5,  5],
                        [-3,  0,  5],
                        [-3, -3, -3]], dtype=np.float32)
    }

    # return all filters
    if directions == "all":
        return filters
    
    # return one chosen filter
    elif isinstance(directions, str):
        if directions in filters:
            return filters[directions]
        else:
            raise ValueError("Invalid direction. Use one of 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', or 'all'.")
    
    # return several chosen filter
    elif isinstance(directions, list):
        selected_filters = {dir: filters[dir] for dir in directions if dir in filters}
        if len(selected_filters) != len(directions):
            raise ValueError("One or more invalid directions provided.")
        return selected_filters
    else:
        raise ValueError("Invalid type for directions. Use 'all', a single direction, or a list of directions.")
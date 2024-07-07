""" 
This file contains the custom filters. Filters are created as numpy arrays and later transformed to tensors.
"""

import numpy as np    
from synthetic_data_generation import generate_pulsars, generate_data


def custom_filters_1():
    """
    Binary custom filters that mimic form of pulsar.
    """

    np_filter_0 = np.array([[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]])

    np_filter_1 = np.array([[0., 0., 0.],
                            [1., 1., 1.],
                            [0., 0., 0.]])

    np_filter_2 = np.array([[0., 1., 0.],
                            [0., 1., 0.],
                            [0., 1., 0.]])

    np_filter_3 = np.array([[1., 0., 0.],
                            [0., 1., 1.],
                            [0., 0., 0.]])

    np_filter_4 = np.array([[1., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.]])

    np_filter_5 = np.array([[1., 1., 0.],
                            [0., 0., 1.],
                            [0., 0., 0.]])

    filters = np.array([[np_filter_0], [np_filter_1], [np_filter_2], [np_filter_3], [np_filter_4], [np_filter_5]], dtype=np.float32)

    return filters

def custom_filters_2(size, num, noise):
    """ 
    Custom filters created by pulsar generator to get pulsar form with noise.
    """

    y_values_list = generate_pulsars(dim=size, num_img=num)

    filters, _ = generate_data(size, y_values_list, noise)

    return filters/255
""" 
This file contains the custom filters. Filters are created as numpy arrays and later transformed to tensors.
"""

import numpy as np    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# my files
from src.cnn_models import SparseAutoencoder, PulsarDetectionNet
from src.helper_functions import replace_value_with_value, fit
from src.lbp_sae_filter import typical_image_selector, get_lbp_images
from src.lbp_sae_utils import train_autoencoder
from src.synthetic_data_generation import generate_pulsars, generate_data, generate_train_test_valid_data

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

def custom_filters_2(noise):
    filters = custom_filters_1()
    modified_filters = replace_value_with_value(filters, 0, noise/100 - 0.5)
    
    return modified_filters

def custom_filters_3(noise):
    filters = custom_filters_1()
    modified_filters = replace_value_with_value(filters, 1, 2)
    modified_filters = replace_value_with_value(modified_filters, 0, noise/100)
    
    return modified_filters

def custom_filters_4(size, num, noise):
    """ 
    Custom filters created by pulsar generator to get pulsar form with noise.
    """
    num = int(num/2)

    y_values_list = generate_pulsars(dim=size, num_img=num)

    filters, _ = generate_data(size, y_values_list, noise)

    return np.array(filters/255, dtype=np.float32)

def custom_filters_5(dataset_name):
    dataset = np.load(f"data/{dataset_name}.npz")
    data = dataset[dataset.files[0]]
    labels = dataset["labels"]
    train_loader, test_loader = generate_train_test_valid_data(data, labels, bs=32)

    model = PulsarDetectionNet(32)
    opt = optim.Adam(model.parameters(), lr=0.01)
    fit(model, 10, opt, nn.CrossEntropyLoss(), train_loader, test_loader, learn_plot=False)
    filters = model.conv1.weight.data.cpu()

    return filters

# not working
def custom_filters_6(data, labels):
    typical_images = typical_image_selector(data, labels)
    lbp_images = get_lbp_images(typical_images)

    lbp_dataset = np.array([img for images in lbp_images.values() for img in images], dtype=np.float32)
    lbp_dataset = lbp_dataset/255
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data), torch.from_numpy(data)), batch_size=1, shuffle=True)

    model = SparseAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_autoencoder(model, optimizer, train_loader, 20)
    encoder_filters = model.encoder[0].weight.data.clone()

    return encoder_filters
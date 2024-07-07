"""
This file contains the functions to generate my synthetic data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from .helper_functions import exponential_decay, my_round, number_iterations, manhatten_distance, get_odd_number

def generate_pulsar_variation(dim, num_img, ranges):
    """
    Calculates the pulsar form with given distribution of total number and the parameter ranges for coincidence for the exponential decay function.
    """

    y_values_list = []
    x_values = np.arange(dim)

    for _ in range(num_img):
        a = np.random.uniform(*ranges[0])
        b = np.random.uniform(*ranges[1])
        c = np.random.uniform(*ranges[2])
        if ranges[3] == "down":
            d = np.random.uniform(-(a - (dim-1)), -(a - (dim*0.2)))
        else:
            d = np.random.uniform(*ranges[3])
        
        y_values = exponential_decay(x_values, a, b, c, d)
        
        y_clipped = np.clip(y_values, -1, dim)
        y_rounded = list(map(my_round, y_clipped))
        y_values_list.append(y_rounded)

    return y_values_list

def generate_pulsars(dim, num_img, distr_percentage=[1/5, 2/5, 2/5], test_seed=None):
    """
    Creates pulsars with different variations.
    """

    ranges = {
        "horizontal": [((0.2 * dim), dim/2), (1/dim, 1.8/dim), (0,0), (0, dim/3)],
        "vertical": [(dim, dim * (4/3)), (3/dim, 5/dim), (0, -dim/1.5), "down"],
        "curves": [((dim-1)/2, (dim-1)), (2/dim, 3.5/dim), (0, -(dim/3)), (0,0)]
    }

    if sum(distr_percentage) != 1:
        raise ValueError("Sum of distr_percentage has to be 1")

    if test_seed:
        np.random.seed(test_seed)

    distribution = number_iterations(distr_percentage, num_img)
    y_values_list = []

    for i, (_, item) in enumerate(ranges.items()):
        pulse_signal = generate_pulsar_variation(dim=dim, num_img=distribution[i], ranges=item) 
        y_values_list.extend(pulse_signal)
    
    return y_values_list

def generate_pulsar_img(dim, y_values, background):
    """
    Takes noisy background and lays the pulsar form on top of it. Fills missing pixels with number of pixels according to manhatten distance.
    """

    final_pulsar_img = background.copy()

    for i in range(dim):
        if y_values[i] >= 0 and y_values[i] <= (dim-1):
            final_pulsar_img[0, (dim-1) - y_values[i], i] = np.random.randint(245, 256)
            if i < (dim -1):
                curr = [(dim-1) - y_values[i], i]
                next = [(dim-1) - y_values[i+1], i+1]
                dist = manhatten_distance(curr, next)
                if dist > 2:
                    filler_pixel = dist - 2 
                    for p in range(1, filler_pixel + 1):
                        final_pulsar_img[0, (dim-1) - y_values[i] + p, i] = np.random.randint(245, 256)

    return final_pulsar_img

def generate_data(dim, y_values_list, noise, test_seed=None, plot=False):
    """
    Creates pulsar and non-pulsar images.
    """

    if test_seed:
        np.random.seed(test_seed)
        
    data = []
    labels = []
    x_values = np.arange(dim)
    num_pulsars = len(y_values_list)

    for i in range(num_pulsars):
        grayscale_img = np.random.randint(0, (0.5*256) + noise * (256 - (0.5*256)), size=(1, dim, dim), dtype=np.uint8)
        pulsar_img = generate_pulsar_img(dim=dim, y_values=y_values_list[i], background=grayscale_img)

        # non pulsar
        data.append(grayscale_img)
        labels.append(1)

        # pulsar
        data.append(pulsar_img)
        labels.append(0)
    
    data, labels = np.array(data), np.array(labels)
    
    if plot:
        fig = plt.figure(figsize=(15,10))

        line_sub = fig.add_subplot(2, 1, 1, aspect='equal')
        pulsar_indices = np.random.choice(np.where(labels == 0)[0], size=5, replace=False)
        num_cols = 5
        for i, idx in enumerate(pulsar_indices):
            pulsar_idx = get_odd_number(idx, len(data)) - 1
            line_sub.plot(x_values, y_values_list[pulsar_idx], label=f"Line {i+1}")
            gray_sub = fig.add_subplot(2, num_cols, num_cols + i + 1)
            gray_sub.imshow(data[idx][0], cmap='gray', vmin=0, vmax=255)
            gray_sub.axis('off')
            gray_sub.set_title(f'Line {i + 1}', fontsize=10)

        line_sub.legend()
        line_sub.set_title('Lineplot')
        line_sub.set_ylim(0, dim-1)

        plt.tight_layout()
        plt.show()

    return  data, labels

def generate_train_test_valid_data(data, labels, bs=32):
    """
    Splits data into train, validation and test data and turns it into DataLoaders for efficient processing.
    """

    data = np.array(data)/255 # divided by 255 to get pixel values between 0 and 1
    labels = np.eye(len(np.unique(labels)))[labels] # 1-hot encoded vector

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    # split train into train and valid data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Converting training and testing set to use DataLoaders for easily iterating over batches
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs)
    valid_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)

    return train_loader, test_loader, valid_loader

def save_dataset(dim, num_img, test_seed=None, dir="my_synthesized_data"):
    """
    Creates datasets with different noise levels and saves it.
    """

    y_values_list = generate_pulsars(dim=dim, num_img=num_img, test_seed=test_seed)

    noise_00, labels = generate_data(dim, y_values_list, 0.0)
    noise_70, _ = generate_data(dim, y_values_list, 0.7)
    noise_80, _ = generate_data(dim, y_values_list, 0.8)
    noise_90, _ = generate_data(dim, y_values_list, 0.9)
    noise_100, _ = generate_data(dim, y_values_list, 1.0)

    parent_dir = os.path.dirname(os.path.dirname(__file__))  # Get the parent directory of 'src'

    data_dir = os.path.join(parent_dir, 'data')  # Create path to 'data' directory in parent_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, dir + '.npz')

    np.savez(file_path, n_00=noise_00, n_70=noise_70,
                     n_80=noise_80, n_90=noise_90, n_100=noise_100, labels=labels)
    
    print(f"\nDataset saved in '{file_path}'")

def main(dim_list, img_list):
    for i in range(len(dim_list)):
        dim = dim_list[i]
        file_path = f"{dim}x{dim}_synthesized_data"

        parent_dir = os.path.dirname(os.path.dirname(__file__))  # Get the parent directory of 'src'
        data_dir = os.path.join(parent_dir, 'data')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
        if os.path.isfile(f"data/{file_path}.npz"):
            print(f"Dataset for {dim}x{dim} already exists. Do you want to recreate it? [Y/N]")
            new = input()
            if new.lower() == "y":
                print(f"Creating {dim}x{dim} dataset")
                num_img = img_list[i]
                save_dataset(dim=dim, num_img=num_img, test_seed=None, dir=file_path)
            else:
                print(f"Not recreating {dim}x{dim} dataset.")
            print()
        
        else:
            num_img = img_list[i]
            save_dataset(dim=dim, num_img=num_img, test_seed=None, dir=file_path)
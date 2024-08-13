"""
This file tests networks with different hyperparameters, to get the best ones.
"""

import itertools
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim

from models import TuneNet
from utils import fit, decorate_text
from synthetic_data_generation import generate_train_test_valid_data

def get_all_configurations(config):
    keys = config.keys()
    values = config.values()

    # Calculate the Cartesian product of all values 
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries for each combination
    configurations = [dict(zip(keys, combo)) for combo in combinations]

    return configurations


def parameter_tuning(data, labels, config, epochs=10, learn_plot=False):
    dim = len(data[0][0])

    configurations = get_all_configurations(config)

    best_config = None 
    best_test_acc = 0

    count = 1
    total = len(configurations)
    start_time = time.time()
    for configuration in configurations:
        c1 = configuration["c1"]
        c2 = configuration["c2"]
        c3 = configuration.get("c3", None)
        fc = configuration["fc"]
        f_size = configuration["f_size"]
        lr = configuration["lr"]
        bs = configuration["bs"]

        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs)

        decorate_text(f"{count}/{total} | c1: {c1}, c2: {c2}"+(f", c3: {c3}" if c3 is not None else "")+f", fc: {fc}, f_size: {f_size}, lr: {lr}, bs: {bs}") 
        model = TuneNet(dim, c1, c2, c3, fc, f_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fun = nn.CrossEntropyLoss()
        _, _, _, test_acc = fit(model, epochs, optimizer, loss_fun, train_loader, test_loader, valid_loader, learn_plot=learn_plot)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_config = configuration
        
        count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    
    print(f"\nBest configuration with Test Accuracy of {best_test_acc}: (Runtime: {duration_str})")
    for key, value in best_config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    config = {
        "c1": [16, 32],
        "c2": [32, 64],
        "fc": [64, 128],
        "f_size": [3, 5],
        "lr": [0.001, 0.0001, 0.00001],
        "bs": [8, 16, 32]
    }

    
    dataset = np.load('data/128x128_synthesized_data.npz')
    labels = dataset["labels"]
    
    i = 1
    print()
    print(f"testing on {dataset.files[i]}")
    print()
    for j in range(3):
        data = dataset[dataset.files[i]]
        parameter_tuning(data, labels, config, learn_plot=False)
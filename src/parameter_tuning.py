"""
This file tests networks with different hyperparameters, to get the best ones.
"""

import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import time

from src.cnn_models import TuneNet
from src.synthetic_data_generation import generate_train_test_valid_data
from src.helper_functions import fit, decorate_text, print_data

def get_all_configurations(config):
    keys = config.keys()
    values = config.values()

    # Berechne das kartesische Produkt aller Werte
    combinations = list(itertools.product(*values))

    # Erzeuge eine Liste von Dictionarys für jede Kombination
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
    
    print(f"\nBest configuration with Test Accuracy of {best_test_acc}:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"{duration_str}")

if __name__ == "__main__":
    config_32x32 = {
        "c1": [16, 32],
        "c2": [32, 64],
        "fc": [64, 128],
        "f_size": [3],
        "lr": [0.001],
        "bs": [32, 64]
    }
    
    config_128x128 = {
        "c1": [32, 64],
        "c2": [64, 128],
        "c3": [64, 128],
        "fc": [128, 256],
        "f_size": [3,5],
        "lr": [0.0001],
        "bs": [32, 64]
    }

    dataset = np.load('data/32x32_synthesized_data.npz')
    
    data = dataset[dataset.files[4]]
    labels = dataset["labels"]

    parameter_tuning(data, labels, config_128x128, learn_plot=True)
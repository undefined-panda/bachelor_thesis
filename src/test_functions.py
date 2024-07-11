"""
This file contains the tests when testing the custom filters on the data.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator

# my files
from src.helper_functions import decorate_text, fit
from src.synthetic_data_generation import generate_train_test_valid_data
from src.custom_filters import *
from src.cnn_models import PulsarDetectionCNN_1

def test_1(dataset, epochs, lr, bs, learn_plot, graph_plot):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[files[0]][0][0])

    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []
    
    decorate_text(f"Test 1: default filters vs fixed binary custom filters ({dim}x{dim})")

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs)

        # default filters
        default_model = PulsarDetectionCNN_1(dim)
        default_model_name = type(default_model).__name__
        decorate_text(f"{default_model_name} with default filters", f"{noise}% noise") if learn_plot else None
        default_optimizer = optim.Adam(default_model.parameters(), lr=lr)
        _, _, default_valid_acc, default_test_acc = fit(default_model, epochs, default_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        default_valid_acc_history.append(default_valid_acc)
        default_test_acc_history.append(default_test_acc)

        # custom filters
        filters = custom_filters_1()
        custom_filters = torch.from_numpy(filters)
        custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
        custom_model = PulsarDetectionCNN_1(dim, custom_filters, custom_bias)
        custom_model_name = type(custom_model).__name__
        decorate_text(f"{custom_model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=lr)
        _, _, custom_valid_acc, custom_test_acc = fit(custom_model, epochs, custom_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        custom_valid_acc_history.append(custom_valid_acc)
        custom_test_acc_history.append(custom_test_acc)
    
    if graph_plot:
        x_range = range(1, epochs + 1)
        for j in range(len(noise_levels)):
            plt.plot(x_range, default_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (default filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        for j in range(len(noise_levels)):
            plt.plot(x_range, custom_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (custom filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        x = np.arange(len(noise_levels))  # Die x-Koordinaten der Balken
        width = 0.35  # Breite der Balken

        plt.bar(x - width/2, default_test_acc_history, width, label='default filters', color='skyblue')
        plt.bar(x + width/2, custom_test_acc_history, width, label='custom filters', color='orange')

        plt.xlabel('Noise Level (in %)')  # Beschriftung der x-Achse
        plt.ylabel('Accuracy')       # Beschriftung der y-Achse
        plt.title('Test Accuracy')  # Titel des Diagramms
        plt.xticks(x, noise_levels)     # Beschriftung der x-Achsen-Ticks

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
    
def test_2(dataset, epochs, lr, bs, learn_plot, graph_plot):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[files[0]][0][0])

    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []

    decorate_text(f"Testing: default filters vs changing binary custom filters ({dim}x{dim})")
    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs)

        # default filters
        default_model = PulsarDetectionCNN_1(dim)
        default_model_name = type(default_model).__name__
        decorate_text(f"{default_model_name} with default filters", f"{noise}% noise") if learn_plot else None
        default_optimizer = optim.Adam(default_model.parameters(), lr=lr)
        _, _, default_valid_acc, default_test_acc = fit(default_model, epochs, default_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        default_valid_acc_history.append(default_valid_acc)
        default_test_acc_history.append(default_test_acc)

        # custom filter
        custom_filters = custom_filters_2(noise)
        custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
        custom_model = PulsarDetectionCNN_1(dim, custom_filters, custom_bias)
        custom_model_name = type(custom_model).__name__
        decorate_text(f"{custom_model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=lr)
        _, _, custom_valid_acc, custom_test_acc = fit(custom_model, epochs, custom_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        custom_valid_acc_history.append(custom_valid_acc)
        custom_test_acc_history.append(custom_test_acc)
    
    if graph_plot:
        x_range = range(1, epochs + 1)
        for j in range(len(noise_levels)):
            plt.plot(x_range, default_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (default filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        for j in range(len(noise_levels)):
            plt.plot(x_range, custom_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (custom filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        x = np.arange(len(noise_levels))  # Die x-Koordinaten der Balken
        width = 0.35  # Breite der Balken

        plt.bar(x - width/2, default_test_acc_history, width, label='default filters', color='skyblue')
        plt.bar(x + width/2, custom_test_acc_history, width, label='custom filters', color='orange')

        plt.xlabel('Noise Level (in %)')  # Beschriftung der x-Achse
        plt.ylabel('Accuracy')       # Beschriftung der y-Achse
        plt.title('Test Accuracy')  # Titel des Diagramms
        plt.xticks(x, noise_levels)     # Beschriftung der x-Achsen-Ticks

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

def test_3(dataset, epochs, lr, bs, learn_plot, graph_plot):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[files[0]][0][0])

    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []
    
    decorate_text(f"Test 3: default filters vs 'mini-pulsar' custom filters ({dim}x{dim})")

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs)

        # default filters
        default_model = PulsarDetectionCNN_1(dim)
        default_model_name = type(default_model).__name__
        decorate_text(f"{default_model_name} with default filters", f"{noise}% noise") if learn_plot else None
        default_optimizer = optim.Adam(default_model.parameters(), lr=lr)
        _, _, default_valid_acc, default_test_acc = fit(default_model, epochs, default_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        default_valid_acc_history.append(default_valid_acc)
        default_test_acc_history.append(default_test_acc)

        # custom filters
        filters = custom_filters_3(size=3, num=len(default_model.conv1.weight), noise=0.0)
        custom_filters = torch.from_numpy(filters)
        custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
        custom_model = PulsarDetectionCNN_1(dim, custom_filters, custom_bias)
        custom_model_name = type(custom_model).__name__
        decorate_text(f"{custom_model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=lr)
        _, _, custom_valid_acc, custom_test_acc = fit(custom_model, epochs, custom_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        custom_valid_acc_history.append(custom_valid_acc)
        custom_test_acc_history.append(custom_test_acc)
    
    if graph_plot:
        x_range = range(1, epochs + 1)
        for j in range(len(noise_levels)):
            plt.plot(x_range, default_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (default filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        for j in range(len(noise_levels)):
            plt.plot(x_range, custom_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
        plt.title('Validation Accuracy per Epoch (custom filters)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, epochs+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

        x = np.arange(len(noise_levels))  # Die x-Koordinaten der Balken
        width = 0.35  # Breite der Balken

        plt.bar(x - width/2, default_test_acc_history, width, label='default filters', color='skyblue')
        plt.bar(x + width/2, custom_test_acc_history, width, label='custom filters', color='orange')

        plt.xlabel('Noise Level (in %)')  # Beschriftung der x-Achse
        plt.ylabel('Accuracy')       # Beschriftung der y-Achse
        plt.title('Test Accuracy')  # Titel des Diagramms
        plt.xticks(x, noise_levels)     # Beschriftung der x-Achsen-Ticks

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
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

def plot_accuracy_epochs(epochs, noise_levels, default_valid_acc_history, custom_valid_acc_history, default_test_acc_history, custom_test_acc_history):
    x_range = range(1, epochs + 1)
    for j in range(len(noise_levels)):
        plt.plot(x_range, default_valid_acc_history[j], label=f"{noise_levels[j]}% noise")
    plt.title('Validation Accuracy per Epoch (default filters)')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, epochs+1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax = plt.gca()
    ax.set_ylim(bottom=0.5)
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
    ax.set_ylim(bottom=0.5)
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

def train_data(dataset, epochs, lr, bs, learn_plot, filters):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[files[0]][0][0])

    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []

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
        custom_filters = torch.from_numpy(filters)
        custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
        custom_model = PulsarDetectionCNN_1(dim, custom_filters, custom_bias)
        custom_model_name = type(custom_model).__name__
        decorate_text(f"{custom_model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=lr)
        _, _, custom_valid_acc, custom_test_acc = fit(custom_model, epochs, custom_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        custom_valid_acc_history.append(custom_valid_acc)
        custom_test_acc_history.append(custom_test_acc) 
    
    return default_test_acc_history, default_valid_acc_history, custom_test_acc_history, custom_valid_acc_history, noise_levels

# default filters vs fixed binary custom filters on model 1
def test_1(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    
    decorate_text(f"Test 1: default filters vs fixed binary custom filters ({dim}x{dim})")

    filters = custom_filters_1()
    default_test_acc_history, default_valid_acc_history, custom_test_acc_history, custom_valid_acc_history, noise_levels = train_data(dataset, epochs, lr, bs, learn_plot, filters)
    
    if graph_plot:
        plot_accuracy_epochs(epochs, noise_levels, default_valid_acc_history, custom_valid_acc_history, default_test_acc_history, custom_test_acc_history)

# default filters vs changing binary custom filters on model 1
def test_2(dataset, epochs, lr, bs, learn_plot, graph_plot):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[files[0]][0][0])

    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []

    decorate_text(f"Test 2: default filters vs changing binary custom filters ({dim}x{dim})")
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
        filters = custom_filters_3(noise)
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
        plot_accuracy_epochs(epochs, noise_levels, default_valid_acc_history, custom_valid_acc_history, default_test_acc_history, custom_test_acc_history)

# default filters vs 'mini-pulsar' custom filters on model 1
def test_3(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    
    decorate_text(f"Test 3: default filters vs 'mini-pulsar' custom filters ({dim}x{dim})")

    filters = custom_filters_3(size=3, num=16, noise=0.0)
    default_test_acc_history, default_valid_acc_history, custom_test_acc_history, custom_valid_acc_history, noise_levels = train_data(dataset, epochs, lr, bs, learn_plot, filters.astype(np.float32))
    
    if graph_plot:
        plot_accuracy_epochs(epochs, noise_levels, default_valid_acc_history, custom_valid_acc_history, default_test_acc_history, custom_test_acc_history)

# default filters vs pre-trained filters from 9x9 dataset. should only be tested on bigger dimensions
def test_4(dataset, dataset_9x9, epochs, lr, bs, learn_plot, graph_plot):
    labels = dataset["labels"]
    files = dataset.files
    dim = len(dataset[dataset.files[0]][0][0])

    # get pre trained network with 9x9
    data_9x9 = dataset_9x9[dataset.files[0]]
    labels_9x9 = dataset_9x9["labels"]
    train_loader_9x9, test_loader_9x9, valid_loader_9x9 = generate_train_test_valid_data(data_9x9, labels_9x9, bs=bs)

    default_model_9x9 = PulsarDetectionCNN_1(9)
    default_optimizer_9x9 = optim.Adam(default_model_9x9.parameters(), lr=0.01)
    fit(default_model_9x9, epochs, default_optimizer_9x9, nn.CrossEntropyLoss(), train_loader_9x9, test_loader_9x9, valid_loader_9x9, learn_plot=learn_plot)

    custom_filters = default_model_9x9.conv1.weight.data.cpu()
    print(custom_filters)
    
    default_test_acc_history = []
    default_valid_acc_history = []
    custom_test_acc_history = []
    custom_valid_acc_history = []
    noise_levels = []

    decorate_text(f"Test 4: default filters vs pre-trained filters from 9x9 dataset ({dim}x{dim})")
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
        custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
        custom_model = PulsarDetectionCNN_1(dim, custom_filters, custom_bias)
        custom_model_name = type(custom_model).__name__
        decorate_text(f"{custom_model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
        custom_optimizer = optim.Adam(custom_model.parameters(), lr=lr)
        _, _, custom_valid_acc, custom_test_acc = fit(custom_model, epochs, custom_optimizer, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
        custom_valid_acc_history.append(custom_valid_acc)
        custom_test_acc_history.append(custom_test_acc)
    
    if graph_plot:
        plot_accuracy_epochs(epochs, noise_levels, default_valid_acc_history, custom_valid_acc_history, default_test_acc_history, custom_test_acc_history)

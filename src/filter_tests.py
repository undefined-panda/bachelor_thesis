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
from src.cnn_models import PulsarDetectionNet

def train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader):
    model = PulsarDetectionNet(dim)
    model_name = type(model).__name__
    decorate_text(f"{model_name} with default filters", f"{noise}% noise") if learn_plot else None
    opt = optim.Adam(model.parameters(), lr=lr)
    _, _, valid_acc, test_acc = fit(model, epochs, opt, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
    
    return valid_acc, test_acc

def train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, filters):
    custom_filters = torch.from_numpy(filters)
    custom_bias = torch.zeros(custom_filters.shape[0], dtype=torch.float32)
    model = PulsarDetectionNet(dim, custom_filters, custom_bias)
    model_name = type(model).__name__
    decorate_text(f"{model_name} with custom filters", f"{noise}% noise")  if learn_plot else None
    opt = optim.Adam(model.parameters(), lr=lr)
    _, _, valid_acc, test_acc = fit(model, epochs, opt, nn.CrossEntropyLoss(), train_loader, test_loader, valid_loader, learn_plot=learn_plot)
    
    return valid_acc, test_acc

def plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist):
    x_range = range(1, epochs + 1)
    for j in range(len(noise_levels)):
        plt.plot(x_range, def_valid_acc_hist[j], label=f"{noise_levels[j]}% noise")
    plt.title('Validation Accuracy per Epoch (default filters)')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, epochs+1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.show()

    for j in range(len(noise_levels)):
        plt.plot(x_range, cust_valid_acc_hist[j], label=f"{noise_levels[j]}% noise")
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

    plt.bar(x - width/2, def_test_acc_hist, width, label='default filters', color='skyblue')
    plt.bar(x + width/2, cust_test_acc_hist, width, label='custom filters', color='orange')

    plt.xlabel('Noise Level (in %)')  # Beschriftung der x-Achse
    plt.ylabel('Accuracy')       # Beschriftung der y-Achse
    plt.title('Test Accuracy')  # Titel des Diagramms
    plt.xticks(x, noise_levels)     # Beschriftung der x-Achsen-Ticks

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def test_1(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    files = dataset.files
    labels = dataset["labels"]

    def_test_acc_hist = []
    def_valid_acc_hist = []
    cust_test_acc_hist = []
    cust_valid_acc_hist = []
    noise_levels = []
    
    decorate_text(f"Testing: default filters vs fixed binary custom filters ({dim}x{dim})")

    cust_filters = custom_filters_1()

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs, with_valid=True)

        # default filters
        def_valid_acc, def_test_acc = train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader)
        def_valid_acc_hist.append(def_valid_acc)
        def_test_acc_hist.append(def_test_acc)

        # custom filters
        cust_valid_acc, cust_test_acc = train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, cust_filters)
        cust_valid_acc_hist.append(cust_valid_acc)
        cust_test_acc_hist.append(cust_test_acc)
    
    if graph_plot:
        plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist)
    
def test_2(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    files = dataset.files
    labels = dataset["labels"]

    def_test_acc_hist = []
    def_valid_acc_hist = []
    cust_test_acc_hist = []
    cust_valid_acc_hist = []
    noise_levels = []
    
    decorate_text(f"Testing: default filters vs changing binary custom filters ({dim}x{dim})")

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs, with_valid=True)

        # default filters
        def_valid_acc, def_test_acc = train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader)
        def_valid_acc_hist.append(def_valid_acc)
        def_test_acc_hist.append(def_test_acc)

        # custom filters
        cust_filters = custom_filters_2(noise)
        cust_valid_acc, cust_test_acc = train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, cust_filters)
        cust_valid_acc_hist.append(cust_valid_acc)
        cust_test_acc_hist.append(cust_test_acc)
    
    if graph_plot:
        plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist)

def test_3(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    files = dataset.files
    labels = dataset["labels"]

    def_test_acc_hist = []
    def_valid_acc_hist = []
    cust_test_acc_hist = []
    cust_valid_acc_hist = []
    noise_levels = []
    
    decorate_text(f"Testing: default filters vs 'mini-pulsar' custom filters ({dim}x{dim})")

    cust_filters = custom_filters_3(3, 16, 0)

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs, with_valid=True)

        # default filters
        def_valid_acc, def_test_acc = train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader)
        def_valid_acc_hist.append(def_valid_acc)
        def_test_acc_hist.append(def_test_acc)

        # custom filters
        cust_valid_acc, cust_test_acc = train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, cust_filters)
        cust_valid_acc_hist.append(cust_valid_acc)
        cust_test_acc_hist.append(cust_test_acc)
    
    if graph_plot:
        plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist)

def test_4(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    files = dataset.files
    labels = dataset["labels"]

    def_test_acc_hist = []
    def_valid_acc_hist = []
    cust_test_acc_hist = []
    cust_valid_acc_hist = []
    noise_levels = []
    
    decorate_text(f"Testing: default filters vs pre-trained custom filters ({dim}x{dim})", 
                  "pre-training my 32x32 synthesized data")

    cust_filters = custom_filters_4()

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs, with_valid=True)

        # default filters
        def_valid_acc, def_test_acc = train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader)
        def_valid_acc_hist.append(def_valid_acc)
        def_test_acc_hist.append(def_test_acc)

        # custom filters
        cust_valid_acc, cust_test_acc = train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, cust_filters)
        cust_valid_acc_hist.append(cust_valid_acc)
        cust_test_acc_hist.append(cust_test_acc)
    
    if graph_plot:
        plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist)

def test_4(dataset, epochs, lr, bs, learn_plot, graph_plot):
    dim = len(dataset[dataset.files[0]][0][0])
    files = dataset.files
    labels = dataset["labels"]

    def_test_acc_hist = []
    def_valid_acc_hist = []
    cust_test_acc_hist = []
    cust_valid_acc_hist = []
    noise_levels = []
    
    decorate_text(f"Testing: default filters vs lbp-sae custom filters ({dim}x{dim})")

    for i in tqdm(range(len(files)-1), desc ="Training models"):
        noise = int(files[i].split("_")[1])
        noise_levels.append(noise)
        data = dataset[files[i]]
        train_loader, test_loader, valid_loader = generate_train_test_valid_data(data, labels, bs=bs, with_valid=True)

        # default filters
        def_valid_acc, def_test_acc = train_default_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader)
        def_valid_acc_hist.append(def_valid_acc)
        def_test_acc_hist.append(def_test_acc)

        # custom filters
        cust_filters = custom_filters_5(data, labels)
        cust_valid_acc, cust_test_acc = train_custom_model(dim, noise, learn_plot, lr, epochs, train_loader, test_loader, valid_loader, cust_filters)
        cust_valid_acc_hist.append(cust_valid_acc)
        cust_test_acc_hist.append(cust_test_acc)
    
    if graph_plot:
        plot_graph(epochs, noise_levels, def_valid_acc_hist, def_test_acc_hist, cust_valid_acc_hist, cust_test_acc_hist)
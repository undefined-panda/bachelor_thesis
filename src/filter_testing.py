"""
This file contains the tests for testing custom filters on the data.
"""
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from synthetic_data_generation import generate_train_test_valid_data
from models import DefaultNet, CustomNet, CannyNet
from utils import fit

def plot_graph(epochs, noise_level, def_acc_hist, cust_acc_hist, title=""):
    plt.figure(figsize=(8, 5))

    x_range = range(1, epochs + 1)
    plt.plot(x_range, def_acc_hist, label=f"default filter", color="#1f77b4")
    max_def_acc = np.argmax(def_acc_hist)
    plt.scatter(x_range[max_def_acc], def_acc_hist[max_def_acc], color='#1f77b4', zorder=5, label=f'Max Default Accuracy ({def_acc_hist[max_def_acc]*100}%)')

    plt.plot(x_range, cust_acc_hist, label=f"custom filter", color="orange")
    max_cust_acc = np.argmax(cust_acc_hist)
    plt.scatter(x_range[max_cust_acc], cust_acc_hist[max_cust_acc], color='orange', zorder=5, label=f'Max Custom Accuracy ({cust_acc_hist[max_cust_acc]*100}%)')

    if len(title) > 0:
        plt.title(f'Accuracy per Epoch ({title}: {noise_level}% noise, {epochs} epochs)')
    else:
        plt.title(f'Accuracy per Epoch ({noise_level}% noise), {epochs} epochs)')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, epochs+1, epochs//20))
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.ylim(top=1.01)

    plt.show()

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        
    return gaussian_2D

def test_canny(data, labels, noise, bs, epochs, lr, custom_filter_list, test_split=0.2, title=""):
    num_classes = len(set(labels))
    dim = len(data[0][0])
    train_data, test_data = generate_train_test_valid_data(data, labels, False, bs, test_split)

    default_model = DefaultNet(dim, num_classes)
    default_opt = optim.Adam(default_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, def_test_acc, _ = fit(default_model, epochs, default_opt, loss_fun, train_data, test_data, learn_plot=False)

    canny_kernel = np.array([[get_gaussian_kernel()]], dtype=np.float32)
    custom_filters = np.array([[f] for f in custom_filter_list], dtype=np.float32)
    custom_model = CannyNet(dim, num_classes, torch.from_numpy(canny_kernel), torch.from_numpy(custom_filters), train_custom_filters=False)

    custom_opt = optim.Adam(custom_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, cust_test_acc, _ = fit(custom_model, epochs, custom_opt, loss_fun, train_data, test_data, learn_plot=False)

    plot_graph(epochs, noise, def_test_acc, cust_test_acc, title)

def test_1(data, labels, noise, bs, epochs, lr, custom_filter_list, test_split=0.2, plot=True, title="", return_acc=False):
    """
    This test compares the accuracy on the test data by adding a conv layer with custom filters, 
    which won't be trained. 
    """

    num_classes = len(set(labels))
    dim = len(data[0][0])
    train_data, test_data = generate_train_test_valid_data(data, labels, False, bs, test_split)

    default_model = DefaultNet(dim, num_classes)
    default_opt = optim.Adam(default_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, def_test_acc, _ = fit(default_model, epochs, default_opt, loss_fun, train_data, test_data, learn_plot=False)

    custom_filters = np.array([[f] for f in custom_filter_list], dtype=np.float32)
    custom_model = CustomNet(dim, num_classes, torch.from_numpy(custom_filters), train_custom_filters=False, custom_filter_layer=True)

    custom_opt = optim.Adam(custom_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, cust_test_acc, _ = fit(custom_model, epochs, custom_opt, loss_fun, train_data, test_data, learn_plot=False)

    plot_graph(epochs, noise, def_test_acc, cust_test_acc, title) if plot else None
    print(f"Accuracy default: {def_test_acc[-1]*100}%\nAccuracy custom: {cust_test_acc[-1]*100}%")

    if return_acc:
        return def_test_acc[-1], cust_test_acc[-1]

def test_2(data, labels, noise, bs, epochs, lr, custom_filter_list, test_split=0.2, plot=True, title="", return_acc=False):
    """
    This test compares the accuracy on the test data by adding a conv layer with custom filters,
    which will be trained.
    """

    num_classes = len(set(labels))
    dim = len(data[0][0])
    train_data, test_data = generate_train_test_valid_data(data, labels, False, bs, test_split)

    default_model = DefaultNet(dim, num_classes)
    default_opt = optim.Adam(default_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, def_test_acc, _ = fit(default_model, epochs, default_opt, loss_fun, train_data, test_data, learn_plot=False)

    custom_filters = np.array([[f] for f in custom_filter_list], dtype=np.float32)
    custom_model = CustomNet(dim, num_classes, torch.from_numpy(custom_filters), train_custom_filters=True, custom_filter_layer=True)

    custom_opt = optim.Adam(custom_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, cust_test_acc, _ = fit(custom_model, epochs, custom_opt, loss_fun, train_data, test_data, learn_plot=False)

    plot_graph(epochs, noise, def_test_acc, cust_test_acc, title) if plot else None
    print(f"Accuracy default: {def_test_acc[-1]*100}%\nAccuracy custom: {cust_test_acc[-1]*100}%")

    if return_acc:
        return def_test_acc[-1], cust_test_acc[-1]

def test_3(data, labels, noise, bs, epochs, lr, custom_filter_list, test_split=0.2, title=""):
    """
    This test compares the accuracy on the test data by changing the filters of the first conv layer
    with custom filters, which won't be trained. 
    """

    num_classes = len(set(labels))
    dim = len(data[0][0])
    train_data, test_data = generate_train_test_valid_data(data, labels, False, bs, test_split)

    default_model = DefaultNet(dim, num_classes)
    default_opt = optim.Adam(default_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, def_test_acc, _ = fit(default_model, epochs, default_opt, loss_fun, train_data, test_data, learn_plot=False)

    custom_filters = np.array([[f] for f in custom_filter_list], dtype=np.float32)
    custom_model = CustomNet(dim, num_classes, torch.from_numpy(custom_filters), train_custom_filters=False, custom_filter_layer=False)

    custom_opt = optim.Adam(custom_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, cust_test_acc, _ = fit(custom_model, epochs, custom_opt, loss_fun, train_data, test_data, learn_plot=False)

    plot_graph(epochs, noise, def_test_acc, cust_test_acc, title)
    print(f"Max accuracy default: {np.max(def_test_acc)}\nMax accuracy custom: {np.max(cust_test_acc)}")

def test_4(data, labels, noise, bs, epochs, lr, custom_filter_list, test_split=0.2, title=""):
    """
    This test compares the accuracy on the test data by changing the filters of the first conv layer
    with custom filters, which will be trained. 
    """

    num_classes = len(set(labels))
    dim = len(data[0][0])
    train_data, test_data = generate_train_test_valid_data(data, labels, False, bs, test_split)

    default_model = DefaultNet(dim, num_classes)
    default_opt = optim.Adam(default_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, def_test_acc, _ = fit(default_model, epochs, default_opt, loss_fun, train_data, test_data, learn_plot=False)

    custom_filters = np.array([[f] for f in custom_filter_list], dtype=np.float32)
    custom_model = CustomNet(dim, num_classes, torch.from_numpy(custom_filters), train_custom_filters=True, custom_filter_layer=False)

    custom_opt = optim.Adam(custom_model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    _, _, _, cust_test_acc, _ = fit(custom_model, epochs, custom_opt, loss_fun, train_data, test_data, learn_plot=False)

    plot_graph(epochs, noise, def_test_acc, cust_test_acc, title)
    print(f"Max accuracy default: {np.max(def_test_acc)}\nMax accuracy custom: {np.max(cust_test_acc)}")
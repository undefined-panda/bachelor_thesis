"""
This file contains helper functions for better print, calculations and the fit function.

@author Jad Dayoub, 7425569
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def exponential_decay(x, a, b, c, d):
    """
    Exponential decaying function that mimics pulsar form.
    """

    return a * np.exp(-b * (x + c)) + d

def manhatten_distance(x, y):
    """
    Calculates the manhatten distance of two points.
    """

    x = np.array(x)
    y = np.array(y)
    return np.sum(np.absolute(x-y))

def my_round(x):
    """ 
    Does commercial rounding on a number.
    """

    x_str = format(x, ".10f")
    decimal_part = x_str.split('.')[1]
    first_decimal_digit = decimal_part[0]
    if x < 0:
        if int(first_decimal_digit) < 5:
            x = math.ceil(x)
        else:
            x = math.floor(x)
    else:
        if int(first_decimal_digit) < 5:
            x = math.floor(x)
        else:
            x = math.ceil(x)
    
    return x

def number_iterations(distribution, total):
    """
    Calculates number of iterations based on distribution.
    """

    raw_distribution = [total * ratio for ratio in distribution]
    
    int_distribution = [int(amount) for amount in raw_distribution]
    
    discrepancy = total - sum(int_distribution)
    for i in range(discrepancy):
        int_distribution[i % len(int_distribution)] += 1
    
    return int_distribution

def print_data(data, labels):
    """ 
    Prints data as grayscale image.
    """

    data_label, label_count = np.unique(labels, return_counts=True)
    for i in range(len(label_count)):
        print(f"Label {data_label[i]}: {label_count[i]}")

    for label in range(len(data_label)):
        label_indices = np.random.choice(np.where(labels == label)[0], size=5, replace=False)
        plt.figure(figsize=(10, 2))
        for i, idx in enumerate(label_indices):
            plt.subplot(1, 5, i+1)
            plt.imshow(data[idx][0], cmap='gray', vmin=0, vmax=255)
            plt.title(f"Label: {label}")
            plt.axis('off')
        plt.show()
    
def decorate_text(text1, text2="", symbol="-", width=90):
    """
    Printing a header with given text.
    """

    line = symbol * width
    half_width = (width - len(text1)) // 2
    line1 = symbol * (half_width - 1) + " " + text1 + " " + symbol * (width - half_width - len(text1) -1)
    
    if text2:
        half_width = (width - len(text2)) // 2
        line2 = symbol * half_width + " " + text2 + " " + symbol * (width - half_width - len(text2))
    else:
        line2 = ""
    
    print(line)
    print(line1)
    if line2:
        print(line2)
    print(line)

def extract_percentage(n):
    """
    Extracts percentage of key value of .npz file.
    """

    if len(n) > 0:
        return str(int(n.split("_")[1]))+"% noise"
    
def get_odd_number(index, x):
    """
    Used to get index when printing data.
    """

    if index % 2 == 0:
        return "Index is even."
    
    odd_index_count = 0
    
    for i in range(x + 1):
        if i % 2 != 0:  # Check if the index is odd
            odd_index_count += 1
            if i == index:
                return odd_index_count
    
    return "Der Index liegt außerhalb des Bereichs."

def replace_value_with_value(filters, value1, value2):
    """
    Replaces a value with another value of a numpy array.
    """

    return np.where(filters == value1, value2, filters)

def train_model(model, epoch, opt, loss_fun, train_data, device, valid_data, learn_plot=True):
    """
    Trains the model on train_data and tests on valid_data, if provided.
    """

    train_loss = 0
    model.train()
    for sample, label in train_data:
        sample, label = sample.to(device), label.to(device)
        opt.zero_grad()
        output = model(sample)
        loss = loss_fun(output, label)
        loss.backward()
        opt.step()
        train_loss += loss.item() * sample.size(0)
    
    train_loss /= len(train_data.dataset)

    if valid_data:
        model.eval()
        valid_loss = 0.0
        correct = 0
        model.eval()
        with torch.no_grad():
            for sample, label in valid_data:
                sample, label = sample.to(device), label.to(device)
                output = model(sample)
                valid_loss += loss_fun(output, label).item() * sample.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                true_labels = label.argmax(dim=1, keepdim=True)
                correct += (pred == true_labels).sum().item()
        
        
        total = len(valid_data.dataset)
        valid_loss /= len(valid_data)
        valid_acc = correct/total
        print(f"Train Epoch {epoch}\tTrain Loss: {train_loss:.6f}\tValid Loss: {valid_loss:.6f}\tValid Acc: {correct}/{total} ({100. * valid_acc}%)") if learn_plot else None
        return train_loss, valid_acc, valid_loss
    
    else:
        print(f"Train Epoch {epoch}\tTrain Loss: {train_loss:.6f}") if learn_plot else None
        return train_loss

def test_model(model, loss_fun, device, test_data, learn_plot=True):
    """
    Tests the model on test_data.
    """

    print() if learn_plot else None
    model.eval()

    test_loss = 0
    correct = 0
    total = len(test_data.dataset)
    all_predictions = []
    all_true_labels = []
    true_labels = []
    with torch.no_grad():
        for sample, label in test_data:
            sample, label = sample.to(device), label.to(device)
            output = model(sample)
            test_loss += loss_fun(output, label).item() * sample.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            true_labels = label.argmax(dim=1, keepdim=True)
            correct += (pred == true_labels).sum().item()

            all_predictions.extend(pred.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())

        test_acc = correct/total

        avg_test_loss = test_loss/total
        print(f"Test Accuracy: {correct}/{total} ({100. * test_acc}%), Avg Loss: {avg_test_loss:.4f}\n") if learn_plot else None
        
        if learn_plot:
            confusion_mat = confusion_matrix(all_true_labels, all_predictions)
            classification_rep = classification_report(all_true_labels, all_predictions)
            
            print("\nClassification Report:\n", classification_rep, "\n")
            print("Confusion Matrix:\n", confusion_mat)
    
    return test_acc, avg_test_loss

def fit(model, epochs, opt, loss_fun, train_data, test_data=None, valid_data=None, learn_plot=True):
    """
    This function trains the model and, if given, also tests it.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss_history = []
    valid_acc_history = []
    valid_loss_history = []
    test_acc_history = []
    test_loss_history = []

    for epoch in tqdm(range(1, epochs + 1), desc="Train model"):
        if valid_data:
            train_loss, valid_acc, valid_loss = train_model(model, epoch, opt, loss_fun, train_data, device, valid_data, learn_plot)
            train_loss_history.append(train_loss)
            valid_acc_history.append(valid_acc)
            valid_loss_history.append(valid_loss)
        else:
            train_loss = train_model(model, epoch, opt, loss_fun, train_data, device, valid_data, learn_plot)
            train_loss_history.append(train_loss)
        
        if test_data:
            test_acc, test_loss = test_model(model, loss_fun, device, test_data, learn_plot)
            test_acc_history.append(test_acc)
            test_loss_history.append(test_loss)
    
    train_loss_history = train_loss_history if train_loss_history else None
    valid_loss_history = valid_loss_history if valid_loss_history else None
    valid_acc_history = valid_acc_history if valid_acc_history else None
    test_acc_history = test_acc_history if test_acc_history else None
    test_loss_history = test_loss_history if test_loss_history else None

    return train_loss_history, valid_loss_history, valid_acc_history, test_acc_history, test_loss_history

def get_num_classes(labels):
    num_classes = np.unique(labels)

    return len(num_classes)
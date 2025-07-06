import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_metric(values, title="Metric", ylabel="Value", xlabel="Epoch", save_path=None):
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

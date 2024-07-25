import numpy as np

import matplotlib
from matplotlib import pyplot as plt

    
def plot_estimated_MI_trainig(true_mi: float, epochs, estimated_MI, estimated_latent_MI=None):
    """
    Plot mutual information estimate during training.
    
    Parameters
    ----------
    true_mi : float
        True value of the mutual information
    epochs : iterable
        Epochs array (x axis)
    estimated_MI : iterable
        Mutual iformation estimates during training
    estimated_latent_MI : iterable (optional)
        Mutual iformation estimates based on latent representation during training
    """
    
    fig, ax = plt.subplots()

    fig.set_figheight(9)
    fig.set_figwidth(16)

    # Grid.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title("Mutual information estimate while training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$ I(X;Y) $")
    
    ax.minorticks_on()

    if not estimated_latent_MI is None:
        ax.plot(epochs, estimated_latent_MI, label="$ \\hat I(\\xi,\\eta) $")
    ax.plot(epochs, estimated_MI, label="$ \\hat I(X,Y) $")
    ax.hlines(y=true_mi, xmin=min(epochs), xmax=max(epochs), color='red', label="$ I(X,Y) $")

    ax.legend(loc='upper left')

    plt.show();
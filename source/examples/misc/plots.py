import torch
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


def plot_pointwise_mi(model: torch.nn.Module, device: str,
                      dimension_x: int, dimension_y: int, x_index: int, y_index: int,
                      grid_size: int=101, xlim: tuple=(0.0, 1.0), ylim: tuple=(0.0, 1.0),
                      default_value: float=0.5, data_points: torch.tensor=None) -> None:
    
    figure, axes = plt.subplots()

    figure.set_figheight(10)
    figure.set_figwidth(10)

    # Grid.
    axes.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    axes.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    axes.set_title("Pointwise mutual information plot")
    axes.set_xlabel("$x_i$")
    axes.set_ylabel("$x_j$")
    
    axes.minorticks_on()

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    # Exit training mode.
    was_in_training = model.training
    model.eval()

    with torch.no_grad():
        x = torch.linspace(xlim[0], xlim[1], grid_size)
        y = torch.linspace(ylim[0], ylim[1], grid_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        
        full_x_grid = default_value * torch.ones((grid_size**2, dimension_x), device=device)
        full_y_grid = default_value * torch.ones((grid_size**2, dimension_y), device=device)
        
        full_x_grid[:,x_index] = x_grid.flatten().to(device)
        full_y_grid[:,y_index] = y_grid.flatten().to(device)
        
        pmi = model(full_x_grid, full_y_grid).view(grid_size, grid_size).detach().cpu()
        
        axes.pcolormesh(x_grid, y_grid, pmi.data.numpy(), shading='gouraud', cmap='coolwarm')

    if not (data_points is None):
        data_points = data_points.detach().cpu().numpy()
        axes.scatter(data_points[:,x_index], data_points[:,dimension_x + y_index], s=3.0, color="black")

    # Return to the original mode.
    model.train(was_in_training)

    plt.show()
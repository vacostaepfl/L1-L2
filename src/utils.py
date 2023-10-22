import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_signal(sparse: np.ndarray, smooth: np.ndarray) -> None:
    """
    Plot the sparse, smooth, and combined signal.

    Parameters:
        sparse (np.ndarray): A NumPy array representing the sparse signal.
        smooth (np.ndarray): A NumPy array representing the smooth signal.
    """
    signal = sparse + smooth
    fig, axes = plt.subplots(1, 3)
    for ax, img, title in zip(
        axes, [sparse, smooth, signal], ["Sparse", "Smooth", "Signal"]
    ):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        divnorm = colors.CenteredNorm(vcenter=0.0)
        im = ax.imshow(img, cmap="seismic", norm=divnorm)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax1)
    # fig.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    plt.show()


def nmse(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Normalized Mean Squared Error (NMSE) between two signals.

    Parameters:
        x1 (np.ndarray): The first signal.
        x2 (np.ndarray): The second signal.

    Returns:
        float: The NMSE between the two signals.
    """
    return np.mean((x1 - x2) ** 2) / np.mean(x1**2)


def wasserstein_dist(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Wasserstein distance between two probability distributions represented by the input signals.

    Parameters:
        x1 (np.ndarray): The first probability distribution.
        x2 (np.ndarray): The second probability distribution.

    Returns:
        float: The Wasserstein distance between the two probability distributions.
    """
    return wasserstein_distance(x1.ravel() / np.sum(x1), x2.ravel() / np.sum(x2))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.operators import NuFFT
from IPython.display import display
from ipywidgets import widgets
import os

EXP_PATH = "exps"


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


def plot_results(
    original_signal: list,
    reconstructed_signal: list,
    lambda1: float,
    lambda2: float,
    op: NuFFT,
    coupled: bool = True,
):
    """
    Plot the results of a coupled or decoupled problem.

    Parameters:
    original_signal (list): A list of original signal data.
    reconstructed_signal (list): A list of reconstructed signal data.
    lambda1 (float): The value of lambda1.
    lambda2 (float): The value of lambda2.
    op (NuFFT): NuFFT operator for plotting frequency samples.
    coupled (bool, optional): Indicates if the problem is coupled. Defaults to True.
    """
    fig = plt.figure(figsize=(10, 5))
    fig.set_facecolor("0.85")
    fig.suptitle(
        str("Coupled" if coupled else "Decoupled")
        + r" problem with $\lambda_1=$"
        + str(lambda1)
        + r" & $\lambda_2=$"
        + str(lambda2),
        y=1,
        fontsize="xx-large",
        verticalalignment="bottom",
    )

    subfig_signals, subfig_sample = fig.subfigures(1, 2, width_ratios=[3, 1])

    # Plot samples
    ax = subfig_sample.subplots(1, 1)
    op.plot_samples(subfig_sample, ax)

    # Plot signals
    subfig_signals.suptitle(
        f"Signal size: {op.dim}",
        position=(0.515, 0.1),
        fontsize="medium",
    )
    signals = np.array(
        [
            original_signal + [np.add(*original_signal)],
            reconstructed_signal + [np.add(*reconstructed_signal)],
        ]
    )

    halfranges = np.max(signals, axis=(0, 2, 3))
    for i, (subfig, title) in enumerate(
        zip(
            subfig_signals.subfigures(2, 1, hspace=-0.3),
            ["Original Signal", "Reconstruction"],
        )
    ):
        subfig.suptitle(
            title, rotation=90, position=(0.1, 0.5), ha="right", va="center"
        )
        for j, (ax, sig, title) in enumerate(
            zip(subfig.subplots(1, 3), signals[i], ["Sparse", "Smooth", "Signal"])
        ):
            if i == 0:
                ax.set_title(title, fontsize="medium")
            ax.set_xticks([])
            ax.set_yticks([])
            # halfrange if needs to set min max colorbar
            divnorm = colors.CenteredNorm(vcenter=0.0, halfrange=halfranges[j])
            im = ax.imshow(
                sig,
                cmap="seismic",
                norm=divnorm,
            )
            if i == 0:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(position="bottom", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, orientation="horizontal")
    plt.show()
    return fig


def save_widget(
    fig: plt.Figure,
    lambda1: float,
    lambda2: float,
    op: NuFFT,
    psnr: int,
    coupled: bool = True,
):
    """
    Saves a Matplotlib figure to a specified directory with a filename based on parameters.

    Parameters:
    - fig (plt.Figure): The Matplotlib figure to be saved.
    - lambda1 (float): The value of lambda1.
    - lambda2 (float): The value of lambda2.
    - op (NuFFT): An instance of the 'SomeClass' class.
    - psnr (int): Peak signal to noise ratio
    - coupled (bool, optional): Whether the figure is coupled. Default is True.

    Returns:
    - None

    This function displays a 'SAVE' button and, when clicked, saves the figure with a filename
    based on the parameters 'lambda1', 'lambda2', 'op', and whether it is coupled or decoupled.

    The figure is saved to a directory structure determined by the values of 'op.dim' and 'op.L'.
    If the directory does not exist, it will be created.

    Example:
    save_widget(fig, 0.1, 0.2, some_op_instance, coupled=True)
    """

    button = widgets.Button(description="SAVE")
    output = widgets.Output()

    display(button, output)

    def on_button_clicked(b):
        filename = save_fig(fig, lambda1, lambda2, op, psnr, coupled)
        with output:
            print(f"Figure saved as {filename}")

    button.on_click(on_button_clicked)


def save_fig(
    fig: plt.Figure,
    lambda1: float,
    lambda2: float,
    op: NuFFT,
    psnr: int,
    coupled: bool = True,
):
    """
    Save a matplotlib figure to a specified directory with a filename generated
    based on the input parameters.

    Parameters:
    - fig (plt.Figure): The matplotlib figure to be saved.
    - lambda1 (float): The value of lambda1.
    - lambda2 (float): The value of lambda2.
    - op (NuFFT): An instance of NuFFT class.
    - psnr (int): The value of PSNR.
    - coupled (bool, optional): indicating whether the problem is coupled or decoupled.
                                Defaults to True.

    Returns:
    - str: The filename of the saved figure.
    """
    directory = f"{EXP_PATH}/{op.dim[0]}x{op.dim[1]}/"
    directory += f"L_{2*op.L/np.prod(op.dim):.0%}/"
    directory += f"psnr_{psnr}/"
    directory += "coupled/" if coupled else "decoupled/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = f"l1_{lambda1:.1e}_"
    name += f"l2_{lambda2:.1e}"
    name += ".png"
    filename = directory + name
    fig.savefig(filename, bbox_inches="tight")
    return filename

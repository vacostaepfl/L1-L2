import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.operators import NuFFT
from IPython.display import display
from ipywidgets import widgets
import os
from pyxu.abc import LinOp
import csv

EXP_PATH = "exps"


def plot_signal(sparse: np.ndarray, smooth: np.ndarray) -> None:
    """
    Plot the sparse, smooth, and combined signal.

    Parameters:
        sparse (np.ndarray): A NumPy array representing the sparse signal.
        smooth (np.ndarray): A NumPy array representing the smooth signal.
    """
    signal = sparse + smooth
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    for ax, img, title in zip(
        axes, [sparse, smooth, signal], ["Sparse", "Smooth", "Signal"]
    ):
        ax.set_title(title, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        divnorm = colors.CenteredNorm(vcenter=0.0)
        im = ax.imshow(img, cmap="seismic", norm=divnorm)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax1)
        cbar.ax.tick_params(labelsize=16)
    # fig.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    return fig


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
    coupled: bool,
    laplacian: bool,
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
        str("Coupled " if coupled else "Decoupled ")
        + "problem "
        + str("with Laplacian" if laplacian else "without Laplacian")
        + r" ($\lambda_1=$"
        + str(lambda1)
        + r" & $\lambda_2=$"
        + f"{lambda2})",
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
        f"Signal size: {(op.N,op.N)}",
        position=(0.515, 0.1),
        fontsize="medium",
    )
    signals = np.array(
        [
            original_signal + [np.add(*original_signal)],
            reconstructed_signal + [np.add(*reconstructed_signal)],
        ]
    )

    halfranges = np.max(np.abs(signals), axis=(0, 2, 3))
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

    The figure is saved to a directory structure determined by the values of 'op.N' and 'op.L'.
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
    coupled: bool,
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
    - coupled (bool): indicating whether the problem is coupled or decoupled.

    Returns:
    - str: The filename of the saved figure.
    """
    directory = f"{EXP_PATH}/{op.N}/"
    if isinstance(op.L, float):
        directory += f"L_{op.L:.0%}/"
    else:
        directory += f"L_{op.L:.0f}/"
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


def objective_func(
    op: NuFFT,
    laplacian_op: LinOp,
    y: np.ndarray,
    sparse_rcstr: np.ndarray,
    smooth_rcstr: np.ndarray,
    lambda1: float,
    lambda2: float,
):
    """
    Calculate the objective function composed of three terms:
    - Data fidelity term: measures the difference between the predicted value and the actual value.
    - Smoothness regularization term: encourages smoothness in the reconstructed signal.
    - Sparsity regularization term: promotes sparsity in the reconstructed signal.

    Args:
    - op (NuFFT): Operator
    - laplacian (LinOP): Laplacian Operator.
    - y (np.ndarray): Actual values.
    - sparse_rcstr (np.ndarray): Sparse reconstruction.
    - smooth_rcstr (np.ndarray): Smooth reconstruction.
    - lambda1 (float): Weight for sparsity regularization.
    - lambda2 (float): Weight for smoothness regularization.

    Returns:
    - tuple: Three terms of the objective
    """
    return (
        1
        / 2
        * np.sum((op(sparse_rcstr.reshape(-1) + smooth_rcstr.reshape(-1)) - y) ** 2),
        (lambda2 / 2) * np.sum(laplacian_op.apply(smooth_rcstr.reshape(-1)) ** 2),
        lambda1 * np.sum(np.abs(sparse_rcstr)),
    )


def compare(
    N,
    laplacian,
    lambda1,
    lambda2,
    sparse_rcstr_coupled,
    sparse_rcstr_decoupled,
    smooth_rcstr_coupled,
    smooth_rcstr_decoupled,
    signal_rcstr_coupled,
    signal_rcstr_decoupled,
):
    """
    Compare different reconstructions.

    Args:
    - N (int): Size of the signal.
    - laplacian (bool): Flag indicating whether Laplacian regularization was used.
    - sparse_rcstr_coupled (numpy.ndarray): Sparse coupled reconstruction.
    - sparse_rcstr_decoupled (numpy.ndarray): Sparse decoupled reconstruction.
    - smooth_rcstr_coupled (numpy.ndarray): Smooth coupled reconstruction.
    - smooth_rcstr_decoupled (numpy.ndarray): Smooth decoupled reconstruction.
    - signal_rcstr_coupled (numpy.ndarray): Signal coupled reconstruction.
    - signal_rcstr_decoupled (numpy.ndarray): Signal decoupled reconstruction.

    Returns:
    - matplotlib.figure.Figure: Figure object displaying the reconstructions.
    """
    fig = plt.figure(figsize=(15, 18))
    if laplacian:
        title = "Reconstruction with Laplacian "
    else:
        title = "Reconstruction without Laplacian "
    fig.suptitle(
        title + r"($\lambda_1=$" + str(lambda1) + r" & $\lambda_2=$" + f"{lambda2})",
        y=1.03,
        fontsize=24,
        x=0.525,
    )
    figs = fig.subfigures(3, 1, hspace=-0.05)
    for i, (f, signals, title) in enumerate(
        zip(
            figs,
            [
                (sparse_rcstr_coupled, sparse_rcstr_decoupled),
                (smooth_rcstr_coupled, smooth_rcstr_decoupled),
                (signal_rcstr_coupled, signal_rcstr_decoupled),
            ],
            ["Sparse Reconstruction", "Smooth Reconstruction", "Signal Reconstruction"],
        )
    ):
        axes = f.subplots(1, 2)
        plt.subplots_adjust(wspace=0)
        f.suptitle(title, fontsize=20)
        for j in range(2):
            divnorm = colors.CenteredNorm(
                vcenter=0.0,
                halfrange=max(np.max(np.abs(signals[0])), np.max(np.abs(signals[1]))),
            )
            im = axes[j].imshow(
                signals[j],
                cmap="seismic",
                norm=divnorm,
            )
            axes[j].set_yticks([])
            axes[j].set_xticks([])
            divider = make_axes_locatable(axes[j])
            if j == 1:
                axes[j].set_title("Decoupled", fontsize=16)
                cax = divider.append_axes(position="right", size="5%", pad=0.5)
                cbar = f.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=16)
            else:
                axes[j].set_title("Coupled", fontsize=16)
                cax = divider.append_axes(position="left", size="5%", pad=0.5)
                cax.axis("off")
        if i == 2:
            f.text(
                s=f"Signal size: {(N,N)}",
                x=0.5,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=16,
                y=0,
            )
    return fig


def difference(
    N,
    laplacian,
    lambda1,
    lambda2,
    sparse_rcstr_coupled,
    sparse_rcstr_decoupled,
    smooth_rcstr_coupled,
    smooth_rcstr_decoupled,
):
    if laplacian:
        title = "Reconstruction difference with Laplacian "
    else:
        title = "Reconstruction difference without Laplacian "

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        title + r"($\lambda_1=$" + str(lambda1) + r" & $\lambda_2=$" + f"{lambda2})",
        fontsize=20,
    )
    for j, signal in enumerate(
        zip(
            [
                sparse_rcstr_coupled - sparse_rcstr_decoupled,
                smooth_rcstr_coupled - smooth_rcstr_decoupled,
            ]
        )
    ):
        divnorm = colors.CenteredNorm(vcenter=0.0, halfrange=np.max(np.abs(signal)))
        im = axes[j].imshow(
            signal[0],
            cmap="seismic",
            norm=divnorm,
        )
        axes[j].set_yticks([])
        axes[j].set_xticks([])

        divider = make_axes_locatable(axes[j])
        if j == 0:
            axes[j].set_title("Sparse", fontsize=16)
            cax = divider.append_axes(position="left", size="5%", pad=0.5)
            cbar = plt.colorbar(im, cax=cax, location="left")
        else:
            axes[j].set_title("Smooth", fontsize=16)
            cax = divider.append_axes(position="right", size="5%", pad=0.5)
            cbar = plt.colorbar(im, cax=cax, location="right")
        cbar.ax.tick_params(labelsize=16)

        fig.text(
            s=f"Signal size: {(N,N)}",
            x=0.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=16,
            y=0.05,
        )

    return fig


def append_to_txt(filename, content):
    with open(filename, "a") as file:
        file.write(content)


def sparse_error(
    N,
    laplacian,
    lambda1,
    lambda2,
    sparse_signal,
    sparse_rcstr_coupled,
    sparse_rcstr_decoupled,
):
    fig = plt.figure(figsize=(16, 15))
    if laplacian:
        title = "Sparse Error with Laplacian "
    else:
        title = "Sparse Error without Laplacian "
    fig.suptitle(
        title + r"($\lambda_1=$" + str(lambda1) + r" & $\lambda_2=$" + f"{lambda2})",
        y=1,
        fontsize=24,
    )
    figs = fig.subfigures(2, 1, hspace=-0.15)

    for i, (f, peaks) in enumerate(zip(figs, ["On spikes", "Off spikes"])):
        plt.subplots_adjust(wspace=0.1)
        axes = f.subplots(1, 2)
        if i == 0:
            divnorm = colors.CenteredNorm(
                vcenter=0.0,
                halfrange=max(
                    np.max(
                        np.abs(
                            (sparse_rcstr_coupled - sparse_signal)[sparse_signal != 0]
                        )
                    ),
                    np.max(
                        np.abs(
                            (sparse_rcstr_decoupled - sparse_signal)[sparse_signal != 0]
                        )
                    ),
                ),
            )
        else:
            divnorm = colors.CenteredNorm(
                vcenter=0.0,
                halfrange=max(
                    np.max(
                        np.abs(
                            (sparse_rcstr_coupled - sparse_signal)[sparse_signal == 0]
                        )
                    ),
                    np.max(
                        np.abs(
                            (sparse_rcstr_decoupled - sparse_signal)[sparse_signal == 0]
                        )
                    ),
                ),
            )
        for j in range(2):
            if j == 0:
                axes[0].set_ylabel(peaks, fontsize=20)
                sig = sparse_rcstr_coupled - sparse_signal
                if i == 0:
                    sig[sparse_signal == 0] = 0
                    axes[j].set_title("Coupled", fontsize=20)
                else:
                    sig[sparse_signal != 0] = 0
            else:
                sig = sparse_rcstr_decoupled - sparse_signal
                if i == 0:
                    sig[sparse_signal == 0] = 0
                    axes[j].set_title("Decoupled", fontsize=20)
                else:
                    sig[sparse_signal != 0] = 0

            im = axes[j].imshow(
                sig,
                cmap="seismic",
                norm=divnorm,
            )
            axes[j].set_yticks([])
            axes[j].set_xticks([])

        cbar_width = 0.5
        cbar_x = (1 - cbar_width) / 2
        cbar_ax = f.add_axes([cbar_x + 0.015, 0.09, cbar_width, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=16)
    f.text(
        s=f"Signal size: {(N,N)}",
        x=0.5,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=20,
        y=-0.04,
    )
    return fig


def write_to_csv(filename, data):
    """
    Write data to a CSV file, appending to existing content or creating a new file with a header row if it's empty.

    Args:
    - filename (str): The name of the CSV file.
    - data (list): A list representing a row of data to be written to the CSV file.

    The CSV file will contain the following header:
    "seed", "size", "coupled", "laplacian", "lambda1", "lambda2", "time", "error", "l2", "l1"
    """
    with open(EXP_PATH + filename, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty
            header = [
                "seed",
                "N",
                "coupled",
                "laplacian",
                "lambda1",
                "lambda2",
                "time",
                "error",
                "l2",
                "l1",
            ]
            writer.writerow(header)  # Write the header only if the file is empty
        writer.writerow(data)

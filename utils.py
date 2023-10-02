import numpy as np


def plot_2_reconstruction(
    sparse: np.ndarray, smooth: np.ndarray, name: str = ""
) -> None:
    """
    Plot the two image, a sparse one and a smooth one

    Parameters
    ----------
    sparse : np.ndarray
        Sparse image to plot
    smooth : np.ndarray
        Smooth image to plot
    name : str
        Name used as title, usually contains the parameters used in the solver
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f"Spare + Smooth Signal : {name}")
    fig.suptitle(name)

    im_p = ax1.imshow(
        sparse,
    )  # vmin=0, vmax=np.max(sparse)
    ax1.axis("off")

    im_s = ax2.imshow(
        smooth,
    )  # vmin=0, vmax=np.max(smooth)
    ax2.axis("off")

    fig.subplots_adjust(
        bottom=0.05, top=0.9, left=0.01, right=0.92, wspace=0.1, hspace=0.1
    )
    cb_ax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_s, cax=cb_ax)
    cb_ax = fig.add_axes([0.445, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_p, cax=cb_ax)

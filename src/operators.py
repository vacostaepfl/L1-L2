import numpy as np
import matplotlib.pyplot as plt
import pyxu.operator as pxo
from pyxu.abc import LinOp

from matplotlib.figure import Figure
from matplotlib.axes import Axes


class NuFFT:
    def __init__(
        self, dim: tuple, L: int | float, theta: float, on_grid: bool, seed: int = False
    ):
        """
        Initialize the NuFFT object.

        Args:
            dim (tuple): Dimensions of the input data.
            L (int | float): The number or proportion of N^2/2 samples to keep in the NuFFT.
            theta (float): Fraction of samples that are Gaussian.
            on_grid (bool): Whether to sample on a grid.
            seed (int, optional): Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.dim: tuple = dim
        self.L = round(L * np.prod(self.dim) / 2) if isinstance(L, float) else L
        assert (
            self.L <= np.prod(self.dim) / 2
        ), f"L={self.L} should not be larger than the product of the dimensions ({np.prod(dim)/2})"
        self.theta: float = theta
        self.on_grid: tuple = on_grid
        self.nb_gaussian: int = round(self.theta * self.L)
        self.nb_uniform: int = self.L - self.nb_gaussian

        self.phi: LinOp = None
        self.gaussian_samples: np.ndarray = None
        self.gaussian_indices: np.ndarray = None
        self.uniform_samples: np.ndarray = None
        self.uniform_indices: np.ndarray = None
        self.samples: np.ndarray = None

        self.compute_NuFFT()
        self.dim_in: int = self.phi.shape[1]
        self.dim_out: int = self.phi.shape[0]

    def __call__(self, x) -> np.ndarray:
        """
        Apply the NuFFT to the input data x.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: The result of the NuFFT applied to x.
        """
        return self.phi.apply(x)

    def compute_NuFFT(self):
        """
        Create the NuFFT linear operator.
        """
        self.mix_sampling()
        self.phi = pxo.NUFFT.type2(
            self.samples, self.dim, isign=-1, eps=1e-3, real=True
        )

    def mix_sampling(self):
        """
        Generate a mixture of Gaussian and uniform samples for NUFFT.
        """
        if self.nb_gaussian > 0 and self.nb_uniform > 0:
            self.gaussian_sampling()
            self.uniform_sampling()
            self.samples = np.concatenate([self.gaussian_samples, self.uniform_samples])
        elif self.nb_gaussian > 0:
            self.gaussian_sampling()
            self.samples = self.gaussian_samples
        else:
            self.uniform_sampling()
            self.samples = self.uniform_samples

    def uniform_sampling(self):
        """
        Generate uniform samples.
        """
        if self.on_grid:
            grid_size = (self.dim[0] // 2, self.dim[1])
            pdf = np.ones(grid_size).T.ravel()
            if self.gaussian_indices is not None:
                pdf[self.gaussian_indices] = 0
            pdf /= pdf.sum()

            self.uniform_indices = np.random.choice(
                grid_size[0] * grid_size[1],
                self.nb_uniform,
                p=pdf,
                replace=False,
            )
            x, y = np.unravel_index(self.uniform_indices, grid_size)
            self.uniform_samples = (2 * np.pi / self.dim[0]) * np.stack(
                [x, y - self.dim[1] // 2]
            ).T
        else:
            self.uniform_samples = np.random.uniform(
                -np.pi, np.pi, (self.nb_uniform, 2)
            )

    def gaussian_sampling(self):
        """
        Generate Gaussian samples.
        """
        if self.on_grid:
            grid_size = (self.dim[0] // 2, self.dim[1])
            std_dev_x = self.dim[0] / 10
            std_dev_y = self.dim[1] / 10
            x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
            pdf_x = np.exp(-0.5 * (x**2) / std_dev_x**2)
            pdf_y = np.exp(-0.5 * ((y - grid_size[1] / 2) ** 2) / std_dev_y**2)
            pdf = pdf_x * pdf_y
            pdf /= pdf.sum()

            self.gaussian_indices = np.random.choice(
                grid_size[0] * grid_size[1],
                self.nb_gaussian,
                p=pdf.T.ravel(),
                replace=False,
            )
            x, y = np.unravel_index(self.gaussian_indices, grid_size)
            self.gaussian_samples = (2 * np.pi / self.dim[0]) * np.stack(
                [x, y - self.dim[1] // 2]
            ).T
        else:
            gaussian_samples = np.random.multivariate_normal(
                [0, 0], [[1, 0], [0, 1]], self.nb_gaussian
            )
            self.gaussian_samples = (
                np.pi * gaussian_samples / (1.5 * np.max(np.abs(gaussian_samples)))
            )

    def plot_samples(self, fig: Figure = None, ax: Axes = None):
        """
        Plot the generated samples.

        Parameters:
            fig (Figure, optional): The Matplotlib Figure object to use for the plot.
            ax (Axes, optional): The Matplotlib Axes object to use for the plot.

        If `fig` and `ax` are not provided, a new Figure and Axes will be created for the plot.
        """
        figsize = (2.5, 5)
        yticks = np.arange(-np.pi, np.pi + 0.1, np.pi / 4)
        xticks = np.arange(0, np.pi + 0.1, np.pi / 4)
        if ax == None and fig == None:
            fig, ax = plt.subplots(figsize=figsize)
        if self.gaussian_samples is not None:
            ax.scatter(
                x=np.abs(self.gaussian_samples[:, 0]),
                y=self.gaussian_samples[:, 1],
                s=2,
                color="tab:blue",
                label="Gaussian",
            )
        if self.uniform_samples is not None:
            ax.scatter(
                x=np.abs(self.uniform_samples[:, 0]),
                y=self.uniform_samples[:, 1],
                s=2,
                color="tab:orange",
                label="Uniform",
            )
        ax.set_xticks(xticks, labels=[str(x / np.pi) + r"$\pi$" for x in xticks])
        ax.set_yticks(yticks, labels=[str(x / np.pi) + r"$\pi$" for x in yticks])
        ax.legend(
            title="on-grid" if self.on_grid else "off-grid",
            handletextpad=0,
            fontsize="x-small",
        )
        ax.grid(visible=True)
        ax.set_axisbelow(True)
        fig.suptitle(
            f"Samples: L={2*self.L/np.prod(self.dim):.0%}", verticalalignment="top"
        )

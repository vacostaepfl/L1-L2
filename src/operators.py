import numpy as np
import pyxu.operator as pxo


def non_uni_FFT(
    dim: tuple, L: int | float, theta: float, on_grid: bool = False
) -> np.ndarray:
    """
    Perform a non-uniform Fast Fourier Transform (NUFFT).

    Args:
        dim (tuple): The dimensions of the input array.
        L (int | float): Number of samples to keep for the transform. If L is a float,
                         it is interpreted as a fraction of the product of dimensions.
        theta (float): A scaling factor for the number of Gaussian samples.
        on_grid (bool, optional): If True, generate samples on a grid; if False, use non-uniform samples.

    Returns:
        np.ndarray: The NUFFT result as a NumPy array.
    """
    if isinstance(L, float):
        L = round(L * np.prod(dim) / 2)

    samples = mix_sampling(dim, L, theta, on_grid)
    phi = pxo.NUFFT.type2(samples, dim, isign=-1, eps=1e-3, real=True)
    return phi


def mix_sampling(dim: tuple, L: int, theta: float, on_grid: bool) -> np.ndarray:
    """
    Generate a mixture of Gaussian and uniform samples for NUFFT.

    Args:
        dim (tuple): The dimensions of the input array.
        L (int | float): Number of samples to keep for the transform. If L is a float,
                         it is interpreted as a fraction of the product of dimensions.
        theta (float): A scaling factor for the number of Gaussian samples.
        on_grid (bool): If True, generate samples on a grid; if False, use non-uniform samples.

    Returns:
        np.ndarray: An array of mixed samples.
    """
    nb_gaussian = round(theta * L)
    nb_uniform = L - nb_gaussian

    if not on_grid:
        if nb_gaussian > 0 and nb_uniform > 0:
            gaussian_samples = gaussian_sampling(nb_gaussian)
            uniform_samples = uniform_sampling(nb_uniform)
            samples = np.concatenate([gaussian_samples, uniform_samples])
        elif nb_gaussian > 0:
            samples = gaussian_sampling(nb_gaussian)
        else:
            samples = uniform_sampling(nb_uniform)
    else:
        samples = set()
        p_x = gaussian_pdf(dim[0])
        p_y = gaussian_pdf(dim[1])
        while len(samples) < nb_gaussian:
            x = np.pi * (2 * np.random.choice(dim[0], p=p_x) / dim[0] - 1)
            y = np.pi * (2 * np.random.choice(dim[1], p=p_y) / dim[1] - 1)
            samples.add((x, y))
        while len(samples) - nb_gaussian < nb_uniform:
            x = np.pi * (2 * np.random.choice(dim[0]) / dim[0] - 1)
            y = np.pi * (2 * np.random.choice(dim[1]) / dim[1] - 1)
            samples.add((x, y))
        samples = np.array(list(samples))
    return samples


def uniform_sampling(nb_samples: int) -> np.ndarray:
    """
    Generate uniform samples in a specified range.

    Args:
        nb_samples (int): Number of uniform samples to generate.

    Returns:
        np.ndarray: An array of uniform samples in the range [-π, π].
    """
    return np.random.uniform(-np.pi, np.pi, (nb_samples, 2))


def gaussian_sampling(nb_samples: int) -> np.ndarray:
    """
    Generate Gaussian samples.

    Args:
        nb_samples (int): Number of Gaussian samples to generate.

    Returns:
        np.ndarray: An array of Gaussian samples scaled to the range [-π, π].
    """
    gaussian_samples = np.random.multivariate_normal(
        [0, 0], [[1, 0], [0, 1]], nb_samples
    )
    gaussian_samples *= np.pi / np.max(np.abs(gaussian_samples))
    return gaussian_samples


def gaussian_pdf(size: int) -> np.ndarray:
    """
    Generate a Gaussian probability density function.

    Args:
        size (int): Size of the probability density function.

    Returns:
        np.ndarray: Gaussian probability density function.
    """
    std_dev = size / 6
    pdf = np.exp(-0.5 * ((np.arange(size) - size / 2) ** 2) / (std_dev**2))
    return pdf / pdf.sum()

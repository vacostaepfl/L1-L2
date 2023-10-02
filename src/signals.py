import numpy as np
from scipy.sparse import rand


def compute_sparse(dim, values_range, density, seed=None):
    value_min, value_max = np.min(values_range), np.max(values_range)
    spikes = rand(dim[0] - 2, dim[1] - 2, density, random_state=seed).toarray()
    spikes = spikes * value_max - value_min
    spikes += value_min

    return np.pad(
        spikes,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )


def compute_smooth(dim, values_range, sigmas_range, nb_gaussian):
    if isinstance(sigmas_range, tuple):
        sigmas = np.random.uniform(*sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, list):
        sigmas = np.random.choice(sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, (float, int)):
        sigmas = sigmas * np.ones(nb_gaussian)
    else:
        ValueError("sigmas should be of type : tuple, list or int/float")

    amplitudes = np.random.uniform(*values_range, nb_gaussian)
    centers = (1 - np.max(sigmas)) * np.random.uniform(-1, 1, (nb_gaussian, 2))

    x = np.linspace(-1, 1, dim[0])
    y = np.linspace(-1, 1, dim[1])
    x, y = np.meshgrid(x, y)
    grid_points = np.vstack((x.flatten(), y.flatten())).T

    smooth = np.zeros(dim)
    for s, c, a in zip(sigmas, centers, amplitudes):
        smooth += a * np.exp(
            -np.sum((grid_points - c) ** 2, axis=1) / (2 * s**2)
        ).reshape(dim)

    return smooth


def compute_y(y0, psnr):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    noise = (
        np.random.normal(0, np.sqrt(mse / 2), (*y0.shape, 2))
        .view(np.complex128)
        .squeeze()
    )
    return y0 + noise

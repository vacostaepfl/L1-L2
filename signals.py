import numpy as np
from scipy.sparse import rand


def compute_sparse(dim, padding, density, values_range, seed=None):
    value_min, value_max = np.min(values_range), np.max(values_range)
    spikes = rand(
        dim[0] - 2 * padding, dim[1] - 2 * padding, density, random_state=seed
    ).toarray()
    spikes *= value_max - value_min
    spikes += value_min
    return np.pad(
        spikes,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )


def compute_smooth(dim, nb_gaussian, sigmas, values_range, centers_range, seed=None):
    np.random.seed(seed)
    if isinstance(sigmas, tuple):
        sigmas = np.random.uniform(*sigmas, nb_gaussian, seed=None)
    elif isinstance(sigmas, list):
        sigmas = np.random.choice(sigmas, nb_gaussian)
    elif isinstance(sigmas, (float, int)):
        sigmas = sigmas * np.ones(nb_gaussian)
    else:
        ValueError("sigmas should be of type : tuple, list or int/float")

    amplitudes = np.random.uniform(*values_range, nb_gaussian)
    centers = centers_range * np.random.uniform(-1, 1, (nb_gaussian, 2))

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


def compute_signal(dim, sparse_param, smooth_param, seed=None):
    x1 = compute_sparse(dim, **sparse_param, seed=seed)
    x2 = compute_smooth(dim, **smooth_param, seed=seed)
    return x1, x2


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

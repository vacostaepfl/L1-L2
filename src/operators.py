import numpy as np
import pyxu.operator as pxo


def non_uni_FFT(dim, L, theta):
    samples = mix_sampling(dim, L, theta)
    phi = pxo.NUFFT.type2(samples, dim, isign=-1, eps=1e-3, real=True)

    return phi


def mix_sampling(dim, L, theta):
    if L <= 1:
        L = round(L * np.prod(dim) / 2)

    nb_gaussian = round(theta * L)
    nb_uniform = L - nb_gaussian

    gaussian_samples = gaussian_sampling(nb_gaussian)
    uniform_samples = uniform_sampling(nb_uniform)

    samples = np.concatenate([gaussian_samples, uniform_samples])
    return samples


def uniform_sampling(nb_samples):
    return np.random.uniform(-np.pi, np.pi, (nb_samples, 2))


def gaussian_sampling(nb_samples):
    gaussian_samples = np.random.multivariate_normal(
        [0, 0], [[1, 0], [0, 1]], nb_samples
    )
    gaussian_samples *= np.pi / np.max(np.abs(gaussian_samples))
    return gaussian_samples

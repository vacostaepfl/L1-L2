import numpy as np
from pyxu.abc import LinOp, QuadraticFunc
from pyxu.operator import (
    SquaredL2Norm,
    L1Norm,
    hstack,
    NullFunc,
    IdentityOp,
    DiagonalOp,
    Laplacian,
)
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError
from src.operators import NuFFT


def solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    coupled: bool,
    laplacian: bool,
):
    """
    Solve the optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        coupled (bool): If True, solve a coupled optimization problem; otherwise, solve a decoupled one.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """

    lambda2 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
    lambda1 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)

    if coupled:
        return coupled_solve(y, op, lambda1, lambda2, laplacian)
    else:
        return decoupled_solve(y, op, lambda1, lambda2, laplacian)


def coupled_solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    laplacian: bool,
) -> tuple:
    """
    Solve the coupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """

    print("Coupled")

    l22_loss = (1 / 2) * SquaredL2Norm(dim=op.dim_out).asloss(y)
    F = l22_loss * hstack([op.phi, op.phi])

    if lambda2 != 0.0:
        if laplacian:
            l2operator = Laplacian((op.N, op.N), mode="wrap")
            L = lambda2 / 2 * SquaredL2Norm(l2operator.shape[0]) * l2operator
        else:
            L = lambda2 / 2 * SquaredL2Norm(op.dim_in)

        F = F + hstack([NullFunc(op.dim_in), L])
    F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

    if lambda1 == 0.0:
        G = NullFunc(2 * op.dim_in)
    else:
        G = hstack([lambda1 * L1Norm(op.dim_in), NullFunc(op.dim_in)])

    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=100) & RelError(eps=1e-4)
    pgd.fit(x0=np.zeros(2 * op.dim_in), stop_crit=sc)
    x = pgd.solution()
    x1 = x[: op.dim_in]
    x2 = x[op.dim_in :]
    return x1, x2


def decoupled_solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    laplacian: bool,
) -> tuple:
    """
    Solve the decoupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """

    print("Decoupled")

    Q_Linop, compute_x2 = Op_x2(op, lambda2, laplacian)

    l22_loss = (1 / 2) * QuadraticFunc((1, op.dim_out), Q=Q_Linop).asloss(y)
    F = l22_loss * op.phi
    F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

    if lambda1 == 0.0:
        G = NullFunc(op.dim_in)
    else:
        G = lambda1 / lambda2 * L1Norm(op.dim_in)

    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=100) & RelError(eps=1e-4)
    pgd.fit(x0=np.zeros(op.dim_in), stop_crit=sc)
    x1 = pgd.solution()
    x2 = compute_x2(x1, y)

    return x1, x2


def Op_x2(op, lambda2, laplacian):
    """
    Compute the linear operator Q_Linop and a function compute_x2 to compute x2.

    Args:
        op: NuFFT object.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.

    Returns:
        tuple: A tuple containing Q_Linop (linear operator) and compute_x2 (function to compute x2).
    """

    # Co-Gram operator = Identity ?
    random_y = np.random.rand(op.dim_out)
    vec = np.array([op.dim_in, 1e-12, *[op.dim_in / 2] * (op.dim_out - 2)])
    diag_op = DiagonalOp(vec)
    cogram_id = np.allclose(op.phi.cogram().apply(random_y), diag_op.apply(random_y))

    if cogram_id:  # Co-Gram operator = Identity
        print("Co-Gram Identity")
        if laplacian:
            B_vec = (1 / vec) * FFT_L_gram_vec(op)
            Q_Linop = DiagonalOp(B_vec / (vec + lambda2 * B_vec))
        else:
            vec = 1 / (
                np.array([op.dim_in, 1e-12, *[op.dim_in / 2] * (op.dim_out - 2)])
                + lambda2 * np.ones(op.dim_out)
            )
            Q_Linop = DiagonalOp(vec)

    else:  # Co-Gram operator ≠ Identity
        if laplacian:
            l2operator = Laplacian((op.N, op.N), mode="wrap")
            B = op.phi.cogram().dagger(damp=0) * op.phi * l2operator.gram() * op.phi.T
            Q_Linop = LinOp.from_array(
                (B * (op.phi.cogram() + lambda2 * B).dagger(damp=0)).asarray()
            )
        else:
            Q_Linop = LinOp.from_array(
                (op.phi.cogram() + lambda2 * IdentityOp(op.dim_out))
                .dagger(damp=0)
                .asarray()
            )

    def compute_x2(x1, y):
        if cogram_id:  # Co-Gram operator = Identity
            if laplacian:
                x2 = (-op.phi.T * DiagonalOp(1 / (vec + lambda2 * B_vec))).apply(
                    op(x1) - y
                )
            else:
                x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)
        else:  # Co-Gram operator ≠ Identity
            if laplacian:
                x2 = (-op.phi.T * (op.phi.cogram() + lambda2 * B).dagger(damp=0)).apply(
                    op(x1) - y
                )
            else:
                x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)

        return x2

    return Q_Linop, compute_x2


def FFT_L_gram_vec(op):
    """
    Compute the vectorized result of the linear operator corresponding to the Laplacian kernel sampled in the frequency domain.

    Args:
        op: NuFFT object.

    Returns:
        np.ndarray: Vector to create the diagonal linear operator.
    """
    Lkernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Lpad = np.pad(Lkernel, ((0, op.N - 3), (0, op.N - 3)))
    Lroll = np.roll(Lpad, -1, axis=(0, 1))
    Lfft_real = np.flip(np.fft.fftshift(np.fft.rfft2(Lroll), axes=0), axis=0)

    samples = (1 / (2 * np.pi / op.N) * op.samples).astype(int)
    samples[:, 1] = samples[:, 1] + op.N // 2 - op.even
    Lvec = (
        op.dim_in
        / 2
        * np.repeat(
            np.real(((Lfft_real[samples[:, 1], samples[:, 0]]) ** 2).ravel()), 2
        )
    )
    return Lvec

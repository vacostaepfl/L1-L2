from matplotlib.font_manager import X11FontDirectories
import numpy as np
from pyxu.abc import LinOp, QuadraticFunc
from pyxu.operator import (
    SquaredL2Norm,
    L1Norm,
    hstack,
    NullFunc,
    IdentityOp,
    DiagonalOp,
)
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError
from src.operators import NuFFT

import time


def solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    coupled: bool,
    l2operator: LinOp = None,
):
    if coupled:
        return coupled_solve(y, op, lambda1, lambda2, l2operator)
    else:
        return decoupled_solve(y, op, lambda1, lambda2, l2operator)


def coupled_solve(
    y: np.ndarray, op: NuFFT, lambda1: float, lambda2: float, l2operator: LinOp = None
) -> tuple:
    """
    Solve a coupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        l2operator (LinOp, optional): Linear operator for L2 regularization. Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """
    l22_loss = (1 / 2) * SquaredL2Norm(dim=op.dim_out).asloss(y)
    F = l22_loss * hstack([op.phi, op.phi])

    if lambda2 != 0.0:
        lambda2 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf) / 2
        if isinstance(l2operator, LinOp):
            L = lambda2 * SquaredL2Norm(l2operator.shape[0]) * l2operator
        else:
            L = lambda2 * SquaredL2Norm(op.dim_in)

        F = F + hstack([NullFunc(op.dim_in), L])
    F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

    if lambda1 == 0.0:
        G = NullFunc(2 * op.dim_in)
    else:
        lambda1 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
        G = hstack([lambda1 * L1Norm(op.dim_in), NullFunc(op.dim_in)])

    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=100) & RelError(eps=1e-4)
    pgd.fit(x0=np.zeros(2 * op.dim_in), stop_crit=sc)
    x = pgd.solution()
    x1 = x[: op.dim_in]
    x2 = x[op.dim_in :]
    return x1, x2


def decoupled_solve(
    y: np.ndarray, op: NuFFT, lambda1: float, lambda2: float, l2operator: LinOp = None
) -> tuple:
    """
    Solve a decoupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        l2operator (LinOp, optional): Linear operator for L2 regularization. Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """
    start = time.time()
    print(start)
    lambda2 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
    lambda1 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)

    # Co-Gram operator = Indentity ?
    random_y = np.random.rand(op.dim_out)
    vec = np.array([op.dim_in, 1e-12, *[op.dim_in / 2] * (op.dim_out - 2)])
    diag_op = DiagonalOp(vec)
    cogram_id = np.allclose(op.phi.cogram().apply(random_y), diag_op.apply(random_y))
    print("check id", time.time() - start)
    if cogram_id:
        print("Co-Gram Identity")
        if isinstance(l2operator, LinOp):
            B = DiagonalOp(1 / vec) * op.phi * l2operator.gram() * op.phi.T
            Q_Linop = LinOp.from_array(
                (B * (diag_op + lambda2 * B).dagger(damp=0)).asarray()
            )
        else:
            vec = 1 / (
                np.array([op.dim_in, 1e-12, *[op.dim_in / 2] * (op.dim_out - 2)])
                + lambda2 * np.ones(op.dim_out)
            )
            Q_Linop = DiagonalOp(vec)

    else:
        if isinstance(l2operator, LinOp):
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
    print("compute l2 op", time.time() - start)
    l22_loss = (1 / 2) * QuadraticFunc((1, op.dim_out), Q=Q_Linop).asloss(y)
    F = l22_loss * op.phi
    F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")
    print("compute lip", time.time() - start)

    if lambda1 == 0.0:
        G = NullFunc(op.dim_in)
    else:
        G = lambda1 / lambda2 * L1Norm(op.dim_in)
    print("compute l1", time.time() - start)

    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=100) & RelError(eps=1e-4)
    pgd.fit(x0=np.zeros(op.dim_in), stop_crit=sc)
    print("solve x1", time.time() - start)
    x1 = pgd.solution()
    if isinstance(l2operator, LinOp):
        x2 = (-op.phi.T * (diag_op + lambda2 * B).dagger(damp=0)).apply(op(x1) - y)
    else:
        x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)
    print("compute x2", time.time() - start)
    return x1, x2

import numpy as np
from pyxu.abc import LinOp
from pyxu.operator import SquaredL2Norm, L1Norm, hstack, NullFunc
from pyxu.opt.solver import PGD
from src.operators import NuFFT


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
        lambda2 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
        if isinstance(l2operator, LinOp):
            L = lambda2 * SquaredL2Norm(l2operator.shape[0]) * l2operator
        else:
            L = lambda2 * SquaredL2Norm(op.dim_in)

        F = F + hstack([NullFunc(op.dim_in), L])
    F.diff_lipschitz = F.estimate_diff_lipschitz()

    if lambda1 == 0.0:
        G = NullFunc(2 * op.dim_in)
    else:
        lambda1 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
        G = hstack([lambda1 * L1Norm(op.dim_in), NullFunc(op.dim_in)])

    pgd = PGD(f=F, g=G, verbosity=100)
    pgd.fit(x0=np.ones(2 * op.dim_in))
    x = pgd.solution()
    x1 = x[: op.dim_in]
    x2 = x[op.dim_in :]
    return x1, x2

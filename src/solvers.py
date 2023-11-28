import numpy as np
from pyxu.abc import LinOp, QuadraticFunc
from pyxu.operator import SquaredL2Norm, L1Norm, hstack, NullFunc, IdentityOp
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError
from src.operators import NuFFT


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
    lambda2 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
    lambda1 *= np.linalg.norm(op.phi.adjoint(y), ord=np.inf)

    # Co-Gram operator = Indentity ?
    cogram_id = np.allclose(
        op.phi.cogram().asarray(), op.dim_in / 2 * np.eye(op.dim_out)
    )

    if cogram_id:
        if isinstance(l2operator, LinOp):
            A = op.phi * l2operator.gram() * op.phi.T
            l22_loss = (1 / 2) * QuadraticFunc(
                (1, op.dim_out),
                Q=A * (IdentityOp(op.dim_out) + lambda2 * A).dagger(damp=0),
            ).asloss(y)
        else:
            l22_loss = (1 / 2) * SquaredL2Norm(op.dim_out).asloss(y)

    else:
        if isinstance(l2operator, LinOp):
            A = op.phi.cogram().dagger(damp=0) * op.phi * l2operator.gram() * op.phi.T
            l22_loss = (1 / 2) * QuadraticFunc(
                (1, op.dim_out),
                Q=A * (op.phi.cogram() + lambda2 * A).dagger(damp=0),
            ).asloss(y)
        else:
            l22_loss = (1 / 2) * QuadraticFunc(
                (1, op.dim_out),
                Q=(op.phi.cogram() + lambda2 * IdentityOp(op.dim_out)).dagger(damp=0),
            ).asloss(y)

    if lambda1 == 0.0:
        G = NullFunc(op.dim_in)
    elif cogram_id and not isinstance(l2operator, LinOp):
        G = lambda1 / lambda2 * (lambda2 + op.dim_in / 2) * L1Norm(op.dim_in)
    else:
        G = lambda1 / lambda2 * L1Norm(op.dim_in)

    F = l22_loss * op.phi
    F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=100) & RelError(eps=1e-4)
    pgd.fit(x0=np.zeros(op.dim_in), stop_crit=sc)
    x1 = pgd.solution()
    if isinstance(l2operator, LinOp):
        x2 = -(op.phi.gram() + lambda2 * l2operator.gram()).pinv(
            op.phi.adjoint(op.phi.apply(x1) - y), damp=0
        )
    else:
        x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)
    return x1, x2

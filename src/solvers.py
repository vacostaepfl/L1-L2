import numpy as np
from pyxu.abc import LinOp
from pyxu.operator import SquaredL2Norm, L1Norm, hstack, NullFunc, IdentityOp
from pyxu.opt.solver import PGD
from pyxu.opt.stop import AbsError, RelError


def composite_solve(
    y: np.ndarray,
    phi: np.ndarray,
    lambda1: float,
    lambda2: float,
    l2operator: LinOp = None,
) -> tuple:
    """
    Solve a composite optimization problem.

    Parameters:
    y (np.ndarray): Input data.
    phi (np.ndarray): The design matrix.
    lambda1 (float): L1 regularization parameter.
    lambda2 (float): L2 regularization parameter.
    l2operator (LinOp, optional): L2 regularization operator. Defaults to None.

    Returns:
    tuple: A tuple containing two np.ndarray elements (x1, x2).

    x1 (np.ndarray): Solution vector for the first part.
    x2 (np.ndarray): Solution vector for the second part.
    """
    l22_loss = (1 / 2) * SquaredL2Norm(dim=phi.shape[0]).asloss(y)
    F = l22_loss * hstack([phi, phi])

    if lambda2 != 0.0:
        if isinstance(l2operator, LinOp):
            L = lambda2 * SquaredL2Norm(l2operator.shape[0]) * l2operator
        else:
            L = lambda2 * SquaredL2Norm(phi.shape[1])

        F = F + hstack([NullFunc(phi.shape[1]), L])
    F.diff_lipschitz = F.estimate_diff_lipschitz()

    if lambda1 == 0.0:
        G = NullFunc(2 * phi.shape[1])
    else:
        G = hstack([lambda1 * L1Norm(phi.shape[1]), NullFunc(phi.shape[1])])

    pgd = PGD(f=F, g=G, verbosity=100)
    stopper = RelError(
        eps=1e-4
    )  # & AbsError(eps=1e-2 * phi.shape[1], var="x", f=F + G)
    pgd.fit(x0=np.ones(2 * phi.shape[1]), stop_crit=stopper)
    x = pgd.solution()
    x1 = x[: phi.shape[1]]
    x2 = x[phi.shape[1] :]
    return x1, x2


# def non_composite_solve(y, phi, lambda1, lambda2):

#     F.diff_lipschitz = F.estimate_diff_lipschitz()

#     if lambda1 == 0.0:
#         G = NullFunc(phi.shape[1])
#     else:
#         G = (lambda1 * L1Norm(phi.shape[1]),)

#     pgd = PGD(f=F, g=G, verbosity=100)
#     stopper = RelError(
#         eps=1e-4
#     )  # & AbsError(eps=1e-2 * phi.shape[1], var="x", f=F + G)
#     pgd.fit(x0=np.ones(2 * phi.shape[1]), stop_crit=stopper)
#     x1 = pgd.solution()
#     x2 = -phi.pinv((phi.apply(x1) - y), lambda2)
#     return x1, x2

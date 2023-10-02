import numpy as np
import pyxu.util as pxu
from pyxu.abc.operator import LinOp
from pyxu.operator.linop.base import NullFunc
from pyxu.operator.func.norm import SquaredL2Norm, L1Norm
from pyxu.operator.blocks import hstack
from pyxu.opt.solver import PGD


def solve(y, phi, lambda1, lambda2, l2operator=None):
    stack = hstack([phi, phi])
    l22_loss = (1 / 2) * SquaredL2Norm(dim=phi.shape[0]).asloss(pxu.view_as_real(y))
    F = l22_loss * stack

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

    pgd = PGD(f=F, g=G)
    pgd.fit(x0=np.zeros(2 * phi.shape[1]))
    x = pgd.solution()
    x1 = x[: phi.shape[1]]
    x2 = x[phi.shape[1] :]
    return x1, x2

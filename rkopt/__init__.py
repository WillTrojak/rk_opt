# -*- coding: utf-8 -*-

from math import factorial
import numpy as np

from rkopt.fourier.baseadvec import FourierAdvec
from rkopt.optimiser.neldermead import nelder_mead
from rkopt.schemes.bdf import get_bdf
from rkopt.temporal.bdf import DualBDF
from rkopt._version import __version__


def optimise(p, alpha, dt, m, s):
    FR = FourierAdvec(p=p, h=1, alpha=alpha, nk=501)
    L = FR.eigen()

    B = get_bdf(2)
    T = DualBDF(B, dt, L)

    cost = lambda x : -T.dtau_max_P(np.concatenate(([1,1], np.asarray(x))), m=m)
    P0 = [1./factorial(i) for i in range(2, s+1)]
    bnds = ((0, 1), ) + tuple((0, None) for i in range(3,s+1))

    res = nelder_mead(fun=cost, x0=np.asarray(P0), bounds=bnds, tol=1e-5, epsilon=2)

    P_opt = np.concatenate(([1,1], np.asarray(res.x)))
    print(P_opt)

    A, b = T.reconstruct_Ab(P_opt)
    dtau_max = T.dtau_max_P(P_opt, m=m)

    return dtau_max, A, b


def dtau_max(p, alpha, dt, m, P):
    FR = FourierAdvec(p=p, h=1, alpha=alpha, nk=1001)
    L = FR.eigen()

    B = get_bdf(2)
    T = DualBDF(B, dt, L)

    return T.dtau_max_P(P, m=m)

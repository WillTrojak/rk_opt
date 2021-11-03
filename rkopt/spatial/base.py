# -*- coding: utf-8 -*-

from math import pi
import numpy as np


class SpatialBase(object):
    def __init__(self, p, h, nk):
        self.p = p
        self.h = h

        self.xs, _ = np.polynomial.legendre.leggauss(p+1)
        self.xf = np.asarray([-1, 1], dtype=complex)

        self.wavenumbers(nk)

    def wavenumbers(self, nk=401, k_tol=1e-5):
        self.K = np.linspace(k_tol, 2*pi*(self.p + 1), nk, dtype=complex)

    def eigen(self):
        pass

    
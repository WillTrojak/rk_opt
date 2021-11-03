# -*- coding: utf-8 -*-

import numpy as np

from rkopt.spatial.base import SpatialBase
from rkopt.poly import jacobi, leg_mode_diff, legendreD
from rkopt.utils import organise


class SpatialAdvec(SpatialBase):
    def __init__(self, p, h, alpha, nk=401) -> None:
        super().__init__(p, h, nk)

        self.alpha = alpha

        self.Cm, self.C0, self.Cp = self._cmatrices()

    def _cmatrices(self):
        alpha = self.alpha

        Cm = np.zeros((self.p + 1, self.p + 1), dtype=complex)
        C0 = np.zeros_like(Cm, dtype=complex)
        Cp = np.zeros_like(Cm, dtype=complex)

        # Correction functions
        hl = np.zeros((self.p + 2, 1), dtype=complex)
        hl[self.p] = (0.5*(-1)**self.p)
        hl[self.p + 1] = -0.5*(-1)**self.p
        hr = np.zeros((self.p + 2, 1), dtype=complex)
        hr[self.p] = 0.5
        hr[self.p + 1] = 0.5

        gl = leg_mode_diff(hl)
        gr = leg_mode_diff(hr)

        # Interpolation
        ll = np.asarray(jacobi(self.p, 0, 0, self.xf[ 0])).reshape(1, gl.size)
        lr = np.asarray(jacobi(self.p, 0, 0, self.xf[-1])).reshape(1, gr.size)
        
        # Differentiation
        for j in range(self.p + 1):
            for i in range(self.p + 1):
                C0[j, i] = legendreD(i, j)

        Cp = (1 - alpha)*np.matmul(gr, ll)
        C0 -= alpha*np.matmul(gl, ll) + (1 - alpha)*np.matmul(gr, lr)
        Cm = alpha*np.matmul(gl, lr)

        return Cm, C0, Cp
    
    def _qmatrix(self, k):
        h = self.h
        J = 2/h
        return J*(self.Cm*np.exp(-1j*k*h) + self.Cp*np.exp( 1j*k*h) + self.C0)


    def eigen(self):
        L = np.zeros((self.K.size, self.p + 1), dtype=complex)

        for i, k in enumerate(self.K):
            G = np.linalg.eigvals(-self._qmatrix(k))
            L[i,:] = G[:]
        return organise(L)[:,0]


# -*- coding: utf-8 -*-

from math import exp
import numpy as np


from rkopt.poly import abs_root_max
from rkopt.temporal.base import DualBase


class DualBDF(DualBase):
    def __init__(self, B, dt, L):
        super().__init__(L)
        self.B = B
        self.dt = dt

        self.S = self._bdf_S()

    def dtau_max_P(self, P, m, **kwargs):
        _, _ = self.reconstruct_Ab(P)
        return self.dtau_max(m, **kwargs)

    def dtau_max(self, m, rtol=1e-6, stol=1e-10, dtau_min=1e-6):
        dtau_0 = dtau_min
        dtau_1 = 1e3*self.dt
        
        if dtau_0 >= dtau_1:
            raise ValueError('dtau_0 >= dtau_1')

        while abs(dtau_1 - dtau_0) > rtol*0.5*abs(dtau_0 + dtau_1):
            dtau_2 = 0.5*(dtau_1 + dtau_0)
            sigma = self.max_amp_factor_S(dtau_2, m)

            if sigma > 1e-13:
                dtau_1 = dtau_2
            else:
                dtau_0 = dtau_2
        self._dtau_max = 0.5*(dtau_1 + dtau_0)

        return self._dtau_max

    def max_amp_factor_S(self, dtau, m):
        P = self._bdf_P(dtau)
        
        sigma = -1
        for i, l in enumerate(self.L):
            Pm = P[i]**m
        
            c = np.zeros_like(self.B, dtype=complex)
            c[0] = 1
            c[1] = -Pm
            for j in range(1, self.B.size):
                c[j] += self.B[j]*(1 - Pm)/(1 - self.dt*l*self.B[0])

            sigma = max(sigma, abs_root_max(c) - np.abs(self.S[i]))

        return sigma

    def _bdf_P(self, dtau):
    
        (s, _) = self.A.shape
        I = np.identity(s, dtype=complex)
        O = np.ones((s, 1), dtype=complex)

        P = np.zeros_like(self.L, dtype=complex)
        for i, l in enumerate(self.L):
            c = np.matmul(np.matmul(self.b, np.linalg.inv(I - dtau*l*self.A)), O)
            P[i] = 1 + dtau*(l - 1./(self.dt*self.B[0]))*c[0, 0]
        return P

    def _bdf_S(self):
        S = np.zeros_like(self.L, dtype=complex)

        for i, l in enumerate(self.L):
            for j in range(1, self.B.size):
                S[i] += self.B[j]*np.exp(self.dt*(j - 1)*l, dtype=complex)
        
        return S
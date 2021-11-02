# -*- coding: utf-8 -*-

import numpy as np


class DualBase(object):
    def __init__(self, L):
        self._dtau_max = None
        self.dt = None
        self.L = L

    def dtau_max(self):
        pass

    def reconstruct_Ab(self, P):
        s = P.size - 1

        kappa = P[2]
        if s == 2:
            A = np.asarray([[0, 0], [kappa, 0]], dtype=complex)
        elif s == 3:
            a32 = P[3]*2/kappa
            
            A = np.asarray([[0, 0, 0], 
                            [kappa/2, 0, 0], 
                            [kappa - a32, a32, 0]], dtype=complex)
        elif s == 4:
            a43 = P[3]*3/(2*kappa)
            a32 = P[4]*3/(a43*kappa)
            A = np.asarray([[0, 0, 0, 0],
                            [kappa/3, 0, 0, 0],
                            [(2*kappa)/3 - a32, a32, 0, 0], 
                            [kappa - a43, 0, a43, 0]], dtype=complex)
        elif s == 5:
            a54 = P[3]*4/(3*kappa);
            a43 = P[4]*2/(a54*kappa);
            a32 = P[5]*4/(a43*a54*kappa);
            A = np.asarray([[0, 0, 0, 0, 0],
                            [kappa/4, 0, 0, 0, 0], 
                            [kappa/2 - a32, a32, 0, 0, 0],
                            [(3*kappa)/4 - a43, 0, a43, 0, 0],
                            [kappa - a54, 0, 0, a54, 0]], dtype=complex)
        elif s == 6:
            a65 = P[3]*5/(4*kappa)
            a54 = P[4]*5/(3*a65*kappa)
            a43 = P[5]*5/(2*a54*a65*kappa)
            a32 = P[6]*5/(a43*a54*a65*kappa)
            
            A = np.asarray([[0, 0, 0, 0, 0, 0],
                            [kappa/5, 0, 0, 0, 0, 0],
                            [(2*kappa)/5 - a32, a32, 0, 0, 0, 0], 
                            [(3*kappa)/5 - a43, 0, a43, 0, 0, 0], 
                            [(4*kappa)/5 - a54, 0, 0, a54, 0, 0],
                            [kappa - a65, 0, 0, 0, a65, 0]], dtype=complex)

        b = np.zeros((1, P.size - 1), dtype=complex)
        b[0,-1] = 1
        self.A = A
        self.b = b

        return A, b

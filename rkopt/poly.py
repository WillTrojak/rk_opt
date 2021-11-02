# -*- coding: utf-8 -*-

import numpy as np


def abs_root_max(p):
    C = companion_matrix(p)
    return np.amax(np.absolute(np.linalg.eigvals(C)))


def companion_matrix(p):
    n = p.size - 1
    C = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        if i>=1:
            C[i,i-1] = 1
        C[i,-1] = -p[n-i]
    return C


def legendreD(n, j):

    if (n % 2) == (j % 2):
        q = 0
    elif n < j:
        q = 0
    else:
        q = 2*j + 1

    return q


def leg_mode_diff(p):
    k = p.size
    q = np.zeros((k-1, 1), dtype=complex)

    for i in range(k):
        for j in range(k - 1):
            q[j] += legendreD(i, j)*p[i]
    return q

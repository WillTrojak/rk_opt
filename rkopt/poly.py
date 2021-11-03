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


def jacobi(n, a, b, z):
    j = [1]

    if n >= 1:
        j.append(((a + b + 2)*z + a - b) / 2)
    if n >= 2:
        apb, bbmaa = a + b, b*b - a*a

        for q in range(2, n + 1):
            qapbpq, apbp2q = q*(apb + q), apb + 2*q
            apbp2qm1, apbp2qm2 = apbp2q - 1, apbp2q - 2

            aq = apbp2q*apbp2qm1/(2*qapbpq)
            bq = apbp2qm1*bbmaa/(2*qapbpq*apbp2qm2)
            cq = apbp2q*(a + q - 1)*(b + q - 1)/(qapbpq*apbp2qm2)

            # Update
            j.append((aq*z - bq)*j[-1] - cq*j[-2])

    return j


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

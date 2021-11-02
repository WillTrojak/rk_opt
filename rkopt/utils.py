# -*- coding: utf-8 -*-

import itertools
import numpy as np


def organise(A):

    (m, n) = A.shape
    
    a = A[0,:]
    b = A[1,:]
    c = closest(a, b)
    b = c

    O = np.empty([m, n], dtype=complex)
    O[0,:] = a
    O[1,:] = b
    
    for i in range(2, m):
        y3 = 2*O[i-1,:] - O[i-2,:]
        O[i,:] = closest(y3, A[i,:])

    return O


def closest(a, b):

    p = perms(b)
    (mp, _) = p.shape

    for i in range(mp):
        diff = np.sum(np.log(np.absolute(p[i,:] - a)))
        if i == 0:
            diff_min = diff
            idx = i
        else:
            if diff < diff_min:
                diff_min = diff
                idx = i

    return p[idx,:]


def perms(x):
    permut = itertools.permutations(x)
    permut_array = np.empty((0, x.size))
    for p in permut:
        permut_array = np.append(permut_array, np.atleast_2d(p), axis=0)

    return permut_array
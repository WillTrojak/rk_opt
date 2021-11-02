# -*- coding: utf-8 -*-

from functools import cache
import numpy as np
from operator import itemgetter, attrgetter


from rkopt.optimiser.base import OptimiseResult


def nelder_mead(fun, x0, bounds=None, tol=1e-6, max_iter=100, 
                alpha=1., beta=2., gamma=0.5, delta=0.5, epsilon=2.):
    if alpha <= 0:
        raise ValueError('Nelder-Mead: alpha <= 0') 
    if beta < 1:
        raise ValueError('Nelder-Mead: beta < 1') 
    if not (0 < gamma < 1):
        raise ValueError('Nelder-Mead: (0 < gamma < 1)') 
    if not (0 < delta < 1):
        raise ValueError('Nelder-Mead: not (0 < delta < 1)')
    if epsilon < 0:
        raise ValueError('Nelder-Mead: epsilon negative')
    
    costs = [[x0, fun(x0)]]

    n_pts = x0.size + 2

    # Initial simplex setup
    x_init = _init_simplex(x0, n_pts-len(costs), epsilon, bounds)
    for i in range(n_pts-len(costs)):
        x = x_init[i,:]
        costs.append([x, fun(x)])

    for i in range(max_iter):
        # Sort 
        costs.sort(key=itemgetter(1))

        # Check for convergence
        if np.linalg.norm(costs[0][0] - costs[1][0], ord=2) < tol:
            break

        xo = _centroid(costs)

        # Reflection
        xr = _reflection(xo, costs[-1][0], alpha, bounds)
        cr = fun(xr)
        if costs[0][1] <= cr < costs[-1][1]:
            costs.pop(-1)
            costs.append([xr, cr])
            continue
        # Expansion
        elif cr < costs[0][1]:
            xe = _expansion(xo, xr, beta, bounds)
            ce = fun(xe)
            if ce < cr:
                costs.pop(-1)
                costs.append([xe, ce])
                continue
            else:
                costs.pop(-1)
                costs.append([xr, cr])
        
        # Contraction
        xc = _contraction(xo, costs[-1][0], gamma, bounds)
        cc = fun(xc)
        if cc < costs[-1][1]:
            costs.pop(-1)
            costs.append([xc, cc])
            continue
        # Shrink
        else:
            x0 = costs[0][0]
            for i, c in enumerate(costs[1:]):
                xs = _shrink(x0, c[0], delta, bounds)
                costs[i+1][0] = xs
                costs[i+1][1] = fun(xs)
            continue
        
    return OptimiseResult(costs[0][0], costs[0][1], i, bounds)


def _centroid(costs):
    xc = 0*costs[0][0]
    for v in costs[:-1]:
        xc += v[0]
    return xc/(len(costs) - 1)


def _contraction(xo, xn, gamma, bounds):
    return _restrict(xo + gamma*(xn - xo), bounds)


def _expansion(xo, xr, beta, bounds):
    return _restrict(xo + beta*(xr - xo), bounds)


def _init_simplex(x0, n, epsilon, bounds):
    xi = np.empty((n, x0.size))
    x_min = np.amin(x0)
    x_width = np.amax(x0) - np.amin(x0)

    i = 0
    while i < n:
        x_c = np.random.rand(x0.size)*x_width + x_min
        d = np.linalg.norm(x_c - x0, ord=2)
        if d <= epsilon and _in_bounds(x_c, bounds):
            xi[i,:] = x_c
            i += 1
    return xi


def _in_bounds(x, bounds):
    if bounds is None:
        return True
    else:
        for i, b in enumerate(bounds):
            if b[0] is not None and x[i] < b[0]:
                return False
            elif b[1] is not None and x[i] > b[1]:
                return False
        return True


def _restrict(x, bounds):
    if bounds is None:
        return x
    else:
        for i, b in enumerate(bounds):
            if b[0] is not None and x[i] < b[0]:
                x[i] = b[0]
            elif b[1] is not None and x[i] > b[1]:
                x[i] = b[1]
        return x


def _reflection(xo, xn, alpha, bounds):
    return _restrict(xo + alpha*(xo - xn), bounds)
        

def _shrink(x0, xi, delta, bounds):
    return _restrict(x0 + delta*(xi - x0), bounds)

# -*- coding: utf-8 -*-

class OptimiseResult(object):
    def __init__(self, x, c, i, bounds=None):
        self.x = x
        self.fun = c
        self.nit = i
        self.bounds = bounds

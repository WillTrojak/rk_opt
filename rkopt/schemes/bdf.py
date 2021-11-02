# -*- coding: utf-8 -*-

import numpy as np


def get_bdf(s):
   BDF = {1: [1, -1], 
          2: [2./3, -4./3, 1./3],
          3: [6./11, -18./11, 9./11, -2./11],
         }

   return np.asarray(BDF[s])


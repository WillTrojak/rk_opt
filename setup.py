#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys


# Python version
if sys.version_info[:2] < (3, 3):
    print('RKOPT requires Python 3.3 or newer')
    sys.exit(-1)

# GiMMiK version
vfile = open('rkopt/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print('Unable to find a version string in rkopt/_version.py')

# Modules
modules = [
    'rkopt.fourier',
    'rk.opt.optimiser',
    'rkopt.scheme',
    'rkopt.temporal',
]

# Data
package_data = {
}

# Hard dependencies
install_requires = [
    'pyfr >= 1.10',
    'numpy >= 1.7'
]

# Info
classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering'
]

# Long Description
long_description = '''Stuff.'''

setup(name='rkopt',
      version=version,

      # Packages
      packages=['rkopt'],
      package_data=package_data,
      install_requires=install_requires,

      # Metadata
      description='FR RK optimiser',
      long_description=long_description,
      maintainer='Will Trojak',
      maintainer_email='wtrojak@gmail.com',
      license='BSD',
      keywords=['FR', 'RK'],
      classifiers=classifiers)

#!/usr/bin/env python3
'''
setup.py

Set up c extensions

'''

from distutils.core import setup, Extension
import numpy as np

module1 = Extension('kinetictools',
                    include_dirs = [np.get_include()],
                    sources = ['pykinetictools.c', 'computeKinetics.c'])

module2 = Extension('spikingtools',
                    include_dirs = [np.get_include()],
                    sources = ['pyspikingtools.c', 'computeSpiking.c'])


setup(name='kinetictools',
        version = '1.0',
        description = "This module computes kinetics block output given the input to the kinetics block, which is the output of the nonlinearity of LNK",
        ext_modules = [module1]
        )

setup(name='spikingtools',
        version = '1.0',
        description = "This module computes spiking block output given the input to the spiking block, which is the output of LNK",
        ext_modules = [module2]
        )



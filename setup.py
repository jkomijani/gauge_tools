# Created by Javad Komijani, University of Tehran, 24/Apr/2020.
# Copyright (C) 2020 Javad Komijani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License <https://www.gnu.org/licenses/>
# for more deteils.


from setuptools import setup,find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy
import sys

USE_CYTHON = False # if False compiles from .c files

def readme():
    with open('README.rst') as f:
        return f.read()

ext = '.pyx' if USE_CYTHON else '.c'

#ext_options = {'compiler_directives': {'profile': True, 'language_level': sys.version_info[0]}, 'annotate': True}
ext_options = {'compiler_directives': {'language_level': sys.version_info[0]}}

ext_modules = [Extension('gauge_tools._gtcore',     ['gauge_tools/_gtcore'    + ext], include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.lib._site',   ['gauge_tools/lib/_site'  + ext], include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.lib._matrix', ['gauge_tools/lib/_matrix'+ ext], include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.lib._rand',   ['gauge_tools/lib/_rand'+ext,"gauge_tools/lib/mt19937ar.c"],include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.lib._rand_matrix',['gauge_tools/lib/_rand_matrix' + ext],include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.util.gaugefix',['gauge_tools/util/gaugefix' + ext], include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.util.smear',  ['gauge_tools/util/smear' + ext],     include_dirs=[numpy.get_include()]),
               Extension('gauge_tools.util.quark',  ['gauge_tools/util/quark' + ext],     include_dirs=[numpy.get_include()])]

packages     = ['gauge_tools', 'gauge_tools.lib', 'gauge_tools.util']
package_dir  = {'gauge_tools':'gauge_tools', 'gauge_tools.lib':'gauge_tools/lib', 'gauge_tools.util':'gauge_tools/util'}
package_data = {}
 
setup(name       =  'gauge_tools',
    version      =  '0.0.0',
    description  =  'Tools for Monte Carlo simulations of gauge fields.',
    packages     =  packages,
    package_dir  =  package_dir,
    package_data =  package_data,
    cmdclass     =  {'build_ext': build_ext},
    ext_modules  =  cythonize(ext_modules, **ext_options),
    url          =  'http://github.com/jkomijani/gauge_tools',
    author       =  'Javad Komijani',
    author_email =  'jkomijani@gmail.com',
    license      =  'JK',
    install_requires=['cython>=0.29.17', 'numpy>=1.8', 'gvar>=9.2'] if USE_CYTHON else ['numpy>=1.8', 'gvar>=9.2'],
    zip_safe     =  False,
)

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


"""
.. _mt19937ar: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html 
.. _randommodule: https://github.com/python/cpython/blob/master/Modules/_randommodule.c

This module is developed using the Mersenne Twister method with improved initialization downloaded from `mt19937ar`_,
which by the way is used in the `random` package of python as described in `randommodule`_.

The main functions in this module are:
    - :meth:`gauge_tools.lib._rand.seed`
    - :meth:`gauge_tools.lib._rand.rand`
    - :meth:`gauge_tools.lib._rand.uniform`
    - :meth:`gauge_tools.lib._rand.randrange`
    - :meth:`gauge_tools.lib._rand.normal`
    - :meth:`gauge_tools.lib._rand.complex_normal`.
"""

from libc.math cimport log, sqrt

from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "mt19937ar.h":
    unsigned long genrand_int32() 	# generates a random number on [0,0xffffffff]-interval
    void init_genrand(unsigned long s)
    void init_by_array(unsigned long init_key[], int key_length)

#==========================================================
def seed(seed='time'):
    """ To set the seed for the random number generator.

    Parameters:
        - ``seed``: (int or "time"): the default value is `seed="time"`,\
                    which sets the seed to the current time as a random number.
    """
    if isinstance(seed,str) and seed=='time':
        import time; seed = int(time.time())
    elif not isinstance(seed,int):
        raise Exception("seed number must be an integer or 'time' if specified")
    seed = abs(seed)
    # we could simply use:
    #    >>> init_genrand(<unsigned long> (seed%4294967296))
    # But, following Python.random which uses CPython._random module 
    # (see https://github.com/python/cpython/blob/master/Modules/_randommodule.c)
    # we use the below commands:
    key_length = (seed.bit_length() -1)//32 +1
    cdef unsigned long *key
    key = <unsigned long*>PyMem_Malloc(key_length * sizeof(unsigned long))
    if key==NULL:
        raise MemoryError()
    for i in range(key_length):
        key[i] = seed%4294967296    # 4294967296 = 2**32
        seed = seed//4294967296
    init_by_array(key, key_length)
    PyMem_Free(key)
    
seed('time') # automatically sets the seed to the "current" time

#==========================================================
cpdef double rand():
    """ Generates a (double) random number on `[0,1)` or `[0,1]` with 53-bit resolution.
    """
    # Notes:
    #   1)  Adopted from `genrand_res53(void)` in `mt19937ar.c`.
    #   2)  Note that 9007199254740992 = 2**53 and 67108864 = 2**26 .
    #   3)  As in `mt19937ar.c` and also the random package of Python
    #       [see https://github.com/python/cpython/blob/master/Modules/_randommodule.c]
    #       to map the random number to `[0,1)` we use a factor of (1.0/9007199254740992.0).
    #   4)  The factor (1.0/9007199254740992.0) would theoretically yield a random number in [0,1).
    #   5)  To obtain a random variable in [0,1] we should multiply it by (1.0/9007199254740991.0).
    #   6)  Note that with machine precision both 1/2**53 and 1/(2**53-1) are equal to 1.1102230246251565e-16
    #       Therefore in practice it does not matter which of these two factor is used in this function.
    cdef unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6 
    return (a*67108864.0+b)*(1.0/9007199254740992.0)

cpdef double uniform(double x, double y):
    """ Generates a (double) random number on `[x,y)` or `[x,y]` with 53-bit resolution,
    where `x` an `y` are the arguments of the function.
    Keep in mind that it would be marginally faster to use `uniform(-1.,1.)` than `uniform(-1,1)`.
    """
    # see `rand()` for comments.
    cdef unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6 
    return x + (y-x)*(a*67108864.0+b)*(1.0/9007199254740992.0)

cpdef int randrange(int n):
    """ Return an integer uniformly chosen from range(n). """
    return genrand_int32()%n

#==========================================================
cpdef double normal():
    """ .. _MILC code: http://physics.utah.edu/~detar/milc.html

    Normal distribution `N(0,1)` using the Marsaglia polar method.
    See also `gaussrand.c` in the `MILC code`_.
    """
    cdef double x, y, rsq=1.0, fac
    while(rsq >= 1.0):
        x = uniform(-1.,1.)
        y = uniform(-1.,1.)
        rsq = x*x + y*y
    fac = sqrt( (-2.0*log(rsq)) / rsq)
    return x * fac

cpdef double complex complex_normal():
    """ .. _MILC code: http://physics.utah.edu/~detar/milc.html

    A complex number with real and imaginary parts that are independent random variables with
    normal distribution `N(0,1/sqrt(2))` so that the expectation value of the power of the complex number is 1.
    This function uses the Marsaglia polar method. See also `gaussrand.c` in the `MILC code`_.
    """
    cdef double x, y, rsq=1.0, fac
    while(rsq >= 1.0):
        x = uniform(-1.,1.)
        y = uniform(-1.,1.)
        rsq = x*x + y*y
    fac = sqrt( (-log(rsq)) / rsq)
    return (x*fac + y*fac*1J)

#==========================================================
def _benchmark(int N=100000, int N_repeat=100):
    """ A short script to compare the speed of `rand` and `uniform`
    defined here compered to the `rand` and `uniform` defined in 
    the random package of python. """
    import time
    import random
    import numpy as np
    seed_number = int(time.time())
    randomrand = random.random
    randomuniform = random.uniform
    cdef Py_ssize_t i,j
    time_4_genrand = []
    time_4_rand = []
    time_4_randomrand = []
    time_4_uniform = []
    time_4_randomuniform = []
    for i in range(N_repeat):
        T1 = time.time()
        for j in range(N):
            genrand_int32()
        time_4_genrand.append((time.time()-T1)/N)
    seed(seed_number)
    random.seed(seed_number)
    for i in range(N_repeat):
        T1 = time.time()
        for j in range(N):
            rand()
        time_4_rand.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            randomrand()
        time_4_randomrand.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            uniform(-3.,3.)
        time_4_uniform.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            randomuniform(-3.,3.)
        time_4_randomuniform.append((time.time()-T1)/N)
    x = time_4_genrand
    print("Test of _rand.genrand():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_4_rand
    print("Test of _rand.rand():\t\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9),end='')
    print("[new value: {}]".format(rand()))
    x = time_4_randomrand
    print("Test of random.random():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9),end='')
    print("[new value: {}]".format(randomrand()))
    x = time_4_uniform
    print("Test of _rand.uniform():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9),end='')
    print("[new value: {}]".format(uniform(-3.,3.)))
    x = time_4_randomuniform
    print("Test of random.uniform():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9),end='')
    print("[new value: {}]".format(randomuniform(-3.,3.)))

#==========================================================

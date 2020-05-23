# Created by Javad Komijani, University of Tehran,  24/Apr/2020.
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

#==========================================================
from ._matrix import  asmatrix
from ._matrix cimport matrix
from ._rand   cimport uniform
from ._rand   import  seed # seed should be set in the main module loading this module

import numpy as np
import itertools
import copy

#==========================================================
class random_special_unitary(object):
    """ Class of ``SU(n)`` random matrices.
    Create an instance by::

            sun = gauge_tools.lib._rand_matrix.random_special_unitary(n) .

    Then calling the method `random_clost_to_unity()` one can generate random ``SU(n)`` matrices close to unity.
    If needed one can multiply several such matrices to get a random matrix not necessarily close to unity.
    The implementation of the matrices is by default with numpy arrays for any values of ``n``.
    In case ``n=3``, one can  

            sun = gauge_tools.lib._rand_matrix.random_special_unitary(n, ndarray=False)

    for implimentation by :class:``gauge_tools.lib._matrix.matrix`` matrices as used in the gauge_tools package.

    """
    ### the Gell-Mann matrices as generators of SU(3)
    GM_lambda = [None]*8
    GM_lambda[0] = np.array([[0,1,0],  [1,0,0],  [0,0,0.]])
    GM_lambda[1] = np.array([[0,-1,0], [1,0,0],  [0,0,0.]])*1j
    GM_lambda[2] = np.array([[1,0,0],  [0,-1,0], [0,0,0.]])
    GM_lambda[3] = np.array([[0,0,1],  [0,0,0],  [1,0,0.]])
    GM_lambda[4] = np.array([[0,0,-1], [0,0,0],  [1,0,0.]])*1j
    GM_lambda[5] = np.array([[0,0,0],  [0,0,1],  [0,1,0.]])
    GM_lambda[6] = np.array([[0,0,0],  [0,0,-1], [0,1,0.]])*1j
    GM_lambda[7] = np.array([[1.,0,0], [0,1,0],  [0,0,-2]])/3**0.5
    ### the Pauli sigma matrices as generators of SU(2)
    P_sigma = [None]*3
    P_sigma[0] = np.array([[0,1],  [1,0.]])
    P_sigma[1] = np.array([[0,-1], [1,0.]])*1j
    P_sigma[2] = np.array([[1.,0], [0,-1]])
    #===================================
    def __init__(self,n=3,ndarray=True):
        """
        Parameters:
            - ``n``       (int):   initiates a ``SU(n)`` matrix.
            - ``ndarray`` (bool):  uses numpy arrays if `True` else uses the matrix type.
        """
        self.n = n       
        if not (isinstance(n,int) and n>1):
            raise Exception("Oops: `n` must be an integer larger than 1.")
        elif n==2:
            self.GM_lambda = self.P_sigma
        elif n>3:
            # We define "generalized" Gell-Mann matrices
            self.GM_lambda = [np.zeros((n,n), dtype=np.complex_) for _ in range(n**2-1)]
            for i, j in itertools.product(range(n),range(n)):
                if i==j and i<(n-1):
                    ind = i+n*i
                    self.GM_lambda[ind][i,i]  = 1
                    self.GM_lambda[ind][i+1,i+1] = -1
                elif i<j:
                    ind1,ind2 = n*i+j, i+n*j
                    self.GM_lambda[ind1][i,j] = 1
                    self.GM_lambda[ind1][j,i] = 1
                    self.GM_lambda[ind2][i,j] = 1j
                    self.GM_lambda[ind2][j,i] = -1j
        if ndarray==False:
            for i in range(len(self.GM_lambda)):
                self.GM_lambda[i] = asmatrix(self.GM_lambda[i])
            self.I = matrix(identity=True)
        else:
            self.I = np.identity(n)
    #===================================
    def gen_samples(self, eps, n_samples=100):
        """ returns a list of `n_samples` of random matrices close to unity.

        Parameters:
            - ``eps``       (float):  a number indicating how much close the random matrix is to unity.
            - ``n_samples`` (int):    number of samples to be generated.
        """
        self.n_samples = n_samples
        samples = []
        for k in range(int(n_samples/2)):
            M = asmatrix(self.random_close_to_unity(eps))
            samples.append(M)
            samples.append(M.H)
        return samples
    #===================================
    def random_close_to_unity(self, eps):
        """ generates a SU(n) random matrix close to unity. """
        H = self.random_hermitian()
        return self._unitarize(self.I + 1j*eps*H)
    #===================================
    def random_hermitian(self):
        """ generate a random traceless hermition matrix with the (generalized) Gell-Mann matrices
        and coefficients uniformly chosen between -1 and 1. 
        """
        n = self.n**2-1
        theta = [uniform(-1.,1.) for _ in range(n)]  # tuple of random numbers -1 <= theta[i] <= +1
        H = theta[0] * self.GM_lambda[0]
        for i in range(1,n):
            H = H + theta[i] * self.GM_lambda[i]
        return H
    #===================================
    @staticmethod
    def _unitarize(M): # more precisely, this is special unitarization as det(result)=1
        # Note that np.vdot(.,.) and np.dot(.,.) are different ....
        if isinstance(M,matrix):
            return M.naive_unitarize()
        n = len(M)
        S = copy.copy(M)
        orth = lambda r1,r2: r1 - r2 * np.vdot(r2, r1)
        for i in range(n):
            for j in range(i):
                S[i] = orth(S[i], S[j])
            S[i] /= np.vdot(S[i],S[i])**0.5
        S /= np.linalg.det(S)**(1./n)
        return S
    #===================================

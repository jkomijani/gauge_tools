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

"""
This module uses the extension types of Cython to define ``3x3``, ``3x1``, and ``1x3`` complex matrices:
    - :class:`gauge_tools.lib._matrix.matrix`:   the extension type of ``3x3`` matrices:
        - ``matrix()``:             returns an empty matrix.
        - ``matrix(zeros=True)``:   returns a zero matrix.
        - ``matrix(identity=True)``: returns an identity matrix.
    - :class:`gauge_tools.lib._matrix.vector`:   the extension type of ``3x1`` matrices or (column) vectors:
        - ``vector()``:             returns an empty column vector.
        - ``vector(zeros=True)``:   returns a zero column vector.
    - :class:`gauge_tools.lib._matrix.rowvector`:  the extension type of ``1x3`` matrices or row vectors:
        - ``rowvector()``:             returns an empty row vector.
        - ``rowvector(zeros=True)``:   returns a zero row vector.

Operators ``+``, ``-``, ``*``, ``+=``, ``-=``, and ``*=`` are supported for objects of the same type.
The operator ``/`` is supported only for division by numbers.
The operator ``*`` is also supported between instances of different classes and numbers in the framework of linear algebra.
For instance, if ``m1`` and ``m2`` are two matrices and ``v`` is a vector, one can use::

    v += (2*m1 - m2) * v

to change the value of ``v``.
"""

#==========================================================
# c\y\thon: boundscheck=False, wraparound=False, nonecheck=False
# Be always careful with cdivision=True because it gives rise to negaive values when using %

#==========================================================
cimport cython
cimport numpy
import numpy as np

from libc.math cimport exp,sin,cos
# from numbers import Number # for type checking, but it was slow, so it is discarded now.

# ctypedef double complex complex_type

#==========================================================
def identity(int n):
    """ returns a ``nxn`` identity matrix, where ``n`` is the argument of the function. """
    return asmatrix(np.identity(n,dtype=np.complex_))

def zeros(int nr, int nc):
    """ returns a ``nr x nc`` zero matrix, where ``nr`` and ``nc`` are the arguments of the function. """
    return asmatrix(np.zeros(nr,nc,dtype=np.complex_))

def ones(int nr, int nc):
    """ returns a ``nr x nc`` one matrix, where ``nr`` and ``nc`` are the arguments of the function. """
    return asmatrix(np.ones(nr,nc,dtype=np.complex_))

#==========================================================
def asmatrix(x):
    """ converts an array-like object (such as a list, np.ndarray, and np.matrix type objects) to a matrix. """
    if isinstance(x, matrix):
        return x
    else:
        x = np.array(x)

    if x.shape == (3,3):
        mat = matrix()
        mat.insert(x)
        return mat
    if x.shape == (1,3):
        vec = rowvector()
        vec.insert(x)
        return vec
    if x.shape == (3,1):
        vec = vector()
        vec.insert(x)
        return vec
    else:
        return np.asmatrix(x)
        
#==========================================================
cdef class matrix:
    """ This is a Cython extension type to declare and define ``3x3`` complex matrices.
    After importing ``matrix`` from :mod:`gauge_tools.lib._matrix`, one can use::

            matrix()
            matrix(zeros=True)
            matrix(identity=True)

    to define an empty, zero, and identity matrices, respectively.

    This extension type is equipped with several special methods such as:
        - ``__add__(matrix x, matrix y)`` supports instances of matrix.

        - ``__sub__(matrix x, matrix y)`` supports instances of matrix.

        - ``__mul__(x, y)``   supports instances of matrix and numbers.

        - ``__truediv__(matrix x, y)``  supports numbers.

        - ``__iadd__(self, matrix y)``  supports instances of matrix.

        - ``__isub__(self, matrix y)``  supports instances of matrix.

        - ``__imul__(self, y)`` supports instances of matrix and numbers.

        - ``__itruediv__(self, y)`` supproted for numbers.

        - ``__getitem__(self, ij)`` to get item(s).\
            For a matrix `m`
                - to get the `(i,j)` element, write `m[i,j]`.
                - to get the `i` row , write `m[i]` or `m[i,:]`; this returns a row vector type object.
                - to get the `j` column , write `m[:,j]`; this returns a (column) vector type object.
                - to get a copy of the whole matrix, write `m[:,:]`.

            In contrary to the numpy convention, only positive indices are supported.

        - ``__setitem__(self, ij, y)`` to set item(s).\
            For a matrix `m`
                - to set the `(i,j)` element, write `m[i,j] = s`.
                - to set the `i` row , write `m[i] = v` or `m[i,:] = v`, where `v` is a row vector type object.
                - to set the `j` column , write `m[:,j] = v`, where `v` is a (column) vector type object.

            In contrary to the numpy convention, only positive indices are supported.

    In addition to these special methods, there are more methods and properties listed below.

    """
    # cdef readonly complex_type e[3][3]

    def __init__(self, zeros=False, identity=False):
        """ Initiates a ``3x3`` complex matrix.
        The matrix is by default an empty matrix unless one changes the default values of options.

        Parameters:
            - ``zeros``     (bool): if `True` returns a zero matrix. The default value is `False`.
            - ``identity``  (bool): if `True` returna an identity matrix. This option overwrites the `zeros` option.\
                                    The default value is `False`.
        """
        cdef int i,j
        if zeros==True or identity==True:
            for i in range(3):
                for j in range(3):
                    if identity and i==j:
                        self.e[i][j] = 1
                    else:
                        self.e[i][j] = 0

    def __pos__(self):
        return self

    def __neg__(self):
        return self*(-1)

    def __add__(matrix x, matrix y):
        """ supports instances of matrix. """
        cdef matrix mat  = matrix.__new__(matrix)
        cdef matrix mat1 = x
        cdef matrix mat2 = y
        cdef int i,j
        for i in range(3):
            for j in range(3):
                mat.e[i][j] = mat1.e[i][j] + mat2.e[i][j]
        return mat

    def __sub__(matrix x, matrix y):
        """ supports instances of matrix. """
        cdef matrix mat  = matrix.__new__(matrix)
        cdef matrix mat1 = x
        cdef matrix mat2 = y
        cdef int i,j
        for i in range(3):
            for j in range(3):
                mat.e[i][j] = mat1.e[i][j] - mat2.e[i][j]
        return mat

    def __mul__(x, y):
        """ supports instances of matrix and numbers. """
        # A comment on multiplication is in order. A test with `%timeit` indicates that
        # `matrix * number`  is faster than `number * matrix`
        # `matrix * complex_number` is faster than `matrix * float` or `matrix * int` because (it seems) it does not need a type conversion
        ### We check the type of x and y below because they can have different types.
        cdef matrix mat = matrix.__new__(matrix)
        cdef matrix mat1
        cdef matrix mat2
        cdef complex_type const
        cdef int i,j
        if isinstance(x, matrix) and isinstance(y, matrix):
            mat1 = x
            mat2 = y
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = mat1.e[i][0]*mat2.e[0][j] + mat1.e[i][1]*mat2.e[1][j] + mat1.e[i][2]*mat2.e[2][j]
            return mat
        elif isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            # y is a number and x is a matrix. We could use `elif isinstance(y, Number)`, but it would be slower.
            mat1  = x
            const = y
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = mat1.e[i][j] * const
            return mat
        elif isinstance(x, float) or isinstance(x, int) or isinstance(x, complex):
            # x is a number and y is a matrix. We could use `elif isinstance(x, Number)`, but it would be slower.
            const = x
            mat2  = y
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = const * mat2.e[i][j]
            return mat
        else:
            return NotImplemented
        
    def __truediv__(matrix x, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it can have different types.
        cdef matrix mat  = matrix.__new__(matrix)
        cdef matrix mat1 = x
        cdef complex_type const
        cdef int i,j
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            # y is a number and x is a matrix. We could use `elif isinstance(y, Number)`, but it would be slower.
            const = 1.0/y
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = mat1.e[i][j] * const
            return mat
        else:
            raise Exception('Bad Types')

    def __iadd__(self, matrix y):
        """ supports instances of matrix. """
        cdef matrix mat2 = y
        cdef int i,j
        for i in range(3):
            for j in range(3):
                self.e[i][j] += mat2.e[i][j]
        return self

    def __isub__(self, matrix y):
        """ supports instances of matrix. """
        cdef matrix mat2 = y
        cdef int i,j
        for i in range(3):
            for j in range(3):
                self.e[i][j] -= mat2.e[i][j]
        return self

    def __imul__(self, y):
        """ supports instances of matrix and numbers. """
        ### We check the type of y below because it accepts different types.
        cdef matrix mat = matrix.__new__(matrix)
        cdef matrix mat2
        cdef complex_type const
        cdef int i,j
        if isinstance(y, matrix):
            mat2 = y
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = self.e[i][0]*mat2.e[0][j] + self.e[i][1]*mat2.e[1][j] + self.e[i][2]*mat2.e[2][j]
            return mat
        elif isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = y
            for i in range(3):
                for j in range(3):
                    self.e[i][j] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __itruediv__(self, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it accepts different types.
        cdef complex_type const
        cdef int i,j
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = 1.0/y
            for i in range(3):
                for j in range(3):
                    self.e[i][j] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __getitem__(self, ij):
        """ to get item(s).
        For a matrix `m`
            - to get the `(i,j)`th element, write `m[i,j]`.
            - to get the `i`th row , write `m[i]` or `m[i,:]`; this returns a row vector type object.
            - to get the `j`th column , write `m[:,j]`; this returns a (column) vector type object.
            - to get a copy of the whole matrix, write `m[:,:]`.
        In contrary to the numpy convention, only positive indices are supported.
        
        Note that similar to numpy array, one can use `:` (or `...`) to get a row/column.
        One should also note that any python objects except integers can play the role of `:` or `...`
        except for boolean objects `True` and `False`, which act like 1 and 0.
        This issue can be fixed by adding more `if`s, but it is not necessary.
        """
        cdef int i=0,j=0,k
        ROW = False # `False` means the row of the requested slice is NOT fixed
        COL = False # `False` means the column...
        if isinstance(ij,int) and ij in range(3):
            i = ij
            ROW = True
        elif isinstance(ij, tuple) and len(ij)==2:
            if isinstance(ij[0],int):
                if ij[0] in range(3):
                    i = ij[0]
                    ROW = True
                else:
                    raise Exception('Bad indices')
            if isinstance(ij[1],int):
                if ij[1] in range(3):
                    j = ij[1]
                    COL = True
                else:
                    raise Exception('Bad indices')
        else:
            raise Exception('Bad indices')

        if ROW and COL:
            return self.e[i][j]
        elif ROW:
            v = rowvector()
            for k in range(3):
                v[k] = self.e[i][k]
            return v 
        elif COL:
            v = vector()
            for k in range(3):
                v[k] = self.e[k][j]
            return v 
        else:
            mat = matrix()
            mat.insert(self)
            return mat

    def __setitem__(self, ij, y):
        """ to set item(s).
        For a matrix `m`
            - to set the `(i,j)`th element, write `m[i,j] = s`.
            - to set the `i`th row , write `m[i] = v` or `m[i,:] = v`, where `v` is a row vector type object.
            - to set the `j`th column , write `m[:,j] = v`, where `v` is a (column) vector type object.
        In contrary to the numpy convention, only positive indices are supported.
        
        Note that similar to numpy array, one can use `:` (or `...`) to set a row/column.
        One should also note that any python objects except integers can play the role of `:` or `...`
        except for boolean objects `True` and `False`, which act like 1 and 0.
        This issue can be fixed by adding more `if`s, but it is not necessary.
        """
        cdef int i=0,j=0,k
        ROW = False # `True` means the row of the requested slice is fixed
        COL = False # `True` means the column...
        if isinstance(ij,int) and ij in range(3):
            i = ij
            ROW = True
        elif isinstance(ij, tuple) and len(ij)==2:
            if isinstance(ij[0],int):
                if ij[0] in range(3):
                    i = ij[0]
                    ROW = True
                else:
                    raise Exception('Bad indices')
            if isinstance(ij[1],int):
                if ij[1] in range(3):
                    j = ij[1]
                    COL = True
                else:
                    raise Exception('Bad indices')
        else:
            raise Exception('Bad indices')

        if ROW and COL:
            self.e[i][j] = y
        elif ROW and isinstance(y, rowvector):
            for k in range(3):
                self.e[i][k] = y.e[k]
        elif COL and isinstance(y, vector):
            for k in range(3):
                self.e[k][j] = y.e[k]
        else:
            raise Exception('Bad set values')

    def __str__(self):
        cdef int i,j
        cdef str fmt = "{:.6g}"
        cdef str r0  = ", ".join([fmt.format(self.e[0][j]) for j in range(3)])
        cdef str r1  = ", ".join([fmt.format(self.e[1][j]) for j in range(3)])
        cdef str r2  = ", ".join([fmt.format(self.e[2][j]) for j in range(3)])
        cdef str mystring = "matrix([[{}],\n        [{}],\n        [{}]])".format(r0,r1,r2)
        return mystring

    def __repr__(self):
        return self.__str__()

    def asarray(self):
        """ returns the matrix as numpy.ndarray. """
        cdef int i,j
        return np.array([[self.e[i][j] for j in range(3)] for i in range(3)])

    cpdef void insert(self, x):
        """ can be used to insert the content of another matrix or a numpy.ndarray. """
        cdef int i,j
        cdef matrix mat
        if isinstance(x, matrix):
            mat = x
            for i in range(3):
                for j in range(3):
                    self.e[i][j] = mat.e[i][j]
        elif isinstance(x, numpy.ndarray):
            for i in range(3):
                for j in range(3):
                    self.e[i][j] = x[i,j]

    cpdef complex_type trace(self):
        """ returns the trace of the matrix. """
        return self.e[0][0] + self.e[1][1] + self.e[2][2]
        
    cpdef complex_type det(self):
        """ returns the determinant of the matrix. """
        cdef int i,j
        cdef complex_type det
        det = self.e[0][0] * (self.e[1][1] * self.e[2][2] - self.e[1][2] * self.e[2][1])\
            + self.e[0][1] * (self.e[1][2] * self.e[2][0] - self.e[1][0] * self.e[2][2])\
            + self.e[0][2] * (self.e[1][0] * self.e[2][1] - self.e[1][1] * self.e[2][0])
        return det

    cpdef double norm(self):
        """ returns the norm of the matrix. """
        cdef int i,j
        cdef double res = 0
        for i in range(3):
            for j in range(3):
                res += (self.e[i][j].real**2 + self.e[i][j].imag**2)
        return res**0.5

    cpdef double norm_sq(self):
        """ returns the norm-squared of the matrix. """
        cdef int i,j
        cdef double res = 0
        for i in range(3):
            for j in range(3):
                res += (self.e[i][j].real**2 + self.e[i][j].imag**2)
        return res

    cpdef vector eig(self):
        """ returns the eigenvalues of the matrix as a vector.
        """
        # Analytic calculation of the matrix eigenvalues.
        # We fist write det(A - s) = a*s^3 + b*s^2 + c*s + d ...
        cdef vector vec  = vector.__new__(vector)
        cdef complex_type a,b,c,d,Del0,Del1,Del2,C,w
        cdef complex_type w1 = (-1+1J*3**0.5)/2., w2 = (-1-1J*3**0.5)/2.
        cdef complex_type x0,x1,x2
        cdef complex_type e[3][3]
        e = self.e
        a = -1
        b =  e[0][0] + e[1][1] + e[2][2]
        c = -e[0][0]*e[1][1] - e[0][0]*e[2][2] + e[0][1]*e[1][0] + e[0][2]*e[2][0] - e[1][1]*e[2][2] + e[1][2]*e[2][1]
        d =  e[0][0]*e[1][1]*e[2][2] - e[0][0]*e[1][2]*e[2][1] - e[0][1]*e[1][0]*e[2][2] \
             + e[0][1]*e[1][2]*e[2][0] + e[0][2]*e[1][0]*e[2][1] - e[0][2]*e[1][1]*e[2][0]

        Del0 = b**2 - 3*a*c
        Del1 = 2*b**3 - 9*a*b*c + 27*a**2*d
        if abs(Del0)<1e-16:
            if abs(Del1)<1e-16:
                C = 0
            else:
                C = (Del1)**(1/3.)
        else:
            Del2 = (Del1**2 - 4*Del0**3)**(0.5)
            C = ((Del1 + Del2)/2)**(1/3.)
        if C==0:
            x0=b/3; x1=b/3; x2=b/3
        else:
            x0=(b+C+Del0/C)/(-3*a)
            x1=(b+C*w1+Del0/C/w1)/(-3*a)
            x2=(b+C*w2+Del0/C/w2)/(-3*a)
            #if x0.real>x1.real:
            #    x0,x1 = x1,x0
            #if x1.real>x2.real:
            #    x1,x2 = x2,x1
            #if x0.real>x1.real:
            #    x0,x1 = x1,x0
        vec.e[0] = x0
        vec.e[1] = x1
        vec.e[2] = x2
        return vec   
 
    cpdef matrix inv(self):
        """ returns the inverse of the matrix. """
        cdef matrix inv = matrix.__new__(matrix)
        cdef int i,j,ip,ipp,jp,jpp
        cdef complex_type det = self.det()
        if det==0:
            raise Exception("Oops: this matrix does not have an inverse because its determinant is zero.")
        for i in range(3):
            ip  = (i+1)%3
            ipp = (i+2)%3
            for j in range(3):
                jp  = (j+1)%3
                jpp = (j+2)%3
                inv.e[j][i] = (self.e[ip][jp]*self.e[ipp][jpp] - self.e[ip][jpp]*self.e[ipp][jp])/det 
        return inv 

    cpdef matrix naive_unitarize(self):
        """ returns a naive (special) unitarized version of the matrix, which is obtained by
        orthonormalizing the rows of the matrix. """
        # This is useful for random matrices close to
        # the identity matrix or as a initial guess for the process of unitarization by
        # the `self.project_SU3()` method.
        cdef matrix mat = matrix.__new__(matrix)
        cdef int i,j
        cdef double r
        cdef complex_type c, vdot=0
        for i in range(2):
            for j in range(3):
                mat.e[i][j] = self.e[i][j]
            if i==1:
                for j in range(3):
                    vdot += (mat.e[0][j].conjugate()*mat.e[1][j])
                for j in range(3):
                    mat.e[1][j] -= mat.e[0][j]*vdot # note that mat.e[0] is already normalized
            r = 0            
            for j in range(3):
                r += (mat.e[i][j].real**2 + mat.e[i][j].imag**2)
            if r==0: # the current row is identical to zero
                raise Exception('Bad matrix; cannot unitarize a singular matrix.')
            c = 1.0/r**0.5
            for j in range(3):
                mat.e[i][j] *= c
        mat.e[2][0] = (mat.e[0][1]*mat.e[1][2] - mat.e[0][2]*mat.e[1][1]).conjugate()
        mat.e[2][1] = (mat.e[0][2]*mat.e[1][0] - mat.e[0][0]*mat.e[1][2]).conjugate()
        mat.e[2][2] = (mat.e[0][0]*mat.e[1][1] - mat.e[0][1]*mat.e[1][0]).conjugate()
        return mat

    cpdef matrix _project_SL3(self):
        """ scales the matrix to make its deteminant unity. """
        cdef complex_type d = self.det()
        if d==-1: # it seems that cython cannot handle (-1)**(1/3.), so we use the following hack
            d *= (1-1e-15*1J)
        return self/d**(1/3.)

    cpdef matrix project_SU3(self, int n_hits=10, double tol=1e-4, int print_progress=0):
        """ projects the current ``3x3`` complex matrix ``Q`` onto SU(3) matrix ``W``
        by maximizing ``Re Tr (Q^\dagger W)``. 

        .. _MILC code: http://physics.utah.edu/~detar/milc.html 

        This code is developed following `project_su3_hit.c` in the `MILC code`_,
        which uses an iterative method based on hits in the various diagonal SU(2) subgroups.

        Parameters:
            - ``n_hits``: number of SU(3) hits. The default value is 10, which is typically good to\
                          reach to a precision with about 15 digits if the matrix is close to a unitary matrix.
            - ``tol``:    tolerance for SU(3) projection. If nonzero, treat `n_hits` as a maximum number of hits.\
                          If zero, treat `n_hits` as a prescribed number of hits. The default value is 1e-6.

        """
        # As an initial guess for the projection process one can use the `.naive_unitarize()` method.
        # In special cases, e.g `X = -I`, this yield `diag(-1,-1,1)` while it should give `e**{i*2\pi/3} I`
        # if we employ the criterion of increasing `Re Tr (Q^\dagger W)`.
        # To fix this bug, we first use the `._project_SL3()` method before using `.naive_unitarize()`.
        # The use of `_project_SL(3)` can be considered as preconditioning the problem.
        cdef matrix Q_adj = self._H()
        cdef matrix W = self._project_SL3().naive_unitarize() # initial guess
        cdef matrix V, h # intermediate matrices 
        cdef int p,q, n_hit, ind
        cdef double v0,v1,v2,v3,v_sq,r
        cdef double old_tr, new_tr, conver=1
        cdef double break_tol
        break_tol = tol if tol>0 else 1e-15
        old_tr = ReTr(Q_adj * W)
        for n_hit in range(n_hits):
            for ind in range(3): # Do three SU(2) hits
                V = W * Q_adj
                #=======================
                # pick out an SU(2) subgroup */
                p = ind%3
                q = (ind+1)%3
                if p>q: p,q = q,p
                #=======================
                # decompose V into SU(2) subgroups using Pauli matrix expansion
                # The SU(2) hit matrix is represented as v0 + i * Sum_j (sigma_j * vj)
                #     
                #     | A+iB     C+iD |     
                # M = |               |
                #     | E+iF     G+iH | 
                #     
                #   =  1/2*[ (A+G)*I + i*(B-H)*sigma_z + i*(F+D)*sigma_x + i*(C-E)*sigma_y ]
                #    + i/2*[ (B+H)*I - i*(A-G)*sigma_z - i*(C+E)*sigma_x - i*(F-D)*sigma_y ]
                #     
                # The second line does not contribute to `Re Tr (Q^\dagger W)`, so we drop it.
                # When the first line is identical to zero, we simply assign the identity matrix
                # as the projection of M onto SU(3).
                v0 = V.e[p][p].real + V.e[q][q].real
                v3 = V.e[p][p].imag - V.e[q][q].imag
                v1 = V.e[p][q].imag + V.e[q][p].imag
                v2 = V.e[p][q].real - V.e[q][p].real
                v_sq = v0**2 + v1**2 + v2**2 + v3**2
                #=======================
                h = matrix(identity=True) # for SU(2) hit
                if v_sq!=0:
                    r = 1./v_sq**0.5    # for normalization
                    h.e[p][p] = (v0-v3*1J)*r
                    h.e[p][q] =(-v2-v1*1J)*r
                    h.e[q][p] = (v2-v1*1J)*r
                    h.e[q][q] = (v0+v3*1J)*r
                W = h*W
            #=======================
            # convergence measure every third hit
            new_tr = ReTr(Q_adj * W)
            conver = (new_tr - old_tr)/old_tr # conver > 0 because trace always increases
            if print_progress==1:
                print("itr={}: old_tr={:.14f}, new_tr={:.12f}, conver={:.1e}".format(n_hit+1, old_tr, new_tr, conver))
            old_tr = new_tr
            if conver<break_tol:
                break
        if n_hits>0 and conver>tol and tol>0: 
            print("project_su3: No convergence: conver = ",conver)
        return W

    cpdef matrix traceless(self):
        """ returns the traceless part of the matrix. """
        cdef int i,j
        cdef matrix mat = matrix.__new__(matrix)
        cdef complex_type tr_3rd = self.trace()/3.
        for i in range(3):
            for j in range(3):
                mat.e[i][j] = self.e[i][j]
            mat.e[i][i] -= tr_3rd
        return mat

    cpdef matrix anti_hermitian_traceless(self):
        """ returns the anti hermitian, traceless part of the matrix. """
        cdef int i,j
        cdef matrix mat = matrix.__new__(matrix)
        cdef complex_type tr_3rd = 1j*(self.trace().imag)/3.
        for i in range(3):
            for j in range(3):
                mat.e[i][j] = (self.e[i][j] - self.e[j][i].conjugate())/2.
            mat.e[i][i] -= tr_3rd
        return mat

    cdef matrix _H(self):
        """ returns the adjoint matrix. """
        cdef matrix mat = matrix.__new__(matrix)
        cdef int i,j
        for i in range(3):
            for j in range(3):
                mat.e[j][i] = self.e[i][j].conjugate()
        return mat
        
    property H:
        """ returns the adjoint matrix. """
        # Note that it is much faster to use _H() rather than .H; the misssing time would be in matrix.__get__
        def __get__(self):
            return self._H()

    property shape:
        """ returns the shape of the matrix. """
        def __get__(self):
            return (3,3)
    

#==========================================================
# NOTE:
#   It would be faster to use the following functions, such as Adj(), instead of the class method _H() or property H
#   Because of the python wrapper for _H() [albeit we must use `cpdef` rather than `cdef` to define _H() if we want to use _H() from python],
#   and because of the __get__ method for property H (__get__ is also porbably a python wrapper).
#==========================================================

cpdef complex_type Tr(matrix xx):
    """ returns ``Tr( matrix )`` . """
    cdef matrix x = xx
    return x.e[0][0] + x.e[1][1] + x.e[2][2]

cpdef double ReTr(matrix xx):
    """ returns the real part of ``Tr( matrix )``. """
    cdef matrix x = xx
    return (x.e[0][0].real + x.e[1][1].real + x.e[2][2].real)

cpdef double ImTr(matrix xx):
    """ returns the imaginary part of ``Tr( matrix )``. """
    cdef matrix x = xx
    return (x.e[0][0].imag + x.e[1][1].imag + x.e[2][2].imag)


cpdef double ReTrTie(matrix xx, matrix yy):
    """ returns the real part of ``Tr( matrix1 * matrix2 )``. """
    cdef matrix x = xx
    cdef matrix y = yy
    cdef int i
    cdef double res=0
    for i in range(3):
        res += (x.e[i][0]*y.e[0][i] + x.e[i][1]*y.e[1][i] + x.e[i][2]*y.e[2][i]).real
    return res

cpdef matrix Adj(matrix xx):
    """ returns the adjoint of the matrix. """
    cdef matrix z = matrix.__new__(matrix)
    cdef matrix x = xx
    cdef int i,j
    for i in range(3):
        for j in range(3):
            z.e[j][i] = x.e[i][j].conjugate()
    return z

cdef inline complex_type cexp(complex_type x):
     return exp(x.real)*(cos(x.imag) + 1J*sin(x.imag))

cpdef matrix matexp_special(matrix xx):
    """ [Note: ***Works only if the eigenvalues of the input are  different***]
    returns the exponential of the 3x3 input matrix by taking advantage of the Cayley–Hamilton theorem
    stating that every square matrix satisfies its own characteristic equation.
    For a 3x3 matrix ``M``, this theorem indicates that::
        
            exp(M) = a I + b M + c M^2

    where ``I`` is the identity matrix and ``a``, ``b``, and ``c`` are three numbers that can be
    calculated using the eigenvalues of M, which satisfy::
        
        exp(s) = a + b s + c s^2
 
    Thus::
        
        |a|         |1  s1  s1^2|   |exp(s1)|
        |b| = inv(  |1  s2  s2^2| ) |exp(s2)|
        |c|         |1  s3  s3^2|   |exp(s3)|

    """
    cdef matrix I = matrix(identity=True)
    cdef matrix M = matrix.__new__(matrix)
    cdef vector exp_v = vector.__new__(vector)
    cdef matrix x = xx
    cdef vector v = x.eig() 
    cdef vector abc
    cdef complex_type s
    cdef i,j
    for i in range(3):
        s = v.e[i]
        exp_v.e[i] = cexp(s) 
        M.e[i][0] = 1
        M.e[i][1] = s 
        M.e[i][2] = s**2 
    v = M.inv() * exp_v
    return v[0]*I + v[1]*x + v[2]*(x*x)

cpdef matrix matexp(matrix xx, int n_trunc=12):
    """ returns the exponential of the input matrix obtained by the Taylor expansion. """
    cdef matrix z = matrix(identity=True)
    cdef matrix y = matrix(identity=True)
    cdef matrix x = xx
    cdef int n
    for n in range(1,n_trunc):
        y *= (x/n)  # y = x**n/n!
        z += y      # z = \sum x**n/n!
        if y.norm_sq()<1e-32:
            break
    return z

#==========================================================
cdef class vector:
    """ This is a Cython extension type to declare and define ``3x1`` complex matrices,
    or complex (column) vectors.
    After importing ``vector`` from :mod:`gauge_tools.lib._matrix`, one can use::

            vector()
            vector(zeros=True)

    to define an empty and zero (column) vectors, respectively.

    This extension type is equipped with several special methods:
        - ``__add__(vector x, vector y)``  supports instances of vector.
        - ``__sub__(vector x, vector y)``  supports instances of vector.
        - ``__mul__(x, y)``  supports instances of matrix, rowvector, and numbers as in linear algebra.
        - ``__truediv__(vector x, y)``  supports numbers.
        - ``__iadd__(self, vector y)``  supports instances of vector.
        - ``__isub__(self, vector y)``  supports instances of vector.
        - ``__imul__(self, y)``         supports numbers.
        - ``__itruediv__(self, y)``     supports numbers.
        - ``__getitem__(self, int i)``  to get an item.\
                        To get the `i` th element of vector `v`, write `v[i]`.
        - ``__setitem__(self, int i, y)``   to set an item.\
                        To set the `i` th element of vector `v`, write `v[i] = y`.

    In addition to these special methods, there are more methods and properties listed below.
    """
    # cdef readonly complex_type e[3]

    def __init__(self, zeros=False):
        """ Initiates a `3x1` complex vector. Being a column vector indicates that one can, e.g. multiply
        an instance of this vector by an instance of a matrix and obtain another
        column vector (as definied in the linear algebra).

        The vector is by default an empty vector unless one changes the default values of options.

        Parameters:
            - ``zeros`` (bool): if `True` returns a zero vector. The default value is `False`.
        """
        cdef int i
        if zeros:
            for i in range(3):
                self.e[i] = 0

    def __pos__(self):
        return self

    def __neg__(self):
        return self*(-1)

    def __add__(vector x, vector y):
        """ supports instances of vector. """
        cdef vector vec  = vector.__new__(vector)
        cdef vector vec1 = x
        cdef vector vec2 = y
        cdef int i
        for i in range(3):
            vec.e[i] = vec1.e[i] + vec2.e[i]
        return vec

    def __sub__(vector x, vector y):
        """ supports instances of vector. """
        cdef vector vec  = vector.__new__(vector)
        cdef vector vec1 = x
        cdef vector vec2 = y
        cdef int i
        for i in range(3):
            vec.e[i] = vec1.e[i] - vec2.e[i]
        return vec

    def __mul__(x, y):
        """ supports instances of matrix, rowvector, and numbers
        as defined in the linear algebra.
        """
        ### We check the type of x and y below because they can have different types.
        cdef vector vec = vector.__new__(vector)
        cdef matrix mat = matrix.__new__(matrix)
        cdef rowvector vec1
        cdef vector vec2
        cdef complex_type const 
        cdef int i,j
        if isinstance(x, matrix): # y is definitely an instance of vector
            mat  = x
            vec2 = y
            for i in range(3):
                vec.e[i] = mat.e[i][0]*vec2.e[0] + mat.e[i][1]*vec2.e[1] + mat.e[i][2]*vec2.e[2]
            return vec
        elif isinstance(x, rowvector): # y is definitely an instance of vector
            vec1 = x
            vec2 = y
            const = vec1.e[0]*vec2.e[0] + vec1.e[1]*vec2.e[1] + vec1.e[2]*vec2.e[2]
            return const
        elif isinstance(y, rowvector): # x is definitely an instance of vector
            vec1 = y
            vec2 = x
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = vec2.e[i] * vec1.e[j]
            return mat
        elif isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = y
            vec2  = x
            for i in range(3):
                vec.e[i] = vec2.e[i] * const 
            return vec
        elif isinstance(x, float) or isinstance(x, int) or isinstance(x, complex):
            const = x
            vec2  = y
            for i in range(3):
                vec.e[i] = const * vec2.e[i]
            return vec
        elif isinstance(y, matrix):
            raise Exception('A column vector can by multipiled by a matrix only as `matrix * vector`.')
        else:
            raise Exception('Bad Types')
        
    def __truediv__(vector x, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it can have different types.
        cdef vector vec  = vector.__new__(vector)
        cdef vector vec1 = x
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = 1.0/y
            for i in range(3):
                vec.e[i] = vec1.e[i] * const
            return vec
        else:
            raise Exception('Bad Types')

    def __iadd__(self, vector y):
        """ supports instances of vector. """
        cdef vector vec2 = y
        cdef int i
        for i in range(3):
            self.e[i] += vec2.e[i]
        return self

    def __isub__(self, vector y):
        """ supports instances of vector. """
        cdef vector vec2 = y
        cdef int i
        for i in range(3):
            self.e[i] -= vec2.e[i]
        return self

    def __imul__(self, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it accepts different types.
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = y
            for i in range(3):
                self.e[i] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __itruediv__(self, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it accepts different types.
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = 1.0/y
            for i in range(3):
                self.e[i] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __getitem__(self, int i):
        """ to get an item. To get the `i`th element of vector `v`, write `v[i]`. """
        if i in range(3):
            return self.e[i]
        else:
            raise Exception('Bad indices')

    def __setitem__(self, int i, complex_type y):
        """ to set an item. To set the `i`th element of vector `v`, write `v[i] = y`. """
        if i in range(3):
            self.e[i] = y
        else:
            raise Exception('Bad indices')

    def __str__(self):
        cdef int i
        cdef str fmt = "{:.6g}"
        cdef str c   = "; ".join([fmt.format(self.e[i]) for i in range(3)])
        cdef str mystring = "vector([{}])".format(c)
        return mystring

    def __repr__(self):
        return self.__str__()

    def asarray(self):
        """ returns the vector as numpy.ndarray. """
        cdef int i
        return np.array([self.e[i] for i in range(3)])

    def aslist(self):
        """ returns the vector as a list. """
        cdef int i
        return [self.e[i] for i in range(3)]

    cpdef void insert(self, x):
        """ can be used to insert the content of another vector (or a numpy.ndarray) in this instance. """
        cdef int i
        cdef vector vec
        if isinstance(x, vector):
            vec = x
            for i in range(3):
                self.e[i] = vec.e[i]
        elif isinstance(x, numpy.ndarray):
            for i in range(3):
                self.e[i] = x[i]

    cpdef double norm(self):
        """ returns the norm of the vector. """
        cdef int i
        cdef double res = 0
        for i in range(3):
            res += (self.e[i].real**2 + self.e[i].imag**2)
        return res**0.5
       
    cpdef double norm_sq(self):
        """ returns the norm-square of the vector. """
        cdef int i
        cdef double res = 0
        for i in range(3):
            res += (self.e[i].real**2 + self.e[i].imag**2)
        return res
       
    cpdef complex_type vdot(self, vector v2):
        """ (for two vectors) `v1.vdot(v2)` returns `v1.H * v2`;
        this is equivalent to `np.vdot(v1,v2)`.
        """
        cdef int i
        cdef complex_type res = 0
        for i in range(3):
            res += (self.e[i] * v2.e[i].conjugate())
        return res

    cpdef vector cross(self, vector v2):
        """ (for two vectors) `v1.cross(v2)` returns the cross product of them.
        """
        cdef vector v = vector.__new__(vector)
        v.e[0] = self.e[1]*v2.e[2] - self.e[2]*v2.e[1]
        v.e[1] = self.e[2]*v2.e[0] - self.e[0]*v2.e[2]
        v.e[2] = self.e[0]*v2.e[1] - self.e[1]*v2.e[0]
        return v

    cpdef vector conjugate(self):
        """ returns the conjugate of the vector. """
        cdef vector v = vector.__new__(vector)
        v.e[0] = self.e[0].conjugate()
        v.e[1] = self.e[1].conjugate()
        v.e[2] = self.e[2].conjugate()
        return v

    cdef rowvector _H(self):
        """ returns the adjoint vector. """
        cdef rowvector vec = rowvector.__new__(rowvector)
        cdef int i
        for i in range(3):
            vec.e[i] = self.e[i].conjugate()
        return vec
        
    property H:
        """ returns the adjoint vector. """
        def __get__(self):
            return self._H()

    property shape:
        """ returns the shape of the vector. """
        def __get__(self):
            return (3,1)

#==========================================================
cdef class rowvector:
    """ This is a Cython extension type to declare and define ``1x3`` complex matrices,
    or complex row vectors:
    After importing ``rowvector`` from :mod:`gauge_tools.lib._matrix`, one can use::

            rowvector()
            rowvector(zeros=True)

    to define an empty and zero row vectors, respectively.

    This extension type is equipped with several special methods:
        - ``__add__(rowvector x, rowvector y)``  supports instances of rowvector.
        - ``__sub__(rowvector x, rowvector y)``  supports instances of rowvector.
        - ``__mul__(x, y)``  supports instances of matrix, vector, and numbers as in linear algebra.
        - ``__truediv__(rowvector x, y)`` supports numbers.
        - ``__iadd__(self, rowvector y)`` supports instances of rowvector.
        - ``__isub__(self, rowvector y)`` supports instances of rowvector.
        - ``__imul__(self, y)``         supports numbers.
        - ``__itruediv__(self, y)``     supports numbers.
        - ``__getitem__(self, int i)``  to get an item.\
                    To get the `i` th element of rowvector `v`, write `v[i]`.
        - ``__setitem__(self, int i, y)`` to set an item.\
                    To set the `i` th element of rowvector `v`, write `v[i] = y`.

    In addition to these special methods, there are more methods and properties listed below.
    """
    # cdef readonly complex_type e[3]

    def __init__(self, zeros=False):
        """ Initiates a `1x3` complex vector. Being a row vector indicates that one can, e.g. multiply
        an instance of this vector to an instance of a matrix and obtain another
        row vector (as definied in the linear algebra).

        The vector is by default an empty vector unless one changes the default values of options.

        Parameters:
            - ``zeros`` (bool): if `True` returns a zero vector. The default value is `False`.
        """
        cdef int i
        if zeros:
            for i in range(3):
                self.e[i] = 0

    def __pos__(self):
        return self

    def __neg__(self):
        return self*(-1)

    def __add__(rowvector x, rowvector y):
        """ supports instances of rowvector. """
        cdef rowvector vec  = rowvector.__new__(rowvector)
        cdef rowvector vec1 = x
        cdef rowvector vec2 = y
        cdef int i
        for i in range(3):
            vec.e[i] = vec1.e[i] + vec2.e[i]
        return vec

    def __sub__(rowvector x, rowvector y):
        """ supports instances of rowvector. """
        cdef rowvector vec  = rowvector.__new__(rowvector)
        cdef rowvector vec1 = x
        cdef rowvector vec2 = y
        cdef int i
        for i in range(3):
            vec.e[i] = vec1.e[i] - vec2.e[i]
        return vec

    def __mul__(x, y):
        """ supports instances of matrix, vector, and numbers
        as defined in the linear algebra.
        """
        ### We check the type of x and y below because they can have different types.
        cdef rowvector vec = rowvector.__new__(rowvector)
        cdef matrix mat = matrix.__new__(matrix)
        cdef rowvector vec1
        cdef vector vec2
        cdef complex_type const 
        cdef int i,j
        if isinstance(y, matrix): # x is definitely an instance of rowvector
            vec1 = x
            mat  = y
            for i in range(3):
                vec.e[i] = vec1.e[0]*mat.e[0][i] + vec1.e[1]*mat.e[1][i] + vec1.e[2]*mat.e[2][i]
            return vec
        elif isinstance(y, vector): # x is definitely an instance of rowvector
            vec1 = x
            vec2 = y
            const = vec1.e[0]*vec2.e[0] + vec1.e[1]*vec2.e[1] + vec1.e[2]*vec2.e[2]
            return const
        elif isinstance(x, vector): # y is definitely an instance of rowvector
            vec1 = y
            vec2 = x
            for i in range(3):
                for j in range(3):
                    mat.e[i][j] = vec2.e[i] * vec1.e[j]
            return mat
        elif isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            vec1  = x
            const = y
            for i in range(3):
                vec.e[i] = vec1.e[i] * const 
            return vec
        elif isinstance(x, float) or isinstance(x, int) or isinstance(x, complex):
            vec1  = y
            const = x
            for i in range(3):
                vec.e[i] = const * vec1.e[i]
            return vec
        elif isinstance(x, matrix):
            raise Exception('A row vector can by multipiled by a matrix only as `vector * matrix`.')
        else:
            raise Exception('Bad Types')
        
    def __truediv__(rowvector x, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it can have different types.
        cdef rowvector vec  = rowvector.__new__(rowvector)
        cdef rowvector vec1 = x
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = 1.0/y
            for i in range(3):
                vec.e[i] = vec1.e[i] * const
            return vec
        else:
            raise Exception('Bad Types')

    def __iadd__(self, rowvector y):
        """ supports instances of rowvector. """
        cdef rowvector vec2 = y
        cdef int i
        for i in range(3):
            self.e[i] += vec2.e[i]
        return self

    def __isub__(self, rowvector y):
        """ supports instances of rowvector. """
        cdef rowvector vec2 = y
        cdef int i
        for i in range(3):
            self.e[i] -= vec2.e[i]
        return self

    def __imul__(self, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it accepts different types.
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = y
            for i in range(3):
                self.e[i] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __itruediv__(self, y):
        """ supports numbers, i.e. `y` must be a number. """
        ### We check the type of y below because it accepts different types.
        cdef complex_type const
        cdef int i
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, complex):
            const = 1.0/y
            for i in range(3):
                self.e[i] *= const
            return self
        else:
            raise Exception('Bad Types')

    def __getitem__(self, int i):
        """ to get an item. To get the `i`th element of row vector `v`, write `v[i]`. """
        if i in range(3):
            return self.e[i]
        else:
            raise Exception('Bad indices')

    def __setitem__(self, int i, complex_type y):
        """ to set an item. To set the `i`th element of row vector `v`, write `v[i] = y`. """
        if i in range(3):
            self.e[i] = y
        else:
            raise Exception('Bad indices')

    def __str__(self):
        cdef int i
        cdef str fmt = "{:.6g}"
        cdef str r   = ", ".join([fmt.format(self.e[i]) for i in range(3)])
        cdef str mystring = "vector([{}])".format(r)
        return mystring

    def __repr__(self):
        return self.__str__()

    def asarray(self):
        cdef int i
        return np.array([self.e[i] for i in range(3)])

    def aslist(self):
        cdef int i
        return [self.e[i] for i in range(3)]

    cpdef void insert(self, x):
        """ can be used to insert the content of another row vector (or a numpy.ndarray) in this instance. """
        cdef int i
        cdef rowvector vec
        if isinstance(x, rowvector):
            vec = x
            for i in range(3):
                self.e[i] = vec.e[i]
        elif isinstance(x, numpy.ndarray):
            for i in range(3):
                self.e[i] = x[i]

    cpdef double norm(self):
        """ returns the norm of the row vector. """
        cdef int i
        cdef double res = 0
        for i in range(3):
            res += (self.e[i].real**2 + self.e[i].imag**2)
        return res**0.5
       
    cpdef double norm_sq(self):
        """ returns the norm-square of the row vector. """
        cdef int i
        cdef double res = 0
        for i in range(3):
            res += (self.e[i].real**2 + self.e[i].imag**2)
        return res
       
    cpdef complex_type vdot(self, rowvector v2):
        """ (for two row vectors) `v1.vdot(v2)` returns `v1 * v2.H`;
        this is equivalent to `np.vdot(v2,v1)`.
        """
        cdef int i
        cdef complex_type res = 0
        for i in range(3):
            res += (self.e[i] * v2.e[i].conjugate())
        return res

    cpdef rowvector cross(self, rowvector v2):
        """ (for two row vectors) `v1.cross(v2)` returns the cross product of them.
        """
        cdef rowvector v = rowvector.__new__(rowvector)
        v.e[0] = self.e[1]*v2.e[2] - self.e[2]*v2.e[1]
        v.e[1] = self.e[2]*v2.e[0] - self.e[0]*v2.e[2]
        v.e[2] = self.e[0]*v2.e[1] - self.e[1]*v2.e[0]
        return v

    cpdef rowvector conjugate(self):
        """ returns the conjugate of the row vector. """
        cdef rowvector v = rowvector.__new__(rowvector)
        v.e[0] = self.e[0].conjugate()
        v.e[1] = self.e[1].conjugate()
        v.e[2] = self.e[2].conjugate()
        return v

    cdef vector _H(self):
        """ returns the adjoint of the vector. """
        cdef vector vec = vector.__new__(vector)
        cdef int i
        for i in range(3):
            vec.e[i] = self.e[i].conjugate()
        return vec
        
    property H:
        """ returns the adjoint of the vector. """
        def __get__(self):
            return self._H()

    property shape:
        """ return the shape of the vector. """
        def __get__(self):
            return (1,3)

#==========================================================
def benchmark(X, int N=1000,int N_repeat=100):
    import time
    time_norm = []
    time_norm_sq = []
    time_det = []
    time_eig = []
    time_exp = []
    time_exp_S = []
    T1 = time.time()
    for i in range(N_repeat):
        T1 = time.time()
        for j in range(N):
            X.norm()
        time_norm.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            X.norm_sq()
        time_norm_sq.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            X.det()
        time_det.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            X.eig()
        time_eig.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            matexp_special(X)
        time_exp_S.append((time.time()-T1)/N)
        T1 = time.time()
        for j in range(N):
            matexp(X)
        time_exp.append((time.time()-T1)/N)
    x = time_norm
    print("Test of norm():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_norm_sq
    print("Test of norm_sq():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_det
    print("Test of det():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_eig
    print("Test of eig():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_exp_S
    print("Test of exp_S():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))
    x = time_exp
    print("Test of exp():\t",end='')
    print("Time = (%.1f +- %.1f) ns; "%(np.mean(x)*1e9,np.std(x)*1e9))

#==========================================================

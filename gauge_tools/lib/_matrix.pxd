# Created by Javad Komijani, University of Tehran, Feb 2020.
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
ctypedef double complex complex_type

cdef class matrix:
    cdef readonly complex_type e[3][3]
    cpdef void insert(self, x)
    cpdef complex_type trace(self)
    cpdef complex_type det(self)
    cpdef double norm(self)
    cpdef double norm_sq(self)
    cpdef vector eig(self)
    cpdef matrix inv(self)
    cpdef matrix naive_unitarize(self)
    cpdef matrix _project_SL3(self)
    cpdef matrix project_SU3(self, int n_hits=*, double tol=*, int print_progress=*)
    cpdef matrix traceless(self)
    cpdef matrix anti_hermitian_traceless(self)
    cdef matrix  _H(self) # ~ .H

cdef class vector:
    cdef readonly complex_type e[3]
    cpdef void insert(self, x)
    cpdef double norm(self)
    cpdef double norm_sq(self)
    cpdef complex_type vdot(self, vector v2)
    cpdef vector cross(self, vector v2)
    cpdef vector conjugate(self)
    cdef rowvector _H(self)

cdef class rowvector:
    cdef readonly complex_type e[3]
    cpdef void insert(self, x)
    cpdef double norm(self)
    cpdef double norm_sq(self)
    cpdef complex_type vdot(self, rowvector v2)
    cpdef rowvector cross(self, rowvector v2)
    cpdef rowvector conjugate(self)
    cdef vector _H(self)


cpdef double ReTr(matrix x)
cpdef double ImTr(matrix x)

cpdef complex_type Tr(matrix x)

cpdef double ReTrTie(matrix x, matrix y)

cpdef matrix Adj(matrix x)
cpdef matrix matexp(matrix x, int n_trunc=*)
cpdef matrix matexp_special(matrix x) # use only if the eigenvalues of x are different

#==========================================================

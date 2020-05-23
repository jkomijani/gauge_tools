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

"""
This module uses the extension types of Cython to define :class:`gauge_tools.lib._site.site`,
which is a class of objects refering to lattice sites.
After importing site from :mod:`gauge_tools.lib._site`, one can use

        XX = site(x0, x1, x2, x3)

to define XX as a site referring to the point ``[x0,x1,x2,x3]`` on the lattice.

Operators ``+``, ``-``, ``+=``, and ``-=`` are defined to shift the site to specifc directions.
For ``mu, nu \in [0,1,2,3]``::

        XX += mu
        XX -= nu

shifts the site ``XX`` by one step in the ``mu`` direction and by one step in the opposite of ``nu`` direction.
One can also define::

        YY = XX + mu
        YY = XX - mu
        YY = XX + mu - nu

where the last line means that ``YY`` is a site obtained by shifting the site ``XX`` by one step
in the ``mu`` direction and by one step in the opposite of ``nu`` direction.

***NOTE*** The ``+`` and ``-`` operators defined here are not commutative: the direction
``mu`` must always come on the right hand side of the operator.
    
"""

#==========================================================
# cython: boundscheck=False, wraparound=False, nonecheck=False
# Do NOT use cdivision=True, otherwise the ``%`` division returns negative values unlike the python ``%`` division, which is always positive.

#==========================================================
cdef int[4] NN = [0,0,0,0] # must be filled using ``set_lattice_size()``

cdef int stride0, stride1, stride2, stride3
cdef int sites_stride0, sites_stride1, sites_stride2
cdef int blocksites_stride0, blocksites_stride1, blocksites_stride2, blocksites_stridea

def set_lattice_size(int _N0, int _N1, int _N2, int _N3, int dim=4):
    NN[0] = _N0
    NN[1] = _N1
    NN[2] = _N2
    NN[3] = _N3
    set_memory_strides(dim)

def get_lattice_size():
    return list(NN)

def set_memory_strides(int dim):
    global stride0, stride1, stride2, stride3
    stride0 = dim*NN[3]*NN[2]*NN[1]
    stride1 = dim*NN[3]*NN[2]
    stride2 = dim*NN[3]
    stride3 = dim
    global sites_stride0, sites_stride1, sites_stride2
    sites_stride0 = NN[3]*NN[2]*NN[1]
    sites_stride1 = NN[3]*NN[2]
    sites_stride2 = NN[3]
    global blocksites_stride0, blocksites_stride1, blocksites_stride2, blocksites_stridea
    b_NN0 = 1 if NN[0]==1 else NN[0]//2
    b_NN1 = 1 if NN[1]==1 else NN[1]//2
    b_NN2 = 1 if NN[2]==1 else NN[2]//2
    b_NN3 = 1 if NN[3]==1 else NN[3]//2
    blocksites_stridea = b_NN3*b_NN2*b_NN1*b_NN0
    blocksites_stride0 = b_NN3*b_NN2*b_NN1
    blocksites_stride1 = b_NN3*b_NN2
    blocksites_stride2 = b_NN3

#==========================================================
cdef class site:
    """ This is a Cython extension type to declare and define lattice sites.
    After importing ``site`` from :mod:`gauge_tools.lib._site`, one can use

        XX = site(x0, x1, x2, x3)

    to define XX as a site referring to the point ``[x0,x1,x2,x3]`` on the lattice.

    This extension type is equipped with several special methods:

        - ``__add__(site XX, int mu)``
        - ``__sub__(site XX, int mu)`` 
        - ``__iadd__(self, int mu)``
        - ``__isub__(self, int mu)``

    where ``mu \in [0,1,2,3]``
    """

    # cdef public int e[4]

    def __init__(self, int x0, int x1, int x2, int x3):
        self.e[0] = x0 % NN[0]
        self.e[1] = x1 % NN[1]
        self.e[2] = x2 % NN[2]
        self.e[3] = x3 % NN[3]
        if NN[0]==0:
            print("Do NOT forget to set the lattice size in this module (_site.pyx) using 'set_lattice_size(Nt,Nx,Ny,Nz)'")

    def __add__(site XX, int mu):
        """ Addition (from right) is supported for integers mu \in {0,1,2,3}; returns site + \hat{mu}. """
        if mu<0 or 3<mu:
            raise Exception('Bad direction')
        cdef site YY = XX
        cdef site ZZ = site.__new__(site)
        cdef int i
        for i in range(4):
            ZZ.e[i] = YY.e[i]
        ZZ.e[mu] = (ZZ.e[mu]+1)%NN[mu]
        return ZZ

    def __sub__(site XX, int mu):
        """ Subtraction (from right) is supported for integers mu \in {0,1,2,3}; returns site - \hat{mu}. """
        if mu<0 or 3<mu:
            raise Exception('Bad direction')
        cdef site YY = XX
        cdef site ZZ = site.__new__(site)
        cdef int i
        for i in range(4):
            ZZ.e[i] = YY.e[i]
        ZZ.e[mu] = (ZZ.e[mu]-1)%NN[mu]
        return ZZ

    def __iadd__(self, int mu):
        """ supports integers mu \in {0,1,2,3}; replaces site with site + \hat{mu}. """
        if mu<0 or 3<mu:
            raise Exception('Bad direction')
        else:
            self.e[mu] = (self.e[mu]+1)%NN[mu]
            return self

    def __isub__(self, int mu):
        """ supports integers mu \in {0,1,2,3}; replaces site with site - \hat{mu}. """
        if mu<0 or 3<mu:
            raise Exception('Bad direction')
        else:
            self.e[mu] = (self.e[mu]-1)%NN[mu]
            return self

    def __str__(self):
        cdef int i
        return [self.e[i] for i in range(4)].__str__()

    def __repr__(self):
        return self.__str__()

    def lattice_size(self):
        return get_lattice_size()

    cpdef int index(self):
        return blocksite_index_(self.e[0], self.e[1], self.e[2], self.e[3])

    cpdef int ks_eta(self,mu):
        # Note that, when XX and YY are instances of the site class, we have:
        #   1) (XX + YY).ks_eta(mu) = XX.ks_eta(mu) * YY.ks_eta(mu)
        #   2a) (\hat\nu).ks_eta(mu)  = -1 if mu>nu else 1
        #   2b) (-\hat\nu).ks_eta(mu) = -1 if mu>nu else 1
        #
        # From (1) and (2) one can easily conclude that 
        #   3) XX.ks_eta(mu) = (XX+mu).ks_eta(mu) 
        #                    = (XX-mu).ks_eta(mu)
        cdef int nu,w=0
        for nu in range(mu):
            w += self.e[nu]
        if w%2==0:
            return 1
        else:
            return -1

    property t:
        def __get__(self):
            return self.e[3]

#==========================================================
cdef inline int blocksite_index_(int x0, int x1, int x2, int x3):
    """
    returns an index for each site by:

        - dividing the lattice to 2^4 hypercubes.

        - labling the sites based on their positions in the hypercubes;\
            this yields 2^4 sets, where all members of a set locate on the same point on the hypercubes.

        - sorting the sets as:
            - ``0 to 7`` for even sites and ``9 to 15`` for odd sites.
            - ``0,2,4,...`` for sites with ``t%2==0`` and ``1,2,3,...`` for sites\
                    with ``t%2==1`` (not that ``x3`` is assumed to be ``t``).

        - assigning an index to each site.

    """
    cdef int a0,a1,a2,a3,a
    a0 = x0%2; x0 = x0//2
    a1 = x1%2; x1 = x1//2
    a2 = x2%2; x2 = x2//2
    a3 = x3%2; x3 = x3//2
    a = blocksite_index_dict[a0*8 + a1*4 + a2*2 + a3]
    return a*blocksites_stridea + x0*blocksites_stride0 + x1*blocksites_stride1 + x2*blocksites_stride2 + x3

# even-odd labeling of sites inside a hypercube
blocksite_index_dict = {ind:n for n,ind in enumerate([0,15,6,9,10,5,12,3,  1,14,7,8,11,4,13,2])}
# even = {0000, 1111, 0110, 1001, 1010, 0101, 1100, 0011}
# odd  = {0001, 1110, 0111, 1000, 1011, 0100, 1101, 0010}

#==========================================================
cpdef inline int link_index_(int x0, int x1, int x2, int x3, int mu) nogil:
    """ maps the point (x0,x1,x2,x3,mu) to a unique integer index. Indices must not be negative. """
    return x0*stride0 + x1*stride1 + x2*stride2 + x3*stride3 + mu 

cpdef int link_index(site XX, int mu):
    """ similar to :meth:`gauge_tools.lib._site.link_index_`
    except that a lattice site is specified by an instance of :class:`gauge_tools.lib._site.site`. """
    return link_index_(XX.e[0], XX.e[1], XX.e[2], XX.e[3], mu)
    
cpdef inline int site_index_(int x0, int x1, int x2, int x3) nogil:
    #""" Maps the point (x0,x1,x2,x3) to a unique integer index. Indices must not be negative. """
    return x0*sites_stride0 + x1*sites_stride1 + x2*sites_stride2 + x3 

cpdef int site_index(site XX):
    return site_index_(XX.e[0], XX.e[1], XX.e[2], XX.e[3])
    
#==========================================================

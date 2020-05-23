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

from ..lib._rand cimport uniform
from ..lib._matrix cimport matrix, ReTr, Adj
from ..lib._site cimport site
from ..lib._site cimport link_index as index

from .._gtcore import gauge_field as _gauge_field

cimport numpy
import copy

#==========================================================
# Copy this block to any new application similar to those
# in `gauge_tools/util/` in order to set up the global variables.

def set_param(param):
    global nx, ny, nz, nt, dim, nc
    nx, ny, nz, nt, dim, nc = param.nx, param.ny, param.nz, param.nt, param.dim, param.nc
    global num_sites, num_links
    num_sites = param.num_sites
    num_links = param.num_links
    global ALL_links, ALL_sites
    ALL_sites = param.ALL_sites
    ALL_links = param.ALL_links

cdef int nx, ny, nz, nt
cdef int dim
cdef int nc

cdef int num_sites, num_links

ALL_sites = lambda: []
ALL_links = lambda: []

#==========================================================
class gaugefix(object):
    """ Utilities for gauge fixing.
    Currently only the Landau (Lorenz) and Coulomb gauges are supported for ``nc=3``.
    """
    #===================================
    def __init__(self, param):
        """ The input is an instance of :class:`gauge_tools.param`,
        which will be used to set the global parameters.
        """
        set_param(param)
    #===================================
    def gaugefix_Coulomb(self, numpy.ndarray U, int gaugefix_dir=3, **kwargs):
        """ See :meth:`gauge_tools.util.gaugefix.gaugefix.gaugefix_Landau`. """
        return self.gaugefix_Landau(U, gaugefix_dir=gaugefix_dir, **kwargs)
    #===================================
    def gaugefix_Landau(self, numpy.ndarray U, int gaugefix_dir=-1, int max_itr=200, double gaugefix_tol=1e-9, fname='',
                            int n_reunit=100, random_overrelax=True, random_overrelax_limits=[1.7,1.9], **sweep_kwargs):
        """ 
        .. _arXiv:1212.5221: https://arxiv.org/abs/1212.5221v2
        .. _MILC code: http://physics.utah.edu/~detar/milc.html

        fixes the gauge of links ``U`` to the Landau (Lorenz) and Coulomb gauges for ``nc=3``,
        and ***replaces*** the original links with the gauge fixed ones.
        The default choice for the gauge is the Landau gauge. For the Coulomb gauge,
        set ``gaugefix_dir`` to ``mu = 0,1,2,3`` depending on the time-like direction.
    
        For a brief review of gauge fixing see e.g. [`arXiv:1212.5221`_].
        Gauge fixing is a large scale problem, with a SU(3) matrix degree of freedom at each site.
        We divide the SU(3) matrix degrees of freedom into two sets with even/odd parity sites.
        If we set aside one of the sets, the other set can be easily optimized because SU(3) matrix
        degrees of freedom get decoupled.

        We tackle the problem by relaxation method. At each iteration we optimize either of
        even or odd parity sites by calling :meth:`gauge_tools.util.gaugefix.gaugefix.sweep_Landau()`,
        and repeat untill the solution converges.

        Parameters:
            - ``U``     (numpy.ndarray):    the links to be gauge fixed.
            - ``gaugefix_dir``    (int):    the default value is -1 indicating the Landau gauge.\
                                            When sets to 0,1,2,3 the gauge fixing condition would be\
                                            the Coulomb gauge in corresponding direction.
            - ``max_itr``         (int):    the maximum number of iterations.
            - ``gaugefix_tol`` (double):    the criterion to stop the iterations.
            - ``fname``        (string):    a file name to dubp the details (if not an empty string).

        Comment:
            This part is developed following `arXiv:1212.5221`_ and `gaugefix2.c` from the `MILC code`_.
        """
        cdef double w_min=0, w_max=0
        cdef int n_itr=0, parity
        cdef double chi2_new=0, chi2_old=0
        cdef double theta_new=0, theta_old=0
        if "overrelax_w" not in sweep_kwargs.keys():
            sweep_kwargs["overrelax_w"] = 1
        if random_overrelax:
            w_min = random_overrelax_limits[0] 
            w_max = random_overrelax_limits[1] 
        chi2_new  = self.chi2_Landau(U, gaugefix_dir)
        theta_new = self.theta_Landau(U, gaugefix_dir)
        if fname!='':
            fw = open(fname, 'a')
            message  = "#iteraton\tchi2\t\ttheta\t\toverrelax_w\n"
            message += "%d\t%.15e\t%.15e\t%g\n"%(n_itr, chi2_new, theta_new, sweep_kwargs["overrelax_w"])
            fw.write(message)
        for n_itr in range(max_itr):
            chi2_old = chi2_new
            parity   = n_itr%2
            if n_itr>0 and random_overrelax:
                sweep_kwargs["overrelax_w"] = uniform(w_min, w_max)
            chi2_new = self.sweep_Landau(U, parity, gaugefix_dir, **sweep_kwargs)
            theta_new= self.theta_Landau(U, gaugefix_dir)
            if fname!='':
                message = "%d\t%.15e\t%.15e\t%g\n"%(n_itr+1, chi2_new, theta_new, sweep_kwargs["overrelax_w"])
                fw.write(message)
            if abs(chi2_new - chi2_old) < gaugefix_tol:
                break
            # Reunitarize every n_reunit iterations to correct numerical errors
            if n_itr>0 and (n_itr % n_reunit)==0:
                _gauge_field.makesure_SU3(U)
        fw.close()
        # Reunitarize at the end, unless we just did it in the loop
        if max_itr>0 and (n_itr % n_reunit)!=0:
            _gauge_field.makesure_SU3(U)
        print("GaugeFix: Ended at step %d. chi-squared %.8e, delta %.3e"%(n_itr, chi2_new, chi2_new-chi2_old))
        return chi2_new
    #===================================
    def sweep_Landau(self, numpy.ndarray U, int parity, int gaugefix_dir=-1, int n_hits=5, double overrelax_w=1.):
        """ carries out one sweep (iteration) of gauge fixing.
    
        See :meth:`gauge_tools.util.gaugefix.gaugefix.gaugefix_Landau`
        and :meth:`gauge_tools.util.gaugefix.local_hit_Landau` for details.

        Parameters:
            - ``U`` (numpy.ndarray):    the links to be sweeped.
            - ``parity`` (0 or 1):      the parity of sites to be hitted.
            - ``gaugefix_dir`` (int):   the default value is -1 indicating the Landau gauge.\
                                        When sets to 0,1,2,3 the gauge fixing condition would be\
                                        the Coulomb gauge in the corresponding direction.
        Output:
            - ``chi2``: the chi-squared per link of the updated links.
        """
        cdef double chi2 = 0
        cdef tuple xx
        for xx in ALL_sites(): # xx = (t,x,y,z)
            if sum(xx)%2==parity:
                chi2 += local_hit_Landau(U, xx, gaugefix_dir=gaugefix_dir, n_hits=n_hits, overrelax_w=overrelax_w)
        return chi2/num_links # normalized to max of 1 for identity links
    #===================================
    def chi2_Landau(self, numpy.ndarray U, int gaugefix_dir=-1):
        """ is a standalone method to measure the chi-squared that ought to be
        maximized for gauge fixing. Note that this measures contributions from
        links connected to all even sites (equal to those of odd sites).

        Parameters:
            - ``U`` (numpy.ndarray):    the links to be gauge fixed.
            - ``gaugefix_dir`` (int):   the default value is -1 indicating the Landau gauge.\
                                        When sets to 0,1,2,3 the gauge fixing condition would be\
                                        the Coulomb gauge in the corresponding direction.
        Output:
            - chi-squared divided by number of links.
        """
        cdef double chi2 = 0
        cdef tuple xx
        for xx in ALL_sites():
            if sum(xx)%2==0:
                chi2 += ReTr( gaugeflow(U, xx, gaugefix_dir=gaugefix_dir) )
        return chi2 / (num_links*nc) # normalized to max of 1 for identity links
    #===================================
    def theta_Landau(self, numpy.ndarray U, int gaugefix_dir=-1):
        """ calculates ``Tr \sum_x gaugeflow(x) * gaugeflow(x)^\dagger``,
        where ``gaugeflow`` is :meth:`gauge_tools.util.gaugefix.gaugeflow`.
        """
        cdef matrix Delta = matrix(zeros=True)
        cdef matrix I = matrix(identity=True)
        cdef matrix K
        cdef double theta=0
        cdef tuple xx
        for xx in ALL_sites():
            K = gaugeflow(U, xx, gaugefix_dir=gaugefix_dir)
            Delta  = K - Adj(K)         # 1/(2i) is taken care below
            Delta -= I*Delta.trace()/nc # subtract the trace to make it traceless
            theta += ReTr( Delta * Adj(Delta) )
        return theta/4 / (num_sites*nc) # 1/4 is for (1/2i) * (-1/2i)
    #===================================

#==========================================================
cpdef double local_hit_Landau(numpy.ndarray U, tuple xx, int gaugefix_dir=-1, int n_hits=5, double overrelax_w=1.):
        """ carries out a local gauge fixing at site xx.

        See :meth:`gauge_tools.util.gaugefix.gaugefix.gaugefix_Landau` for details of the strategy of gauge fixing.

        This method only updates the links connected to site ``x``.

        Following Eq. (19) of [arXiv:1212.5221v2], we define ``K(x)``::

                K(x) = \sum_{\mu} (U_\mu(x) + U_\mu(x-\hat{\mu})^\dagger)
    
        and project ``K(x)`` onto SU(3) to obtain ``g(x)^\dagger``.
        We call ``K(x)`` gauge flow of links from site ``x``, and ``K(x)^\dagger`` gauge flow of links to site ``xx``.
        It turns out that the imaginary part of ``K(x)`` has interesting properties::

                Delta(x) = (K(x) - K(x)^\dagger)/(2i) = \partial_\mu A_\mu (x) (in the continuum limit).
                \sum_x Delta(x) = 0  (globally).

        In Landaue gauge we aim to maximize the real trace of ``\sum_{x} K(x)`` (globally),
        which would be equivalent to vanishing imaginary part of the flow (``Delta(x)``) locally.

        Parameters:
            - ``U``   (numpy.ndarray):      the links to be gauge fixed.
            - ``xx``  ( (int,int,int,int) ): specifying the site for a local hit. 
            - ``gaugefix_dir``  (int):      the default value is -1 indicating the Landau gauge.\
                                            When sets to 0,1,2,3 the gauge fixing condition would be\
                                            the Coulomb gauge in the corresponding direction.
            - ``n_hits``        (int):      number of hits in SU(3) projection.
            - ``overrelax_w``   (double):   the overrelaxation coefficient.

        Output:
            contribution of the current site to the total chi-squared (after local optimization).
        """
        cdef matrix K = matrix(zeros=True)
        cdef matrix I = matrix(identity=True)
        cdef matrix g, g_adj
        cdef int mu, ind0, ind1
        cdef site XX = site(*xx)
        for mu in range(dim):
            if mu==gaugefix_dir:
                continue
            ind0 = index(XX,mu)
            ind1 = index(XX-mu,mu)
            K += (U[ind0] + Adj(U[ind1]))
        g_adj = K.project_SU3(n_hits=n_hits,tol=0)
        if overrelax_w!=1:
            g_adj = (I + overrelax_w*(g_adj-I)*(I + 0.5*(overrelax_w-1)*(g_adj-I))).project_SU3(n_hits=n_hits)
        g = Adj(g_adj)
        for mu in range(dim):
            ind0 = index(XX,mu)
            ind1 = index(XX-mu,mu)
            U[ind0] = g * U[ind0]
            U[ind1] = U[ind1] * g_adj
        return ReTr(g*K)/nc

#==========================================================
cpdef matrix gaugeflow(numpy.ndarray U, tuple xx, int gaugefix_dir=-1):
        """ 
        .. _arXiv:1212.5221: https://arxiv.org/abs/1212.5221v2

        Following Eq. (19) of `arXiv:1212.5221`_, we define ``K(x)``::

                K(x) = \sum_{\mu} (U_\mu(x) + U_\mu(x-\hat{\mu})^\dagger)
    
        We call ``K(x)`` flow of links from site ``x``, and ``K(x)^\dagger`` flow of links to site ``xx``.
        It turns out that the imaginary part of ``K(x)`` has interesting properties::

                Delta(x) = (K(x) - K(x)^\dagger)/(2i) = \partial_\mu A_\mu (x) in the continuum limit.

                \sum_x Delta(x) = 0  (globally).

        In Landau gauge we aim to maximize the real trace of ``\sum_{x} K(x)`` (globally),
        which would be equivalent to vanishing imaginary part of the flow (``Delta(x)``) locally.

        Parameters:
            - ``U`` (numpy.ndarray):    the links to be gauge fixed.
            - ``xx`` ( (int,...,int) ): specifying the site for a local hit. 
            - ``gaugefix_dir`` (int):   the default value is -1 indicating the Landau gauge.\
                                        When sets to 0,1,2,3 the gauge fixing condition would be\
                                        the Coulomb gauge in the corresponding direction.
        """
        cdef matrix K = matrix(zeros=True)
        cdef int mu, ind0, ind1
        cdef site XX = site(*xx)
        for mu in range(dim):
            if mu==gaugefix_dir:
                continue
            ind0 = index(XX,mu)
            ind1 = index(XX-mu,mu)
            K += (U[ind0] + Adj(U[ind1]))
        return K

#==========================================================

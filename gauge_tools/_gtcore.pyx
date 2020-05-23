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

from .lib._rand cimport uniform, rand, randrange
from .lib._matrix cimport matrix, ReTrTie, ReTr, Adj, matexp
from .lib._rand_matrix import random_special_unitary

from .lib._site import  set_lattice_size
from .lib._site cimport site

import gvar as gv
import numpy as np
import time
import itertools
import copy

import cython
cimport numpy

from libc.math cimport exp
from .lib._rand import seed; seed('time') # to set the seed to the current time

#==========================================================
class param(object):
    """ A class to save the parameters of the lattice simulation,
    and also to set the global variables.

    Parameters: ``nx, ny, nz, nt, dim, nc``, which are the size of
    lattice in ``x``, ``y``, ``z``, ``t`` directions, the ``dimension``
    and ``number of colors``, respectively.
    """
    def __init__(self, nx_,ny_,nz_,nt_,dim_,nc_):
        self.set_globals(nx_,ny_,nz_,nt_,dim_,nc_)
        # save a copy of globals for use outside of this module
        self.nx,self.ny,self.nz,self.nt,self.dim,self.nc = nx,ny,nz,nt,dim,nc
        self.num_sites = num_sites
        self.num_links = num_links
        self.ALL_sites = ALL_sites
        self.ALL_links = ALL_links
    #===================================
    @staticmethod
    def set_globals(nx_,ny_,nz_,nt_,dim_,nc_):
        if   dim_==4: pass
        elif dim_==3: nz_ = 1
        elif dim_==2: nz_,ny_ = 1,1
        else:
            raise Exception("`dim` must be either 2,3, or 4.")
        #=====
        global nx, ny, nz, nt, dim, nc
        nx,ny,nz,nt,dim,nc = nx_,ny_,nz_,nt_,dim_,nc_
        set_lattice_size(nx,ny,nz,nt,dim) 
        #=====
        global num_sites, num_links
        num_sites = nx*ny*nz*nt
        num_links = nx*ny*nz*nt*dim
        #=====
        global stride0, stride1, stride2, stride3
        stride0 = dim*nt*nz*ny
        stride1 = dim*nt*nz
        stride2 = dim*nt
        stride3 = dim
        #===== we now define some lambda functions for sweeping all links/sites
        global ALL_links, ALL_sites
        ALL_links = lambda: itertools.product(range(nx),range(ny),range(nz),range(nt),range(dim))
        ALL_sites = lambda: itertools.product(range(nx),range(ny),range(nz),range(nt))
    #===================================

#==========================================================
class gauge_field(object):
    """ A gauge field with a generic action. """
    #===================================
    def __init__(self, create_links=True,  **raise_kwargs):
        self.SU3  = random_special_unitary(nc) # use SU3 to denote the instance independent of nc
        if create_links:
            self.create_links(**raise_kwargs)
        else:
            self.U = None
    #===================================
    def create_links(self, str load_fname='', int load_ind=0):
        cdef numpy.ndarray[object, ndim=1] U
        if load_fname!='':
            U = self.load_cfg(load_fname)
        elif num_links>0 and nc==3: # deosn't support other values of nc yet!
            U = np.array([matrix(identity=True) for _ in range(num_links)])
        else:
            raise Exception("Cannot create U because the global variable num_links is not set yet or nc!=3.")
        self.U = U
        self.sweep_cntr = load_ind
        self.history = {'accept_rate':{}}
    #===================================
    def sweep(self, calc_staples, double beta, int n_hits=10, double eps_hit=0.24, int n_su3_samples=100):
        """ The main input is a function denoted by `calc_staples` that is used to calulate the relevant part of the action.
        Note that with the Metropolis algorithm used here, we calculate the relevant contribution to the action
        from the link at site `xx` and `mu` direction rather than calculating the total action.
        We organized this as::

                 (- beta/nc) * Re Tr ( U_mu(xx) Gamma_mu(xx) ) 

        where the staples ``Gamma_mu(xx)`` are common for all hits of the Metropolis update of the link.
        Note that the staples defined here are the adjoint of the conventional staples.

        Parameters:
            - ``calc_staples``: calculates the staples ``Gamma_mu(xx)`` needed to compute the relevant action.
            - ``beta``:         is the beta of the action.
            - ``n_hits``:       number of hits to each link.
            - ``eps_hit``:      the epsilon used in generating SU(n) random matrices.

        """
        cdef double Snn, Snn_prime, dS
        cdef double c1 = -beta/nc
        cdef int accept_cntr = 0
        cdef tuple nn
        cdef matrix Unn, Unn_prime, Gnn
        cdef numpy.ndarray[object, ndim=1] U = self.U
        cdef list samples = self.SU3.gen_samples(eps_hit, n_su3_samples)
        for nn in ALL_links(): # nn = (x,y,z,t,mu)
            Gnn = calc_staples(U, nn)
            Unn = U[index_(*nn)]
            Snn = c1 * ReTrTie(Unn, Gnn)
            for _ in range(n_hits):
                Unn_prime = Unn * samples[randrange(n_su3_samples)]
                Snn_prime = c1 * ReTrTie(Unn_prime, Gnn)
                dS        = Snn_prime - Snn
                if dS<0 or exp(-dS)>rand(): # if True update the link and Snn
                    Unn.insert(Unn_prime)
                    Snn   = Snn_prime
                    accept_cntr += 1
        self.sweep_cntr += 1
        self.history['accept_rate'][self.sweep_cntr] =  accept_cntr/(num_links*n_hits)
    #===================================
    @staticmethod
    def save_cfg(U, fname):
        np.save(fname, U)
    #===================================
    @staticmethod
    def load_cfg(fname):
        return np.load(fname, allow_pickle=True)
    #===================================
    def loadin_cfg(self, fname):
        self.U = np.load(fname, allow_pickle=True)
    #===================================
    @staticmethod
    def makesure_SU3(numpy.ndarray U): # to correct the numerical errors
        cdef Py_ssize_t ind 
        for ind in range(num_links):
            U[ind] = U[ind].naive_unitarize()
    #===================================
    @staticmethod
    def avg_links(numpy.ndarray U):
        cdef Py_ssize_t i
        cdef matrix avg = matrix(zeros=True)
        for i in range(len(U)): avg += U[i]
        return avg/len(U)
    #===================================

#==========================================================
class MC_util(object):
    """ Utilities for generating and analyzing configurations from Monte Carlo simulation.
    """
    #===================================
    def __init__(self, actn_util, meas_util, GF):
        """ all three inputs are instances of action_util, measurement_util, and gauge_field classes. """
        self.actn = actn_util
        self.meas = meas_util
        self.GF   = GF
        self.history  = {'u0':{}}
    #===================================
    def gen_cfgs(self, n_cfgs, n_skip, mode="update", load_fname=None, load_ind=None,
                    n_hits=10, eps_hit=0.24, print_parameters=True, save_=True,
                    fname_lambda=None, **sweep_kwargs):
        """ generates new configurations.

        Parameters:
            - ``n_cfgs`` (int):  number of configurations to generate.

            - ``n_skip`` (int):  number of sweeps between two kept configuration.

            - ``mode`` (str):   should be either:
                - "from-scratch":   to start a new set of configurations from scratch,
                - "load&update":    to load a previously saved configuration and update it,
                - "update":         to update the present configuration (self.GF.U).

            - ``laod_fname`` (str): if ``mode=="load&update"``, the filename of the configuration to be loaded.

            - ``load_ind`` (int):   if ``mode=="load&update"``, the index (`sweep_cntr`) associated to
                                    the configuration to be loaded.

            - ``n_hits`` (int):     number of hits in the Metropolis algorithm

            - ``eps_hits`` (double): the value of `eps` for generating random matrices in the Metropolis algorithm.

            - ``print_parameters`` (boold): if `True`, prints parameters of simulation before startin to simulate.

            - ``save_`` (bool):     if `False`, does not save any configurations (useful for updating the simulation parameters).

            - ``fname_lambda``:     a function to construct a file name to save the configurations\
                                    such as ``fname_lambda = lambda ind_cfg: "cfg_{}".format(ind_cfg)``.

        """

        if not (isinstance(n_cfgs, int) and n_cfgs>0):
            return

        if mode=="from-scratch":
            header = "Start to thermalize a new ensemble of gauge field configurations with parameters:"
            self.GF.create_links()
        elif mode=="load&update":
            header = "Expanding gauge field configurations with parameters:"
            header += "\n(The starting cfg for updating is '{}' [ind={}].)".format(load_fname, load_ind)
            self.GF.create_links(load_fname=load_fname, load_ind=load_ind)
        elif mode=="update":
            if not isinstance(self.GF.U, np.ndarray):
                raise Exception("Cannot generate confinguration because the starting point is not clear.")
            header = "Expanding gauge field configurations with parameters:"
            # self.GF already exists
        else:
            raise Exception("Oops: the starting point is not clear.")

        if print_parameters:
            print(header)
            print("   nx,ny,nz,nt = {}".format( (nx,ny,nz,nt) ) )
            print("   {}".format(self.actn.fmt_action_details()))
            print("   eps_hit = {}".format(eps_hit))
            print("   n_hits  = {}".format(n_hits))

        # generate configurations 
        for n in range(n_cfgs):
            T1 = time.time()
            for j in range(n_skip):
                self.GF.sweep(self.actn.calc_staples, self.actn.beta, n_hits=n_hits, eps_hit=eps_hit, **sweep_kwargs)
            self.GF.makesure_SU3(self.GF.U) # to correct the numerical errors
            if save_ and fname_lambda!=None:
                self.GF.save_cfg( self.GF.U, fname_lambda(self.GF.sweep_cntr) )
                T2 = time.time()
                cntr = self.GF.sweep_cntr
                rate = self.GF.history['accept_rate'][cntr]
                print("A new cfg is generated and saved; sweep_ind={} and accept_rate={:.2g} (#TIME = {:.4g})".format(cntr, rate, T2-T1))
    #===================================
    def thermalize(self, n_therm, update_u0=False, binsize=10, **kwargs):
        """ to thermalize a new ensemble.

        Parameters:
            - ``n_therm`` (int): number of sweeps for thermalization.

            - ``update_u0`` (bool): if `True``, updates the tadpole improvement factor `u0` in the process\
                                of thermalizing. This can be done only if `n_therm>=50`.

            - ``binsize`` (int): meaningful only when updating `u0`. Spcefies the number of sweeps between updating\
                                the value of `u0`.
        """
        # For updating the value of `u0` we assume that `n_therm>=50`.
        if update_u0==False or n_therm<50:
            self.gen_cfgs(1, n_therm, mode="from-scratch", **kwargs)
        else:
            T1 = time.time()
            print("*** Thermalization along with updating `u0` every %d sweeps ***"%binsize)
            # n_therm = n1 + n2, where in the first n1 sweeps we update u0 every `binsize` sweeps,
            #                       and in the next n2 sweeps we set u0 to the average of the last 4 measurements
            n1 = binsize*(n_therm//binsize - 1)
            n2 = n_therm - n1
            for i in range(n1//binsize):
                if i==0:
                    self.gen_cfgs(1, binsize, mode="from-scratch", save_=False, **kwargs)
                else:
                    self.gen_cfgs(1, binsize, mode="update", print_parameters=False, save_=False, **kwargs)
                self._update_action()
            print("The first %d sweeps are perfomed with updaing `u0` every %d sweeps"%(n1,binsize), end=' ')
            print("(#TIME = {:.4g})".format(time.time()-T1))
            u0 = [self.history['u0'][i] for i in range(n1-3*binsize, n1+1, binsize)]
            avg_u0 = float("%.3f"%np.mean(u0))
            print("For the next %d sweeps u0 is set to plaq**(1/4) = %g (average of the last %d updates)"%(n2, avg_u0, 4))
            self._update_action(u0=avg_u0)
            self.gen_cfgs(1, n2, mode="update", save_=True, **kwargs)
        print("Thermalization concludes.")
    #===================================
    def _update_action(self, u0=None):
        if u0==None:
            plaq = self.meas.avg_plaq(self.GF.U)
            u0 = plaq**0.25 if plaq>0 else self.actn.u0
            self.history['u0'][self.GF.sweep_cntr] = u0
        self.actn.set_staples(u0)
    #===================================
    def eval_fn(self, fn, cfg_list, fname_lambda, avg_data=True, binsize=1):
        """ to evaluate `fn` for a list of configurations and calculate the
        Monte Carlo average of `fn`. Note that `fn` can be a list of functions
        that only depend on link varibles; e.g.::

                fn = lambda U: (calc_a_a(U), calc_a_2a(U))

        where ``calc_a_a`` and ``calc_a_2a`` are themselves are function of ``U``.

        Parameters:
            - ``fn``    (function): described above
            - ``cfgs_list`` (list): a list of indices of configurations to be loaded and used for the Monte Carlo averages.
            - ``fname_lambda``:     a function to construct a file name to load the configurations\
                                    such as ``fname_lambda = lambda ind_cfg: "cfg_{}".format(ind_cfg)``.
        """
        dset = [fn(gauge_field.load_cfg( fname_lambda(n_cfg) )) for n_cfg in cfg_list]
        if avg_data:
            if binsize>1: dset = self.bin(dset, binsize)
            return gv.dataset.avg_data(dset)
        else:
            return dset
    #===================================
    @staticmethod
    def bootstrap(G):
        n_cfgs = len(G)
        G_bootstrap = []    # new ensemble
        for i in range(n_cfgs):
            alpha = randrange(n_cfgs)  # choose random configuration
            G_bootstrap.append(G[alpha])  # keep G[alpha] 
        return G_bootstrap
    #===================================
    @staticmethod
    def bin(G, binsize):
        G_binned = []                  # binned ensemble
        for i in range(0,len(G),binsize): # loop on bins
            G_avg = 0
            for j in range(binsize):      # loop on bin elements
                G_avg = G_avg + G[i+j]
            G_binned.append(G_avg/binsize)
        return G_binned
    #===================================


#==========================================================
class action_util(object):
    """ For defining the action used in the simulations.

    Parameters:
        - ``grid``   (class):   an instance of `grid_util` class for access to grid utilities.
        - ``action``  (str):    can be for example `'W'`, which is the Wilson action, or `'imp'` for improved actions.\
                                The default value is `'W'`.
        - ``beta``  (float):    the lattice `beta = 6/g^2/u0**4`, where `g` is the coupling\
                                and `u0` is the tadpole improvement factor.
        - ``u0``:   (float):    tadpole improvement `u0`; the default value is 1 indicating no improvement.

    Note that our definition of ``beta`` absorbs ``1/u0^4``.
    """
    #===================================
    def __init__(self, action="W", beta=None, u0=1):
        self.beta  = beta
        self.u0    = u0
        self.action = action
        self.set_staples(u0)
    #===================================
    def set_staples(self, u0):
        """ Sets the `staples` needed to calculate the relevant contribution to the action
        corresponding to the link at site xx and mu direction;
        this is organized as Re Tr ( U_mu(xx) Gamma_mu(xx) );
        so we first calculate Gamma_mu(xx), which is common for all hits of the Metropolis update
        of the link, and then we tie U_mu(xx) and Gamma_mu(xx) using `ReTrTie`.
        """
        self.u0 = u0 # update u0 in case set_staples is called with a different value of u0
        if self.action=="W":
            self.calc_staples = lambda U, nn: self.calc_staples_Wilson(U, nn)
        elif self.action=="imp":
            self.calc_staples = lambda U, nn: self.calc_staples_improved(U, nn, u0**2)
        else:
            print("The action type is not specified correctly!")
    #===================================
    def fmt_action_details(self):
        if self.action=='W':
            fmt = "action = {}, beta = {}".format(self.action, self.beta)
        else:
            fmt = "action = {}, beta = {}, u0 = {}".format(self.action, self.beta, self.u0)
        return fmt
    #===================================
    @staticmethod
    def calc_staples_Wilson(numpy.ndarray U, tuple nn): # nn = (x,y,z,t,mu)
        cdef site XX = site(*nn[:4])
        cdef int mu = nn[4]
        cdef int nu, ind0, ind1, ind2, ind3, ind4, ind5
        cdef matrix Gamma0_mu = matrix(zeros=True)
        for nu in range(dim):
            if nu==mu: continue
            # We first determine the indices of the matrices that are going to be multiplied
            #                ---          @===
            #   Gamma0_mu:  |   |    +    |   |   
            #               @===           ---
            ind0 = index(XX,nu); XX+=nu
            ind1 = index(XX,mu); XX+=mu; XX-=nu
            ind2 = index(XX,nu); XX-=nu
            ind3 = index(XX,nu); XX-=mu
            ind4 = index(XX,mu);
            ind5 = index(XX,nu); XX+=nu
            # Now multiply the matrices and sum them
            Gamma0_mu += U[ind0] * U[ind1] * Adj(U[ind2])
            Gamma0_mu += Adj(U[ind5]) * U[ind4] * U[ind3]
        return Adj(Gamma0_mu)
    #===================================
    @staticmethod
    def calc_staples_improved(numpy.ndarray U, tuple nn, double u0_sq): # nn = (x,y,z,t,mu)
        #===============================
        # In the following cartoon
        #       1)  the horizontal axis refers to the \mu direction
        #       2)  the vertical axis refers to the \nu direction
        #       2)  "===" indicates the U_\mu (not is staples)
        #       3)  "---" and "|" indicate the links present in staples
        #       4)  The point shown by `@` shows the (t,x,y,z) site
        #                ---          @===
        #   Gamma0_mu:  |   |    +    |   |   
        #               @===           ---
        #                     --- ---        @=== ---
        #   Gamma1_mu:       |       |   +   |       |
        #                    @=== ---         --- ---
        #                             --- ---         ---@===
        #   Gamma2_mu:               |       |   +   |       |
        #                             ---@===         --- ---
        #                ---      @===
        #               |   |     |   |
        #   Gamma3_mu:  |   |  +  |   |   
        #               @===       ---
        #===============================
        cdef site XX = site(*nn[:4]) # shown by `@` in the above cartoon
        cdef int mu = nn[4]
        cdef int nu, ind0, ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8, ind9
        cdef matrix Gamma0_mu = matrix(zeros=True)
        cdef matrix Gamma1_mu = matrix(zeros=True)
        cdef matrix Gamma2_mu = matrix(zeros=True)
        cdef matrix Gamma3_mu = matrix(zeros=True)
        cdef matrix Gamma_mu  = matrix(zeros=True)
        for nu in range(dim):
            if nu==mu: continue
            #===========
            #                ---          @===
            #   Gamma0_mu:  |   |    +    |   |   
            #               @===           ---
            # determine the indices of the matrices that are going to be multiplied
            ind0 = index(XX,nu); XX+=nu
            ind1 = index(XX,mu); XX+=mu; XX-=nu
            ind2 = index(XX,nu); XX-=nu
            ind3 = index(XX,nu); XX-=mu
            ind4 = index(XX,mu);
            ind5 = index(XX,nu); XX+=nu
            # Now multiply the matrices and sum them
            Gamma0_mu += U[ind0] * U[ind1] * Adj(U[ind2])
            Gamma0_mu += Adj(U[ind5]) * U[ind4] * U[ind3]
            #===========
            #                     --- ---        @=== ---
            #   Gamma1_mu:       |       |   +   |       |
            #                    @=== ---         --- ---
            # determine the indices of the matrices that are going to be multiplied
            ind0 = index(XX,nu); XX+=nu
            ind1 = index(XX,mu); XX+=mu
            ind2 = index(XX,mu); XX+=mu; XX-=nu
            ind3 = index(XX,nu); XX-=mu
            ind4 = index(XX,mu); XX+=mu; XX-=nu
            ind5 = ind4
            ind6 = index(XX,nu); XX-=mu
            ind7 = index(XX,mu); XX-=mu
            ind8 = index(XX,mu)
            ind9 = index(XX,nu); XX+=nu
            # Now multiply the matrices and sum them
            Gamma1_mu += U[ind0] * U[ind1] * U[ind2] * Adj(U[ind3]) * Adj(U[ind4])
            Gamma1_mu += Adj(U[ind9]) * U[ind8] * U[ind7] * U[ind6] * Adj(U[ind5])
            #===========
            #                             --- ---         ---@===
            #   Gamma2_mu:               |       |   +   |       |
            #                             ---@===         --- ---
            # determine the indices of the matrices that are going to be multiplied
            XX-=mu
            ind0 = index(XX,mu)
            ind1 = index(XX,nu); XX+=nu
            ind2 = index(XX,mu); XX+=mu
            ind3 = index(XX,mu); XX+=mu; XX-=nu
            ind4 = index(XX,nu); XX-=nu
            ind5 = index(XX,nu); XX-=mu
            ind6 = index(XX,mu); XX-=mu
            ind7 = index(XX,mu)
            ind8 = index(XX,nu); XX+=nu
            ind9 = ind0; XX+=mu
            # Now multiply the matrices and sum them
            Gamma2_mu += Adj(U[ind0]) * U[ind1] * U[ind2] * U[ind3] * Adj(U[ind4])
            Gamma2_mu += Adj(U[ind9]) * Adj(U[ind8]) * U[ind7] * U[ind6] * U[ind5]
            #===========
            #                ---      @===
            #               |   |     |   |
            #   Gamma3_mu:  |   |  +  |   |   
            #               @===       ---
            # determine the indices of the matrices that are going to be multiplied
            ind0 = index(XX,nu); XX+=nu
            ind1 = index(XX,nu); XX+=nu
            ind2 = index(XX,mu); XX+=mu; XX-=nu
            ind3 = index(XX,nu); XX-=nu
            ind4 = index(XX,nu); XX-=nu
            ind5 = index(XX,nu); XX-=nu
            ind6 = index(XX,nu); XX-=mu
            ind7 = index(XX,mu)
            ind8 = index(XX,nu); XX+=nu
            ind9 = index(XX,nu); XX+=nu
            # Now multiply the matrices and sum them
            Gamma3_mu += U[ind0] * U[ind1] * U[ind2] * Adj(U[ind3]) * Adj(U[ind4])
            Gamma3_mu += Adj(U[ind9]) * Adj(U[ind8]) * U[ind7] * U[ind6] * U[ind5]
        Gamma_mu = Gamma0_mu*(5./3.) - (Gamma1_mu + Gamma2_mu + Gamma3_mu)/(12.*u0_sq)
        return Adj(Gamma_mu)


#==========================================================
class measurement_util(object):
    def __init__(self):
        pass
    #===================================
    @staticmethod
    def avg_plaq(U): 
        cdef int nu, mu
        cdef double result = 0
        cdef tuple xx
        cdef site XX
        for xx in ALL_sites():
            XX = site(*xx)
            for mu in range(dim):
                for nu in range(mu):
                    result += plaq(U, XX, mu,nu)
        return result/(num_sites)/(dim*(dim-1)/2) # averaging over all sites and planes
    #===================================
    @staticmethod
    def all_plaq(U): # to study the distribution of the plaquette values
        cdef int i=0
        cdef tuple xx
        cdef site XX
        results_array = np.zeros(num_sites* (dim*(dim-1)//2))
        for xx in ALL_sites():
            XX = site(*xx)
            for mu in range(dim):
                for nu in range(mu):
                    results_array[i] = plaq(U, XX, mu,nu)
                    i += 1
        return np.array(results_array).reshape(nx,ny,nz,nt, (dim*(dim-1)//2))
    #===================================
    @staticmethod
    def avg_munu_loops(numpy.ndarray U, int mu, int nu, int mu_step, int nu_step):
        """ 
        Calculates the renormalized (divided by nc) trace of all plananr Wilson loops in `mu-nu` plane
        with given steps in `mu` and `nu` directions and averages over them.
        
        The `mu-nu` loops are organized according to the following cartoon:: 

                   ---------<--------
                   |                |
                   v                ^ nu
                   |      mu        |
                   @------->---------

        where ``@`` is the starting point, which will be summed over all sites of the lattice.
        """
        cdef double result = 0
        cdef long long counter = 0
        cdef matrix l1, l2
        cdef site XX
        for xx in ALL_sites():
            #=============
            # l_1 = product of all links from xx to the opposite side of the rectangle from below
            XX = site(*xx)
            l1 = U[index(XX,mu)]; XX += mu
            for _ in range(1,mu_step):
                l1 *= U[index(XX,mu)]; XX += mu
            for _ in range(nu_step):
                l1 *= U[index(XX,nu)]; XX += nu
            #=============
            # l2 =  product of all links from xx to the opposite side of the rectangle from above
            XX = site(*xx)
            l2 = U[index(XX,nu)]; XX += nu
            for _ in range(1,nu_step):
                l2 *= U[index(XX,nu)]; XX += nu
            for _ in range(mu_step):
                l2 *= U[index(XX,mu)]; XX += mu
            #=============
            result += ReTr(l1 * Adj(l2))
            counter+= 1
        return result/nc/counter # averaging over all loops
    #===================================
    def avg_planar_loops(self, U, l1=1, l2=1): 
        """ Calculates the renormalized trace of all `l1 times l2` planar Wilson loops
        and averages over them."""
        result = 0
        for mu in range(dim):
            for nu in range(mu):
                result += self.avg_munu_loops(U, mu, nu, l1, l2)
                if l1==l2: continue
                result += self.avg_munu_loops(U, mu, nu, l2, l1)
        n_loops = dim*(dim-1)
        if l1==l2: n_loops/=2
        return result/n_loops
    #===================================
    def avg_planar_RxT_loops(self, U, R, T, temporal_dir=3):
        """ Calculates the renormalized trace of  `R times T` planar Wilson loops
        in all sites and space-time planes and averages over them."""
        return np.mean([self.avg_munu_loops(U, mu, temporal_dir, R, T) for mu in range(1,dim)])
    #===================================
    @staticmethod
    def calc_path_link(numpy.ndarray U, site XX, tuple path_tuple):
        """ Calculates a Wilson line along the path specified by `path_tuple`.
        Each element of `path_tuple` is itself a tuple (or a list) specifying a direction and the number of steps in that direction.
        """
        cdef matrix link = matrix(identity=True)
        cdef site QQ = (XX+0)-0 # define a new variable 
        cdef int mu, steps
        for mu,steps in path_tuple:
            if steps>0:
                for _ in range(steps):
                    link *= U[index(QQ,mu)]
                    QQ   += mu
            else:
                for _ in range(abs(steps)):
                    QQ   -= mu
                    link *= Adj(U[index(QQ,mu)])
        return link, QQ
    #===================================
    def avg_RxT_loops(self, U, R_tuple_list, max_T): 
        """ Calculates renormalized (divided by nc) trace of all Wilson loops in a `RxT` space-time surfase
        where `R_tuple` specifies the spatial path         and `max_T` the temporal duration.

        Parameters:
            - ``U`` (nd.array): a link configuration for calculating the `RxT` loops
            - ``R_tuple_list`` (tuple or list): is either a tuple or a list of tuples.
                Each tuple has elements specifying a spatial direction and number of steps in that direction.
                For instance `R_tuple = ((0,2),(1,1),(2,0))` means `2,1,0` steps in `x,y,z` directions, respectively.
                When `R_tuple_list` is a list of typles, it means to take an average over all tuples inside the list.
                This is useful for averaging over all spatial rotations of a tuple.
            - ``max_T``: specifies the range for steps in termporal direction.

        Output:
            - an array with `1+max_T` elements each of which corresponds to number of steps in the temporal direction \
            ranging from `T=0` to `T=max_T`. Note that when `T=0`, the outut is simply one.

        """
        #===========================
        # RxT loop is organized according to following cartoon 
        #           ---------<--------
        #           |    l3^\dagger  |
        # l4^dagger v                ^ l2
        #           |      l1        |
        #           -------->---------
        # R|__
        #   T
        #===========================
        if isinstance(R_tuple_list,list):
            avg = 0
            for R_tuple in R_tuple_list:
                avg = avg + self.avg_RxT_loops(U, R_tuple, max_T)
            return avg/len(R_tuple_list)
        elif isinstance(R_tuple_list,tuple):
            R_tuple = R_tuple_list
        #===========================
        cdef int counter = 0
        cdef matrix l1,l2,l3,l4
        cdef site XX
        cdef int T
        calc_path_link = self.calc_path_link
        RxT = np.zeros(1+max_T)
        for xx in ALL_sites():
            XX_l1 = site(*xx)
            l1 = matrix(identity=True)
            l3 = matrix(identity=True)
            l4, XX_l3 = calc_path_link(U,XX_l1,R_tuple)
            RxT[0] += 3 # Tr( Identity ) at T=0
            for T in range(1,1+max_T):
                l1 *= U[index(XX_l1,3)]; XX_l1+=3 # mu=3 implies time
                l3 *= U[index(XX_l3,3)]; XX_l3+=3
                l2, _ = calc_path_link(U,XX_l1,R_tuple)
                RxT[T] += ReTr( l1 * l2 * ( l4 * l3 ).H )
            counter += 1
        return RxT/nc/counter
    #===================================
    @staticmethod
    def avg_polyakov_loops(U):
        """ Calculates the average of Polyakov lools `<L>`.
        """
        cdef int x,y,z,t=0
        cdef int counter=0
        cdef double polyakov=0
        cdef site XX
        ALL_spatialsites = lambda: itertools.product(range(nx),range(ny),range(nz))
        for x,y,z in ALL_spatialsites():
            XX = site(x,y,z,t)
            l  = U[index(XX,3)] # mu=3 implies time
            for T in range(1,nt):
                XX +=3 
                l  *= U[index(XX,3)]
            polyakov += ReTr(l)
            counter += 1
        polyakov /= (counter*nc)
        return polyakov

#==========================================================
cdef inline double plaq(numpy.ndarray U, site XX, int mu, int nu):
    """ The normalizaed trace of the plaquette operator, which involves the product of link variables
    around the smallest square at site `xx = (x,y,z,t)` in the `mu-nu` plane::
        P_mu_nu = 1/3 Re Tr (  U_mu(x) U_nu(x+mu) U_mu(x+nu)^\dagger U_nu(x)^\dagger )
    """
    return 1./nc * ReTr( U[index(XX,mu)] * U[index(XX+mu,nu)] * U[index(XX+nu,mu)].H * U[index(XX,nu)].H )

#==========================================================
# Do not remove cdef from nt, etc. Otherwise index_(...) will be very slow
cdef int nx, ny, nz, nt
cdef int dim
cdef int nc

cdef int stride0, stride1, stride2, stride3

cdef int num_sites, num_links=0

ALL_sites = lambda: []
ALL_links = lambda: []

# the `nogil` versions of the following functions are much faster when profiling the code,
# but do not have any effects in practice (without profiling)!

cpdef inline int index_(int x0, int x1, int x2, int x3, int mu) nogil:
    """ maps the point (x0,x1,x2,x3) to a unique integer index. Indices must not be negative. """
    return x0*stride0 + x1*stride1 + x2*stride2 + x3*stride3 + mu 

cpdef int index(site XX, int mu):
    """ similar to :meth:`gauge_tools._gtcore.index_`
    except that a lattice site is specified by an instance of :class:`gauge_tools.lib._site.site`. """
    return index_(XX.e[0], XX.e[1], XX.e[2], XX.e[3], mu)
    
#==========================================================

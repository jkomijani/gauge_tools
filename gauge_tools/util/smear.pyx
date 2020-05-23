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

from ..lib._matrix cimport matrix, ReTr, ReTrTie, matexp
from ..lib._site cimport site
from ..lib._site cimport link_index as index
from ..lib._site import link_index_ as index_

from .._gtcore import gauge_field as _gauge_field

import numpy as np
cimport numpy

pi = np.pi

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
class smear(object):
    """ Utilities for smearing gauge links. Currently only gradient flow and APE smearing are supported for ``nc=3``.
    """
    #===================================
    def __init__(self, param):
        """ The input is an instance of :class:`gauge_tools.param`,
        which will be used to set the global parameters.
        """
        set_param(param)
    #===================================
    @staticmethod
    def gradient_flow(numpy.ndarray U, calc_staples, double max_flowtime=1., double eps=0.01, fname='', clover=False, automatic_break=True):
        """
        .. _arXiv:1401.2441: https://arxiv.org/abs/1401.2441
        .. _arXiv:1006.4518: https://arxiv.org/abs/1006.4518

        performs the gradient flow to the links and ***replaces*** the original links with the smeared ones.
        We follow Eq. (7) of [`arXiv:1401.2441`_] and Appenix C of [`arXiv:1006.4518`_].
        For the improved actions, the tadpole improvement factor can be an issue.
        At least naively, it seems that one should drop the factor because as flow time increases
        the average of plaquettes gets closer to 1. On the other hand it seems that this factor is important
        al least at small flow times. In any case, needs to be investigated.

        Parameters:
            - ``U``     (numpy.ndarray):    gauge links.
            - ``calc_staples``:  a method to calculate the staples needed for the gradient flow.
            - ``max_flowtime``  (double):   the limit value of flow time.
            - ``eps``           (double):   the flow time steps.
            - ``fname``         (str):      if not an empty string, a file name to dump details.
            - ``clover``        (bool):     for calculating ``<E>`` (***Under Investigation***)
            - ``automatic_break`` (bool):   if `True` stops the flow if ``t^2 <E>`` reaches `1/2`.
        """
        cdef int i=0, n_steps = round(max_flowtime/eps)
        cdef double tau
        cdef matrix omg, Znn
        cdef matrix I = matrix(identity=True)
        Z = np.array([matrix(zeros=True) for _ in range(num_links)])
        e_v0 = np.zeros(n_steps+1)
        e_v1 = np.zeros(n_steps+1)
        e_v2 = np.zeros(n_steps+1)
        topo = np.zeros(n_steps+1)
        plaq = np.zeros(n_steps+1)
        def RungeKutta_step(int RK_ind):
            cdef int ind
            for nn in ALL_links():
                ind = index_(*nn)
                omg = U[ind] * calc_staples(U, nn) # omega
                Znn = omg.anti_hermitian_traceless()*(-eps)
                # here Z is ``eps`` times  the Z defined in equation C.1 of [arXiv:1006.4518]
                if RK_ind==1:
                    Z[ind] = Znn*(1./4.)
                elif RK_ind==2:
                    Z[ind] = Znn*(8./9.) - Z[ind]*(17./9.)
                elif RK_ind==3:
                    Z[ind] = Znn*(3./4.) - Z[ind]
                else:
                    raise Exception("Oops: the RK_ind must be either 0, 1, or 2")
            for ind in range(num_links):
                U[ind] = matexp( Z[ind] ) * U[ind]
        e_v0[0],e_v1[0],e_v2[0],topo[0],plaq[0] = energy_topocharge_plaq(U,clover=clover)
        if fname!='':
            fw = open(fname, 'a')
            message  = "#flow-time   energy_density_v0    energy_density_v1    energy_density_v2    topological-charge    average-plaquette\n"
            message += "%.3f\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n"%(0,e_v0[0],e_v1[0],e_v2[0],topo[0],plaq[0])
            fw.write(message)
        for i in range(1,n_steps+1):
            RungeKutta_step(1)
            RungeKutta_step(2)
            RungeKutta_step(3)
            e_v0[i],e_v1[i],e_v2[i],topo[i],plaq[i] = energy_topocharge_plaq(U,clover=clover) # e: energy, q: topological charge, p: plaquette
            tau = eps*i 
            if fname!='':
                message = "%.3f\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n"%(tau,e_v0[i],e_v1[i],e_v2[i],topo[i],plaq[i])
                fw.write(message)
            if automatic_break and (tau**2*e_v0[i] > 0.5):
                break
        if fname!='':
            fw.close()
        mydict = dict(flowtime=eps*np.arange(i+1), e_v0=e_v0[:i+1], e_v1=e_v1[:i+1], e_v2=e_v2[:i+1], topo=topo[:i+1], plaq=plaq[:i+1])
        mydict['a'] = calc_lattice_spacing(**mydict)
        return mydict
    #===================================
    @staticmethod
    def ape_smear(U, int n_smear=4, double eps=1/12., double u0=1., space_only=True, project_SU3=True):
        """ performs the APE smearing to the links and ***replaces*** the original links with the smeared ones.
        By default, the smearing is done only for spatial links, but there is an option to smear
        the temporal links too. Also by default the smeared links are projected to SU(3), but one can disable it.
        """
        #============
        # First job is to determine how many times the smearing needs to be performed
        if n_smear>1:
            smear.ape_smear(U, n_smear=1, eps=eps, u0=u0, space_only=space_only, project_SU3=project_SU3)
            for n in range(1,n_smear):
                smear.ape_smear(U, n_smear=1, eps=eps, u0=u0, space_only=space_only, project_SU3=project_SU3)
            return n_smear
        #============
        U_smeared  = _gauge_field().U
        smearing_directions    = range(dim-1) if space_only else range(dim)
        nonsmearing_directions = [3] if space_only else []
        nstaples   = 2*len(smearing_directions)-2
        cdef double c1 = (1 - nstaples*eps)
        cdef double c2 = eps/u0**2
        cdef int nu
        cdef Py_ssize_t ind, ind0, ind1, ind2, ind3, ind4, ind5
        cdef matrix Gamma0_mu
        cdef site XX
        for xx in ALL_sites():
            XX = site(*xx)
            for mu in smearing_directions:
                Gamma0_mu = matrix(zeros=True)
                for nu in smearing_directions:
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
                    Gamma0_mu += U[ind0] * U[ind1] * U[ind2].H
                    Gamma0_mu += U[ind5].H * U[ind4] * U[ind3]
                ind = index(XX,mu)
                U_smeared[ind] = c1 * U[ind] + c2 * Gamma0_mu
                if project_SU3:
                    U_smeared[ind] = U_smeared[ind].project_SU3()
            for mu in nonsmearing_directions:
                ind = index(XX,mu)
                U_smeared[ind] = U[ind] # set the links in the time direction to the non-smeared ones
        for ind in range(len(U)): # insert the smeared links in `U`
            U[ind] = U_smeared[ind]
    #===================================
        
#==========================================================
def energy_topocharge_plaq(numpy.ndarray U, clover=False):
    """
    returns three values for the energy and also the topological charge and the average plaq
    corresponding to the input links ``U``.
    
    Parameters:
        - ``U`` (numpy.ndarray): the gauge links
        - ``clover`` (bools):    if `True` uses the average over 4 plaquettes in a plane about\
                                 every site to calculate ``E``.

    ***NOTE:*** at least for a lattice with ``a~0.25`` fm, the colver form seems not working;
    requires further investigation. ***This function will change in next versions.***

    """
    #.. _arXiv:hep-lat/0203008: https://arxiv.org/abs/hep-lat/0203008  
    #
    #first calculates ``i 4 g F_{mu,nu}`` in the lattice units.
    #For details of algebra see Eq. (25) of [`arXiv:hep-lat/0203008`_].
    #
    #Then uses F_{mu,nu} to calculate energy and topological charge.
    #We also separately calculate the average of plqaquette.
    #
    #We use three different methods to calculate the energy density.
    #.... we average over 4 plaquettes in a plane about every site
    #as shown in Fig. 1 of [http://arxiv.org/abs/1006.4518].
    #This would yeild a symmetric definition of the field strength tensor.


    cdef site XX, YY, ZZ, WW 
    cdef matrix mat
    cdef numpy.ndarray[object, ndim=1] i4gF_munu_site
    cdef double e_v0=0, e_v1=0, e_v2=0, topo_charge=0, avg_plaq=0
    i4gF_munu_site_v1 = np.array([matrix(zeros=True) for _ in range(6)])
    if clover:
        i4gF_munu_site_v2 = np.array([matrix(zeros=True) for _ in range(6)]) # the clover version
    else:
        i4gF_munu_site_v2 = i4gF_munu_site_v1
    planes = [(0, 0,1), (1, 0,2), (2, 0,3), (3, 1,2), (4, 1,3), (5, 2,3)]
    for xx in ALL_sites(): # xx = (x,y,z,t)
        XX = site(*xx)
        for plane_ind,mu,nu in planes:
            YY = XX - mu
            ZZ = XX - nu
            WW = YY - nu
            mat  = U[index(XX,mu)]   * U[index(XX+mu,nu)]   * U[index(XX+nu,mu)].H  * U[index(XX,nu)].H
            avg_plaq += ReTr(mat)
            i4gF_munu_site_v1[plane_ind] = mat.anti_hermitian_traceless()
            if clover:
                mat += U[index(XX,nu)]   * U[index(YY+nu,mu)].H * U[index(YY,nu)].H     * U[index(YY,mu)]
                mat += U[index(YY,mu)].H * U[index(WW,nu)].H    * U[index(WW,mu)]       * U[index(ZZ,nu)]
                mat += U[index(ZZ,nu)].H * U[index(ZZ,mu)]      * U[index(ZZ+mu,nu)]    * U[index(XX,mu)].H
                i4gF_munu_site_v2[plane_ind] = mat.anti_hermitian_traceless()
        topo_charge -= ReTrTie(i4gF_munu_site_v2[0], i4gF_munu_site_v2[5])
        topo_charge += ReTrTie(i4gF_munu_site_v2[1], i4gF_munu_site_v2[4])
        topo_charge -= ReTrTie(i4gF_munu_site_v2[2], i4gF_munu_site_v2[3])
        for i in range(6):
            e_v1 -= ReTrTie(i4gF_munu_site_v1[i], i4gF_munu_site_v1[i])
            if clover:
                e_v2 -= ReTrTie(i4gF_munu_site_v2[i], i4gF_munu_site_v2[i])
    # Normalization:
    # energy_density: 1/4**2 from the coeff 4 in i4gF; 1/2 for 1/2 F_munu F_munu; 2 for permutation of indices
    # topo_charge:    1/4**2 from the coeff 4 in i4gF; 1/(32 * pi**2) from definition; 8 for permutation of indices
    avg_plaq          /= (nc * num_sites * dim*(dim-1)/2)
    e_v0  = 6 * (1-avg_plaq) * dim*(dim-1)/2 # note that 6 comes from beta * g0**2
    e_v1 /= (num_sites)
    if clover:
        e_v2 /= (4**2 * num_sites)
        topo_charge  /= (4**2 * 4*pi**2)
    else:
        e_v2 = e_v1
        topo_charge  /= (4*pi**2)
    return e_v0, e_v1, e_v2, topo_charge, avg_plaq
    #===================================

#==========================================================
def calc_lattice_spacing(w0=0.1714, c0=0.3, e_def='e_v0', **kwargs):
    """
    .. _arXiv:1503.02769: https://arxiv.org/abs/1503.02769

    returns the lattice spacing ``a`` determined using the ``w0`` parameter.
    As the default value, we use ``w0=0.1714`` fm,
    which is the central value given in [`arXiv:1503.02769`_].

    Parameters:
        - ``w0``    (float): the phaysical value of ``w0`` parameter in `fm`.
        - ``c0``    (float): the constant corresponding to ``w0``.
        - ``e_def`` (str):   specifies the defintion of ``<E>`` used for scale setting;\
                            the default value, ie 'e_v0' corresponds to ``<E>`` from avegage plaquettes.\
                            Other options are 'e_v1' and 'e_v2'.\
                            See :meth:`gauge_tools.util.smear.energy_topocharge_plaq` to find out the differences.
    """ 
    t = kwargs['flowtime']  # must be a numpy ndarray
    E = kwargs[e_def]       # must be a nunmy ndarray
    eps = t[1]-t[0]
    g = t**2 * E
    t = (t[1:]+t[:-1])/2    # average of two times
    der_g = t * (g[1:]-g[:-1])/eps
    try:
        from scipy.interpolate import interp1d
        f = interp1d(der_g, t, kind='cubic')
        w0_by_a_sq = f(c0) # in lattice units
        a = w0/w0_by_a_sq**0.5
    except:
        a = None
    return a

#==========================================================


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
This file contains examples that correspond to the exercises suggested in these `lecture notes`_.
.. _lecture notes: https://arxiv.org/abs/hep-lat/0506036v1
"""

import gvar as gv
import numpy as np
import pickle
import time

try:
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    PLOTS = True
except:
    PLOTS = False
    print("Cannot generate plots because matplolib did not get imported correctly")

#==========================================================
#==========================================================
# Example 1:: 
#   Generate gauge configurations and compute Wilson loops.
#==========================================================
def generate_ensemble(size_list=[8,8,8,8], beta=5.5, action='W', u0=1, n_therm=200, n_cfgs=5, n_skip=50, eps_hit=0.24,
        update_u0=False, ens_tag='', **lat_kwargs):
    """
    This function generates a new ensemble of gauge configurations.
    
    The default values of the parameters correspond to the exercise on page 35 of the `lecture notes`_.

    Parameters:
        - ``size_list`` ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``action``    (str):      `'W'` stands for Wilson action and `'imp'` stand for an improved action.
        - ``beta``      (float):    the lattice beta; with our convension this beta already contains the tadpole improvement factor.
        - ``u0``        (float):    the `u0` used for tadpole improvement (irrelevant for the non-improved action).
        - ``n_therm``   (int):      number of sweeps for thermalizing the gauge configurations.
        - ``n_cfgs``    (int):      number of gauge configurations requested to be generated and saved.
        - ``n_skip``    (int):      number of sweeps between two saved configurations.
        - ``update_u0`` (bool):     if `True` updates the value of `u0` in the process of thermalizing.
        - ``ens_tag``   (str):      a unique tag (label) describing the ensemble:\
                                    this tag is used to automatically create a file name for saving (loading) a gauge configuration;\
                                    `fname = "{}{}.npy".format(ens_tag, ind_cfg)`, where `ind_cfg` is explained below.

    A note on `ind_cfg`:
        this index counts the number of all sweeps including the sweeps for thermalization. With the default values\
        of the parameters of this function we will have `ind_cfg \in [100,150,200,250,300]`
        for 5 gauge configurations that are going to be saved. (`ind_cfg` is nothing but `sweep_cntr` in the code.) 

    Usage:
        (1) For generating 5 gauge configurations with Wilson action, one can simply use::

                >>> import gauge_tools as gt
                >>> gt.examples.generate_ensemble(n_cfgs=5, ens_tag='W_')

            This generates 5 gauge configurations and saves each of them to a seperate file in the current directory.
            One can of course change default values of the parameters of the function.

        (2) For generating 5 gauge configurations with the improved action, one can change the parameters to something like::

                >>> import gauge_tools as gt
                >>> gt.examples.generate_ensemble(beta=1.719/0.797**4, u0=0.797, n_cfgs=5, ens_tag='imp_')

        (3) One can also let the code dynamically determine the value of `u0`. To this end, for instance one can use

                >>> import gauge_tools as gt
                >>> gt.examples.generate_ensemble(beta=1.719/0.797**4, update_u0=True, n_cfgs=5, ens_tag='imp_')

    """
    fname_lambda = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, beta=beta, action=action, u0=u0, **lat_kwargs)
    lat.MC.thermalize(n_therm, fname_lambda=fname_lambda, update_u0=update_u0, eps_hit=eps_hit)
    lat.MC.gen_cfgs(n_skip=n_skip, n_cfgs=n_cfgs-1, fname_lambda=fname_lambda, eps_hit=eps_hit)
    generate_ensemble.lat = lat

#==========================================================
def expand_ensemble(size_list=[8,8,8,8], beta=5.5, action='W', u0=1, load_ind=None, n_cfgs=15, n_skip=50, eps_hit=0.24,
        ens_tag='', **lat_kwargs):
    """
    This function expands an existing ensemble by loading the last (saved) configuration
    and updating it to generate more gauge configurations.

    Parameters:
        similar to the parameters of :meth:`gauge_tools.examples.generate_ensemble` except that instead of `n_therm` and `update_u0`
        there is a new parameter called `load_ind` which is the index (`sweep_cntr`) of the gauge configuration
        that should be loaded to start with.

    Usage:
        (1) For generating (and saving) 15 more gauge configurations with Wilson action, one can use:: 

                >>> import gauge_tools as gt
                >>> gt.examples.expand_ensemble(load_ind=400, n_cfgs=15, ens_tag='W_')

            where it is assumed that the index (`ind_cfg`) of the last saved configurations is 400.
    """
    fname_lambda = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    load_kwargs = dict(mode="load&update", load_fname=fname_lambda(load_ind), load_ind=load_ind)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, beta=beta, action=action, u0=u0, **lat_kwargs)
    lat.MC.gen_cfgs(n_skip=n_skip, n_cfgs=n_cfgs, fname_lambda=fname_lambda, eps_hit=eps_hit, **load_kwargs)
    expand_ensemble.lat = lat

#==========================================================
def measure_Wilson_loops(size_list=[8,8,8,8],cfgs_list=range(200,1200,50), binsize=1,
        ens_tag='', **lat_kwargs):
    """
    This function computes Monte Carlo averages of `a x a` and `a x 2a` Wilson loops
    exploiting the gauge configurations generated with functions :meth:`gauge_tools.examples.generate_ensemble` and :meth:`gauge_tools.examples.expand_ensemble`.

    This corresponds to the exercise on page 35 of the `lecture notes`_.
    For the exercise simply use::
        
                >>> import gauge_tools as gt
                >>> gt.examples.measure_Wilson_loops(cfgs_list=range(200,1200,50), ens_tag='W_')

    The output looks like::

                Calculating averages of 'axa' and 'ax2a' Wilson loops:
                'axa':0.49774(96), 'ax2a':0.2606(13)

    Parameters:
        - ``size_list`` ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``cfgs_list`` (list):     a list of indices of configurations to be loaded and used for the Monte Carlo averages.
        - ``binsize``   (int):      bin size used for estimation of uncertainties. 
        - ``ens_tag``   (str):      a unique tag (label) describing the ensemble:\
                                    for details see the parameters of function :meth:`gauge_tools.examples.generate_ensemble`. 
    """
    fname_lambda = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    calc_a_a   = lambda U: lat.meas.avg_planar_loops(U, l1=1, l2=1)
    calc_a_2a  = lambda U: lat.meas.avg_planar_loops(U, l1=1, l2=2)
    fn = lambda U: (calc_a_a(U), calc_a_2a(U))
    print("Calculating averages of 'axa' and 'ax2a' Wilson loops:")
    T1  = time.time()
    avg = lat.MC.eval_fn(fn, cfgs_list, fname_lambda, binsize=binsize, avg_data=True)
    print("'axa':{}, 'ax2a':{}".format(*avg), end=''),
    print("\t(#TIME = {:.4g})".format(time.time()-T1))
    measure_Wilson_loops.avg = avg
    measure_Wilson_loops.lat = lat

#==========================================================
#==========================================================
# Example 2::
#   Compute the static potential using generated gauge configurations (in Example 1).
#==========================================================
def static_potential(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), max_R=4.1, max_T=5,
        ens_tag='', smear_tag='', do_smear=True, smearing_dict={'u0':0.84},
        figname='', **lat_kwargs):
    """
    This function computes the statice potential from Monte Carlo averages of `R times T` Wilson loops
    for different values of `R` and `T`.

    This corresponds to the exercise on page 37 of the `lecture notes`_.
    Here we smear the links first and then calculate the Wilson loops.
    One can simply disable the smearing by setting the option `do_smear=False`.
    Note that one can first run :meth:`gauge_tools.examples.ape_smear` to smear the links, save them, and then use them here.
    With a combination of the options `ens_tag` and `smear_tag` one can specify the links to be read,
    and with the option `do_smear` one can request an in-place smearing. In this case, one can control the
    smearing parameters by the `smearing_dict`. The latter one is used by default.
    For the exercise simply use::
       
                >>> import gauge_tools as gt
                >>> gt.examples.static_potential(cfgs_list=range(200,1200,50), ens_tag='W_', figname='static_W_smear4.pdf')

    Parameters:
        - ``size_list``     ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``cfgs_list``     (list):     a list of indices of configurations to be loaded for further calculations.
        - ``ens_tag``       (str):      a unique tag (label) describing the ensemble:\
                    for details see the parameters of :meth:`gauge_tools.examples.generate_ensemble`.
        - ``smear_tag``     (str):      useful if one is going to use the already saved smeared links;\
                    for details see the parameters of :meth:`gauge_tools.examples.ape_smear`.
                                        Ignore this if the smearing is going to be done in place.
        - ``do_smear``      (bool):     For smearing the links before calculating the Wilson loops;\
                                        the default value is `True`.
        - ``smearing_dict`` (dict)      a dictionary to control the smearing parameters;\      
                                        for available options see :meth:`gauge_tools.examples.ape_smear`.
        - ``max_R``         (float):    calculates the static potential for distances `\le max_R`.
        - ``max_T``         (int):      calculates the static potential for all times `\le max_T`.
        - ``figname``       (str):      if not an empyt string, creates a figure and saves it as a pdf in `figname`.
 
    """
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    #===============
    fname_load = lambda ind_cfg: "{}{}{}.npy".format(ens_tag, smear_tag, n_cfg)
    fname_save = "{}{}{}.p".format(ens_tag, smear_tag, 'Wilson_loops')
    #===============
    dset = []
    range_T = range(1,1+max_T)
    range_R, spatial_paths = define_paths(max_R)
    func = lambda U: [lat.meas.avg_RxT_loops(U, path_R, max_T) for path_R in spatial_paths]
    #===============
    for n_cfg in cfgs_list:
        T1 = time.time()
        U  = lat.GF.load_cfg(fname_load(n_cfg))
        if do_smear:
            lat.smear.ape_smear(U, **smearing_dict) # the smearing is performed on `U`
        dset.append(func(U))
        print(" RxT is evaluated for cfg={} (#TIME = {:.4g})".format(n_cfg, time.time()-T1))
    W = gv.dataset.avg_data(dset)
    pickle.dump(dict(r=range_R, Wmean=gv.mean(W), Wcov=gv.evalcov(W)), open(fname_save, 'wb'))
    if figname!='' and PLOTS:
        V = gv.log(W[:,:-1] / np.roll(W, -1,axis=1)[:,:-1])
        plot_static_potential(range_R, V, figname=figname)
    static_potential.lat  = lat

#==========================================================
def define_paths(max_R):
    """
    Calculates all paths from the origin in the x-y-z volume subject to the following conditions:
            1) the total distance from the origin is not bigger than max_R.
            2) the number of steps in x-y or x-z or y-z planes are equal; e.g. (2,2,1) steps in (x,y,z) direction.
    Each path is shown by a tuple of at most three tuples e.g.
            ((0,s0),(1,s1),(2,s2))
    which means `s0,s1,s2` steps in the `0,1,2` directions, respectively. And
            ((0,s0),)
    means a path with only `s0` steps in the `0` direction.
    Here we group together the paths based on the cubic symmetry; e.g. (1,0,0) ~ (0,1,0) under pi/2 rotation of the x-y plane.
    We put the group into a list, and we organize the output as a list of lists, where each list contains a group of paths invariant
    under the cubic symmetry.

    Output:
        range_R         (list of dinstance):  the distance from the origin of the corresponding group in spatial_paths
        spatial_paths   (list of lists):      as discribed above.
    """
    range_R = []
    spatial_paths = []
    Tpl = lambda x,y,z: ((0,x),(1,y),(2,z))
    #=============
    # Firts group: the path is parallel to one of the axese.
    rot_00a = lambda a: [((0,a),), ((1,a),), ((2,a),)]
    #=============
    # second group: the path is in a plane (e.g x-y plane) with equal steps in each direction in the plane (e.g. x and y directions).
    rot_0aa = lambda a: [((0,a),(1,a)),  ((1,a),(2,a)),  ((2,a),(0,a)), \
                         ((0,a),(1,-a)), ((1,a),(2,-a)), ((2,a),(0,-a))]
    #=============
    # third group: all steps (in x, y, and z directions) are equal but can be postive or negative.
    rot_aaa = lambda a: [Tpl(a,a,a), Tpl(a,-a,a), Tpl(a,a,-a), Tpl(a,-a,-a)]
    #=============
    # fourth group: a part of the path in a plane (e.g x-y plane) has equal steps in each direction in the plane (e.g. x and y directions),
    #               but there is another part with steps not equal to...
    rot_baa = lambda b,a: [Tpl(a,a,b),  Tpl(a,b,a),  Tpl(b,a,a), 
                           Tpl(a,-a,b), Tpl(a,-b,a), Tpl(b,-a,a), 
                           Tpl(a,a,-b), Tpl(a,b,-a), Tpl(b,a,-a)] # `a` is repeated twice and `a \neq b`.
    #=============
    for x in range(1+int(max_R)):
        for y in range(1+int((max_R**2-x**2)**0.5)):
            if y>x: continue
            for z in range(1+int((max_R**2-x**2-y**2)**0.5)):
                if z>y: continue
                # Note that  `z \le y \le x`, therefore either `z=y=x` or `z<x`.
                # We want to average over all paths that are similar under a rotations by pi/2 degrees (cubic symmetry),
                # so we put each set of similar paths in a list so that we can take an average over them.
                if x+y+z==0:
                    continue              # x=y=z=0
                elif x>y and y>z:
                    continue              # x>y>z
                elif x==y and y==z:
                    mylist = rot_aaa(x)   # x=y=z > 0
                elif y==z and z==0:
                    mylist = rot_00a(x)   # x > y=z=0
                elif y==z and z>0:
                    mylist = rot_baa(x,y) # x > y=z > 0
                elif y==x and z==0:
                    mylist = rot_0aa(x)   # x=y > z=0
                elif y==x and z>0:
                    mylist = rot_baa(z,x) # x=y > z > 0
                else:
                    pass # the program won't reach here since `z \le y \le z`. 
                spatial_paths.append(mylist)
                range_R.append((x**2+y**2+z**2)**0.5)
    return range_R, spatial_paths

#==========================================================
def plot_static_potential(r, V, figname='', T_snap=2):
    fig = plt.figure(figsize=(4,4))
    Y_mean = [V_r[T_snap] for V_r in gv.mean(V)]
    Y_sdev = [V_r[T_snap] for V_r in gv.sdev(V)]
    plt.errorbar(r, Y_mean, Y_sdev, fmt='.', capsize=4)
    plt.xlabel(r"$R/a$"); plt.ylabel(r"$V(R)|_{%d,%d}$"%(T_snap,1+T_snap))
    plt.xlim([min(r)-0.2,max(r)+0.2])
    plt.grid()
    plt.tight_layout()
    if figname!='':
        fig.savefig(figname,format="pdf")
    
#==========================================================
def ape_smear(size_list=[8,8,8,8], cfgs_list=range(200,1200,50),
        n_smear=4, eps_smear=1/12., u0=0.84, space_only=True, project_SU3=True,
        ens_tag='', smear_tag='smear4_', **lat_kwargs):
    """
    This function loads the gauge configurations generated
    with functions :meth:`gauge_tools.examples.generate_ensemble` and :meth:`gauge_tools.examples.expand_ensemble`,
    smears them and saves them.

    This corresponds to the exercise on page 37 of the `lecture notes`_.
    For the exercise simply use::

                >>> import gauge_tools as gt
                >>> gt.examples.ape_smear(cfgs_list=range(200,1200,50), ens_tag='W_')

    Parameters:
        - ``size_list``     ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``cfgs_list``     (list):     a list of indices of configurations to be loaded for smearing.
        - ``n_smear``       (int):      number of smearings.
        - ``eps_smear``     (float):    the `eps` used for smearing.
        - ``u0``            (float):    the `u0` used for tadpole improvement.
        - ``ens_tag``       (str):      a unique tag (label) describing the ensemble:\
                for details see the parameters of function :meth:`gauge_tools.examples.generate_ensemble`. 
        - ``smear_tag``     (str):      a string used to construct a file name for saving a smeared gauge configuration;\
                                        see below for clarification. 
        - ``space_only``    (bool):     the default value is `True` indicating that the smearing is going to be done only in spatial planes.\
                                        If set to `False`, the smearing will be performed for termpotal links too,\
                                        and also includes spatial-temporal planes.
        - ``project_SU3``   (bool):     the default value is `False` indicating that the smeared links are projected to SU(3) in the end.\
                                        Note that the algorithm used here works only for SU(3) gauges. 
   
    A note on the file names:
        from the parameter `ens_tag` this function constructs two file names:
            - to load an existing configuration:
                    `fname_load = "{}{}.npy".format(ens_tag, ind_cfg)`.
            - to save the smeared configuration:
                    `fname_save = "{}{}{}.npy".format(ens_tag, smear_tag, ind_cfg)`.

    """
    fname_load = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    fname_save = lambda ind_cfg: "{}{}{}.npy".format(ens_tag, smear_tag, ind_cfg)
    smearing_dict = dict(eps=eps_smear, n_smear=n_smear, u0=u0, space_only=space_only, project_SU3=project_SU3)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    for n_cfg in cfgs_list:
        T1 = time.time()
        U  = lat.GF.load_cfg(fname_load(n_cfg))
        lat.smear.ape_smear(U, **smearing_dict) # the smearing is performed on `U`
        np.save(fname_save(n_cfg), U)
        print(" cfg={} is smeard and saved (#TIME = {:.4g})".format(n_cfg, time.time()-T1))
    ape_smear.lat = lat

#==========================================================
#==========================================================
# Example 3::
#   Fix the gauge of a configuration to the Landau gauge.
#==========================================================
def Landau_gauge(size_list=[8,8,8,8], cfgs_list=range(200,1200,50),
        max_itr=1000, gaugefix_tol=1e-9, fname_output='',
        ens_tag='', gauge_tag='Landau_', **lat_kwargs):
    """
    This function loads the gauge configurations generated
    with functions :meth:`gauge_tools.examples.generate_ensemble` and :meth:`gauge_tools.examples.expand_ensemble`,
    fixes the gauge to the Landau gauge them and saves them.

    Parameters:
        - ``size_list``     ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``cfgs_list``     (list):     a list of indices of configurations to be loaded for gauge fixing.
        - ``max_itr``       (int):      maximum number of iteration in gauge fixing.
        - ``gaugefix_tol``  (float):    the tolerance to hault the process of gauge fixing.
        - ``ens_tag``       (str):      a unique tag (label) describing the ensemble:\
                for details see the parameters of function :meth:`gauge_tools.examples.generate_ensemble`. 
        - ``gauge_tag``     (str):      a string used to construct a file name for saving the gauge-fixed configuration;\
                                        see below for clarification. 
        - ``fname_output``  (string):   if not an empty string, dumps the details of gauge fixing in `fname_output`.

    A note on the file names:
        from the parameter `ens_tag` this function constructs two file names:
            - to load an existing configuration:      
                    `fname_load = "{}{}.npy".format(ens_tag, ind_cfg)`.
            - to save the gauge fixed configuration:
                    `fname_save = "{}{}{}.npy".format(ens_tag, gauge_tag, ind_cfg)`.

    """
    fname_load = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    fname_save = lambda ind_cfg: "{}{}{}.npy".format(ens_tag, gauge_tag, n_cfg)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    for n_cfg in cfgs_list:
        T1 = time.time()
        U  = lat.GF.load_cfg(fname_load(n_cfg))
        if fname_output!='':
            with open(fname_output, 'a') as fw:
                fw.write("#cfg = {}\n".format(n_cfg))
        lat.gaugefix.gaugefix_Landau(U, max_itr=max_itr, gaugefix_tol=gaugefix_tol, fname=fname_output)
        # Note that `U` is already updated inside the lat.smear.gradient_flow function
        np.save(fname_save(n_cfg), U)
        print(" cfg={} gauge-fixed and saved (#TIME = {:.4g})".format(n_cfg, time.time()-T1))
    Landau_gauge.lat = lat

#==========================================================
#==========================================================
# Example 4::
#   Calculate quark propagagor
#==========================================================
def staggered_quark_prop(size_list=[8,8,8,8], cfgs_list=range(200,1200,50),
        src_type='evenodd_wall', t0=0, color_list=[0],
        ens_tag='', gauge_tag='Landau_',
        prop_tag='eoprop_', mass=0.5,
        do_smear=True, smearing_dict={'u0':0.84}, tadpole=False, **lat_kwargs):
    """
    This function calculating the propagator of a staggered quark.
    Here we smear the links before calculating the quark propagator.
    One can simply disable the smearing by setting the option `do_smear=False`.
    The smearing parameters can be changed by manipulating `smearing_dict`.

    Parameters:
        - ``size_list`` ([int]*4):  a list of 4 positive integers specifying the lattice size in `[x,y,z,t]` directions.
        - ``cfgs_list`` (list):     a list of indices of configurations to be loaded for further calculations.
        - ``src_type``  (str):      a string indicating the source type; the available types are\
                                    'point_src', 'corner_wall', 'even_wall', and 'evenodd_wall', and 'evenminusodd_wall'.
        - ``t0``        (int):      the time slice of the source.
        - ``color_list`` (list):    the color charge of the source; for now the code only accepts one charge, e.g. [0].
        - ``ens_tag``   (str):      a unique tag (label) describing the ensemble:\
                                    for details see the parameters of function :meth:`gauge_tools.examples.generate_ensemble`. 
        - ``gauge_tag`` (str):      a string used to construct a file name for laoding the gauge-fixed configuration;\
                                    see below for clarification. 
        - ``prop_tag``  (str):      a string used to construct a file name for saving the propagators;\
                                    see below for clarification. 
        - ``mass``      (int):      the mass of the quark.
        - ``do_smear``  (bool):     For smearing the links before calculating the quark propagator;\
                                    the default value is `True`.
        - ``smearing_dict`` (dict): a dictionary to control the smearing parameters;\      
                                    for available options see :meth:`gauge_tools.examples.ape_smear`.
        - ``tadpole``   (bool):     if `True`, performs a tadpole improvement on the links befor calculating the propagator;\
                                    the default value is `False`.
   
    A note on the file names:
        from the parameter `ens_tag` this function constructs two file names:
            - to load an existing configuration:\      
                    `fname_load = "{}{}{}.npy".format(ens_tag, gauge_tag, ind_cfg)`.
            - to saved the propagator:\
                    `fname_save = "{}{}m{}_{}.npy".format(ens_tag, prop_tag, mass, ind_cfg)`.

    """
    fname_load = lambda ind_cfg: "{}{}{}.npy".format(ens_tag, gauge_tag, ind_cfg)
    fname_save = lambda ind_cfg: "{}{}m{}_{}.npy".format(ens_tag, prop_tag, mass, ind_cfg)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    lat.quark.set_source(src_type,t0=t0,color_list=color_list)
    for n_cfg in cfgs_list:
        T1 = time.time()
        U  = lat.GF.load_cfg(fname_load(n_cfg))
        if do_smear:
            lat.smear.ape_smear(U, **smearing_dict) # the smearing is performed on `U`
        lat.quark.calc_propagator(U, mass, n_restart=1, tadpole=tadpole)
        lat.quark.save_propagator(fname_save(n_cfg))
        print(" cfg={} quark propagator is calculated and saved for mass={} (#TIME = {:.4g})".format(n_cfg, mass, time.time()-T1))
    staggered_quark_prop.lat = lat

#==========================================================
def avg_quark_prop(size_list=[8,8,8,8], cfgs_list=range(200,1200,50),
        src_type='evenodd_wall', t0=0, color_list=[0],
        ens_tag='', prop_tag='eoprop_', mass=0.5,
        figname='', **lat_kwargs):
    """
    This function loads the propagators calculated in :meth:`gauge_tools.examples.staggered_quark_prop`
    and averages them. Most key word arguments are similar to those of :meth:`gauge_tools.examples.staggered_quark_prop`,
    except for ``figname``, which if not an empyt string, is a signal to create a figure and save it as a pdf in `figname`.
    """
    fname_load = lambda ind_cfg: "{}{}m{}_{}.npy".format(ens_tag, prop_tag, mass, ind_cfg)
    fname_save = "{}{}m{}_avg.p".format(ens_tag, prop_tag, mass)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    props_list = []
    props_projected = []
    for n_cfg in cfgs_list:
        prop_v_field = np.load(fname_load(n_cfg), allow_pickle=True)
        props_list.append(prop_v_field)
        props_projected.append( gt.util.quark.propagator.ks_project_spatialmom(prop_v_field, color=color_list[0]))
    props_proj_gvar = gv.dataset.avg_data(props_projected)
    pickle.dump(dict(mean=gv.mean(props_proj_gvar), cov=gv.evalcov(props_proj_gvar)), open(fname_save, 'wb'))
    avg_quark_prop.props_list = props_list
    avg_quark_prop.props_proj_gvar = props_proj_gvar
    avg_quark_prop.lat = lat
    if figname!='' and PLOTS:
        plt.ion()
        fig = plt.figure()
        plt.errorbar(range(len(props_proj_gvar)), np.abs(gv.mean(props_proj_gvar)), gv.sdev(props_proj_gvar),fmt='.',label='interacting')
        free_theory(gt, src_type=src_type, t0=t0, mass=mass, color_list=color_list, print_=False)
        plt.title('qaurk propagator in free and interacting theory')
        plt.legend()
        plt.yscale('log')
        fig.savefig(figname,format="pdf")

def free_theory(gt, src_type="evenodd_wall", t0=0, mass=0.5, color_list=[0], print_=False):
    src = gt.util.quark.quark_src(src_type, t0)
    src.build_src(color_list)
    free = gt.util.quark.propagator(src)
    free.build_free_prop(mass)
    if print_:
        gt.util.quark.propagator.print_vector_field(free.v_field,n_blocks=7)
    free_projected = gt.util.quark.propagator.ks_project_spatialmom(free.v_field, color=color_list[0])
    if PLOTS:
        plt.plot(np.abs(free_projected), 's', label='Free theory')

#==========================================================
def gradient_flow(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), action='W',
                    max_flowtime=1, eps=0.01, save_field=False, fname_output='',
                    ens_tag='', figname='', **lat_kwargs):
    """
    This function uses the gradient flow method to smear the gauge fields,
    and calculates the lattice spacing. The topological charge is also calculated but not investigated yet.

    ***NOTE*** The materials related to the gradeint flow are NOT complete yet.
    This function will probably change in the next versions.

    """
    fname_load = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    fname_save = lambda ind_cfg: "{}{}_flowtime{}.npy".format(ens_tag, n_cfg, max_flowtime)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, action=action, u0=1, **lat_kwargs) # we do not use tadpole improvement
    a_dset = []
    for n_cfg in cfgs_list:
        T1= time.time()
        U = lat.GF.load_cfg(fname_load(n_cfg))
        if fname_output!='':
            with open(fname_output, 'a') as fw:
                fw.write("#cfg = {}\n".format(n_cfg))
        mydict = lat.smear.gradient_flow(U, lat.actn.calc_staples, max_flowtime, eps, fname=fname_output)
        # Note that `U` is already updated inside the lat.smear.gradient_flow function
        a = mydict['a']
        if save_field:
            np.save(fname_save(n_cfg), U)
            print(" cfg={:.6g} is calculated at flowtime {} and saved; yeilds a={} fm (#TIME = {:.4g})".format(n_cfg,max_flowtime,a,time.time()-T1))
        else:
            print(" a={:.6g} fm\tfor cfg={} (#TIME = {:.4g})".format(a, n_cfg, time.time()-T1))
        a_dset.append(a)
    a = gv.dataset.avg_data(a_dset)
    print(" average: a = {} fm".format(a))
    gradient_flow.lat = lat
    gradient_flow.mydict = mydict

#==========================================================
def measure_polyakov_loops(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), binsize=1,
                    ens_tag='', **lat_kwargs):
    """ This function calculate the polyakov loops on the given configurations.
    """
    fname_lambda = lambda ind_cfg: "{}{}.npy".format(ens_tag, ind_cfg)
    import gauge_tools as gt
    lat = gt.lattice(*size_list, **lat_kwargs)
    fn  = lambda U: lat.meas.avg_polyakov_loops(U)
    print("Calculating averages of polyakov loops:")
    T1  = time.time()
    avg = lat.MC.eval_fn(fn, cfgs_list, fname_lambda, binsize=binsize, avg_data=True)
    print("<polyakov-loops>: {}".format(avg),end=''),
    print("\t(#TIME = {:.4g})".format(time.time()-T1))
    measure_polyakov_loops.avg = avg
    measure_polyakov_loops.lat = lat

#==========================================================

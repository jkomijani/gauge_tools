.. _examples:

:mod:`Examples` 
===================================================

The exercises suggested in the `lecture notes`_ are presented here as primary examples:
    - :meth:`gauge_tools.examples.generate_ensemble`
    - :meth:`gauge_tools.examples.expand_ensemble`
    - :meth:`gauge_tools.examples.measure_Wilson_loops`
    - :meth:`gauge_tools.examples.static_potential`
    - :meth:`gauge_tools.examples.ape_smear`
There are more advanced examples too:
    - :meth:`gauge_tools.examples.Landau_gauge`
    - :meth:`gauge_tools.examples.staggered_quark_prop`
    - :meth:`gauge_tools.examples.avg_quark_prop`
    - :meth:`gauge_tools.examples.gradient_flow`, which also provides a method for scale setting
    - :meth:`gauge_tools.examples.measure_polyakov_loops` .

By looking at the scripts of these examples one can figure out how to use the package
for other applications.
To run the examples with limited number of configurations, 
you can simply run the python file in ``test/run_examples.py``,
and compare the output with ``test/run_examples.log``.

.. _lecture notes: https://arxiv.org/abs/hep-lat/0506036v1

`generate`
----------
.. automethod:: gauge_tools.examples.generate_ensemble(size_list=[8,8,8,8], beta=5.5, action='W', u0=1, n_therm=200, n_cfgs=5, n_skip=50, eps_hit=0.24, update_u0=False, ens_tag='')

`expand`
--------
.. automethod:: gauge_tools.examples.expand_ensemble(size_list=[8,8,8,8], beta=5.5, action='W', u0=1, load_ind=None, n_cfgs=15, n_skip=50, eps_hit=0.24,ens_tag='')

`Wilson loops`
--------------
.. automethod:: gauge_tools.examples.measure_Wilson_loops(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), binsize=1, ens_tag='')

`potential`
-----------
.. automethod:: gauge_tools.examples.static_potential(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), max_R=4.1, max_T=5, ens_tag='', smear_tag='', do_smear=True, smearing_dict={'u0':0.84}, figname='')

`APE smear`
-----------
.. automethod:: gauge_tools.examples.ape_smear(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), n_smear=4, eps_smear=1/12., u0=0.84, space_only=True, project_SU3=True, ens_tag='', smear_tag='smear4_')

`Landau gauge`
--------------
.. automethod:: gauge_tools.examples.Landau_gauge(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), max_itr=1000, gaugefix_tol=1e-9, fname_output='',ens_tag='', gauge_tag='Landau_')

`quark propagator`
------------------
.. automethod:: gauge_tools.examples.staggered_quark_prop(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), src_type='evenodd_wall', t0=0, color_list=[0], ens_tag='', gauge_tag='Landau_', prop_tag='eoprop_', mass=0.5, do_smear=True, smearing_dict={'u0':0.84}, tadpole=False)

`avg. quark prop.`
------------------
.. automethod:: gauge_tools.examples.avg_quark_prop(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), src_type='evenodd_wall', t0=0, color_list=[0], ens_tag='', prop_tag='eoprop_', mass=0.5, figname='')

`gradient flow`
---------------
.. automethod:: gauge_tools.examples.gradient_flow(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), action='W', max_flowtime=1, eps=0.01, save_field=False, fname_output='', ens_tag='', figname='')

`Polyakov loops`
----------------
.. automethod:: gauge_tools.examples.measure_polyakov_loops(size_list=[8,8,8,8], cfgs_list=range(200,1200,50), binsize=1, ens_tag='')



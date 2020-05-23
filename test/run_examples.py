#!/usr/bin/python3

import gauge_tools as gt

import time

T0 = time.time()

size_list = [8,8,8,8]
action  = 'W'
beta    = 5.5
u0      = 1.0

ens_tag = 'W_l8x8_b5500_'

update_u0 = False

###==========================================================
gt.examples.generate_ensemble(size_list=size_list, action=action, beta=beta, u0=u0, ens_tag=ens_tag, n_cfgs=1, n_therm=200, update_u0=update_u0)
gt.examples.expand_ensemble(  size_list=size_list, action=action, beta=beta, u0=u0, ens_tag=ens_tag, n_cfgs=3, load_ind=200)
gt.examples.measure_Wilson_loops(size_list=size_list, ens_tag=ens_tag, cfgs_list=range(200,400,50))

gt.examples.static_potential(size_list=size_list, ens_tag=ens_tag, cfgs_list=range(200,400,50), max_R=4.1, max_T=3, figname='static_pot.pdf')

gt.examples.Landau_gauge(size_list=size_list, ens_tag=ens_tag, cfgs_list=range(200,400,50), max_itr=1000,gaugefix_tol=1e-9,fname_output='Landau.out')
gt.examples.staggered_quark_prop(size_list=size_list,ens_tag=ens_tag,cfgs_list=range(200,400,50),prop_tag='eoprop_',mass=0.5,src_type='evenodd_wall')
gt.examples.avg_quark_prop(size_list=size_list,ens_tag=ens_tag,cfgs_list=range(200,400,50), prop_tag='eoprop_', mass=0.5, figname='quark_eoprop.pdf')

gt.examples.gradient_flow(size_list=size_list, ens_tag=ens_tag, cfgs_list=range(200,400,50), action='W', max_flowtime=1, eps=0.03, fname_output='gflow.out') 

gt.examples.measure_polyakov_loops(size_list=size_list, ens_tag=ens_tag, cfgs_list=range(200,400,50))

###==========================================================
print("#Total time = {}".format((time.time()-T0)))

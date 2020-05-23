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


from ._gtcore import param, action_util, measurement_util, MC_util, gauge_field

from .util.gaugefix import gaugefix
from .util.quark import quark_field
from .util.smear import smear

from .lib._matrix import matrix, vector, ReTr
from .lib._site   import site

from gauge_tools  import examples

class lattice(object):
    """ The main object for setting up a lattice.
    An instance of this class is equipped with utilities available in

        - ``.param``: used to organize the lattice parameters;\
                    an instance of class :class:`gauge_tools.param`.
        - ``.actn``:  utilities for defining the action of interest;\
                    an instance of class :class:`gauge_tools.action_util`.
        - ``.meas``:  utilities for computing (measuring) quantities such as Wilson loops;\
                    an instance of class :class:`gauge_tools.measurement_util`.
        - ``.GF``:    utilities for manipulating gauge fields;\
                    an instance of class :class:`gauge_tools.gauge_field`.
        - ``.MC``:    utilities for Monte Carlo simulations;\
                    an instance of class :class:`gauge_tools.MC_util`.
        - ``.quark``: utilities for calculating quark propagators;\
                    an instance of class :class:`gauge_tools.util.quark.quark_field`.
        - ``.smear``: utilities for smeaing links, such as APE smearing and gradient flow;\
                    an instance of class :class:`gauge_tools.util.smear.smear`.
        - ``.gaugefix``: utilities for gauge fixing;\
                    an instance of class :class:`gauge_tools.util.gaugefix.gaugefix`.

    Parameters:
        - ``nx,ny,nz,nt`` (*all* int):  size of lattice in 4 directions.
        - ``dim``               (int):  number of directions. The default value is 4.
        - ``nc``                (int):  number of colors. The default value is 3.
        - ``**action_kwargs``:          options for defining an action.\
                                        If not given, the default sets will be used.

    Most applications currently assume `nc=3` and `dim=4`.
    The plan is to enable simulations with `dim<4` and arbitrary `nc`
    in the next versions.
    """
    def __init__(self, nx, ny, nz, nt, dim=4, nc=3, **action_kwargs):
        self.param  =   param(nx,ny,nz,nt,dim,nc)
        self.actn   =   action_util(**action_kwargs)
        self.meas   =   measurement_util()
        self.GF     =   gauge_field(create_links=False) # If created, the gauge links would be in `self.GF.U`
        self.MC     =   MC_util(self.actn, self.meas, self.GF) # MC: Monte Carlo
        self.quark  =   quark_field(self.param)
        self.smear  =   smear(self.param)
        self.gaugefix = gaugefix(self.param)


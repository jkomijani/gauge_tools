:mod:`Main objects`
===================================================
The main objects of this package are defined in two files:
    - `gauge_tools/__init__.py` ,
    - `gauge_tools/_gtcore.pyx` .

These objects themselves use a set of more basic objects and functions defined
in :ref:`lib` such as :class:`gauge_tools.lib._matrix.matrix`,
:class:`gauge_tools.lib._site.site`, and :meth:`gauge_tools.lib._rand.uniform`.

Moreover, there are some application available in :ref:`util` such as
:class:`gauge_tools.util.smear.smear`, :class:`gauge_tools.util.gaugefix.gaugefix`,
and :class:`gauge_tools.util.quark.quark_field`
that are useful for various applications.

The interface of the package with the user is meant to be :class:`gauge_tools.lattice`.
The commands::

        import gauge_tools as gt
        lat = gt.lattice(8,8,8,8)

set up all necessary things for a simulation of a ``8x8x8x8`` lattice.
Details of simulation, e.g. the choice of `action`, can be controlled through
optional arguments.
As one can see in :class:`gauge_tools.lattice`, (most) applications available
in :ref:`util` are already accessible in ``lat``.

To start a new simulation or application, depending on the details:
    - use the functions available in :ref:`examples`,
    - write new functions similar to those defined in `gauge_tools/examples.py` ,
    - or write new applications similar to those in `gauge_tools/util/`.

Below are the main objects of the package.

:mod:`lattice`
------------------
.. autoclass:: gauge_tools.lattice
   :members:

.. autoclass:: gauge_tools.param
   :members:

:mod:`gauge field`
------------------
.. autoclass:: gauge_tools.gauge_field
   :members:

:mod:`Monte Carlo`
------------------
.. autoclass:: gauge_tools.MC_util
   :members:

:mod:`action`
------------------
.. autoclass:: gauge_tools.action_util
   :members:

:mod:`measurement`
------------------
.. autoclass:: gauge_tools.measurement_util
   :members:


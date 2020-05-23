Introduction
------------
This package contains tools for Monte Carlo simulations of gauge theories.
The module is mainly written for lattice QCD simulations in 4 dimensions
and in the quenched limit without quarks in the sea.
It is developed in the process of teaching lattice QCD following these `lecture notes`_.
This package can be particularly useful for students who would like to learn lattice QCD,
for professors who would like to teach QCD with some simple examples illustrating
different aspects of QCD, and for those who are interested to investigate pure gauge
theories.

The exercises suggested in the `lecture notes`_ are given as primary examples.
By looking at them one can easily figure out how to use the package.
There are more examples that are beyond the materials in the `lecture notes`_
such as gauge fixing, calculating the propagator of a quark, smearing the gauge fields
with the gradient-flow method, and setting the lattice scale.
For a quick start see :ref:`examples`.

This is just the start of the project. I would like to expand it to include more
utilities and also to generalize to ``SU(n)`` theories.
I am looking forward to hearing from you if you have any suggestions.

| Created by Javad Komijani, University of Tehran, May 2020
| Copyright (C) 2020 Javad Komijani

.. _lecture notes: https://arxiv.org/abs/hep-lat/0506036v1

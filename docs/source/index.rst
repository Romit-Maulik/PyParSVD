.. pyparsvd documentation master file, created by
   sphinx-quickstart on Mar 19 12:02:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyParSVD's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



* The **GitHub repository** of this package can be found at `PyParSVD <https://github.com/Romit-Maulik/PyParSVD>`_ along installation instructions, and how to get started.

* **Tutorials** can be found at `PyParSVD-Tutorials <https://github.com/Romit-Maulik/PyParSVD/tree/main/tutorials>`_

* The package uses `Travis-CI <https://travis-ci.com>`_ for **continuous integration**.



Summary
==================

The `PyParSVD` library implements both a serial and a parallel singular value decomposition (SVD).
The implementation of the library is conveniently:
- Distributed using MPI4Py (for parallel SVD);
- Streaming - data can be shown in batches to update the left singular vectors;
- Randomized - further acceleration of any serial components of the overall algorithm.

The `PyParSVD` library is organized following an abstract factory design pattern, where we define
a base class in `parsvd_base.py`, called `ParSVD_Base` :ref:`ParSVD base class`, that implements
functions and parameters available to all derived classes. In addition, it implements two abstract
functions, `initialize()` and `incorporate_data()` which implementation must be provided by the
derived classes.

The classes derived from the base class are the following:
  - `ParSVD_serial` (implemented in `parsvd_serial.py`) :ref:`ParSVD serial`
  - `ParSVD_parallel` (implemented in `parsvd_parallel.py`) :ref:`ParSVD parallel`

These derived classes contain the actual implementation of two different different versions
of SVD algorithms, one that is serial (`parsvd_serial.py`) and one that is parallel (`parsvd_parallel.py`).

**It should be noted that the design pattern chosen allows for the
easy addition of derived classes that can implement a new SVD versions.**

The distributed (parallel) implementation of the SVD in `parsvd_parallel.py` follows
`(Wang et al. 2016) <https://www.sciencedirect.com/science/article/pii/S0377042715005774>`_.
The streaming algorithm used in the library for both `parsvd_serial.py` and `parsvd_parallel.py`
is from `(Levy and Lindenbaum 1998) <https://ieeexplore.ieee.org/abstract/document/723422>`_,
where the parallel QR algorithm (the TSQR method) required for the streaming feature follows
`(Benson et al. 2013) <https://ieeexplore.ieee.org/document/6691583>`_.
Finally, the randomized algorithm adopted in the library follows
`(Halko et al 2013) <https://epubs.siam.org/doi/abs/10.1137/090771806>`_.

Additionally to these modules, we also provide some post-processing capabilities
to visualize the results. These are implemented in `postprocessing.py` :ref:`Postprocessing module`.
The functions in post-processing can be accessed directly from the base class, and in particular
from the `ParSVD object` returned by the `initialize()` and `incorporate_data()` function.
They can also be accessed separately from the base class, as the post-processing module
constitutes a standalone module. In practice, once you run an analysis, you can load the
results at a later stage and use the post-processing module to visualize the results or
you can implement you own visualization tools, that best suit your needs.

Indices and table
-----------------

* :ref:`genindex`
* :ref:`modindex`



ParSVD main modules
===================

The ParSVD main modules constitutes the backbone of the `PyParSVD` library.
They are constituted by the base class:

  - `ParSVD_Base` (implemented in `parsvd_base.py`) :ref:`ParSVD base class`

along with its derived classes:

  - `ParSVD_serial` (implemented in `parsvd_serial.py`) :ref:`ParSVD serial`
  - `ParSVD_parallel` (implemented in `parsvd_parallel.py`) :ref:`ParSVD parallel`

ParSVD base class
-----------------

The **ParSVD base class** is intended to hold functions that are shared
by all derived classes. It follows an abstract factory design pattern.

.. automodule:: pyparsvd.parsvd_base
	:members: ParSVD_Base

ParSVD serial
-------------

.. automodule:: pyparsvd.parsvd_serial
	:members: ParSVD_Serial

ParSVD parallel
---------------

.. automodule:: pyparsvd.parsvd_parallel
	:members: ParSVD_Parallel



Postprocessing module
=====================

The postprocessing module is intended to provide some limited support to post-process
the data and results produced by **pyparsvd**. The key routines implemented are

.. automodule:: pyparsvd.postprocessing
	:members: plot_singular_values,
			  plot_1D_modes

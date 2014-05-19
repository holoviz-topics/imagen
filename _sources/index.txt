.. image:: _static/patterntypes_small.png

******
ImaGen
******

.. toctree::
   :maxdepth: 2

The ImaGen package provides comprehensive support for creating
resolution-independent one and two-dimensional spatial pattern
distributions. ImaGen consists of a large library of primarily
two-dimensional patterns, including mathematical functions, geometric
primitives, images read from files, and many ways to combine or select from any
other patterns. These patterns can be used in any Python program that needs
configurable patterns or a series of patterns, with only a small amount of
user-level code to specify or use each pattern.

Example usage
_____________

Running the following code in an IPython Notebook session generates two
Gaussian patterns:

.. notebook:: imagen index.ipynb

Installation
============

ImaGen requires NumPy (http://numpy.scipy.org/) and Param (http://ioam.github.com/param/).

Official releases of ImaGen are available at http://pypi.python.org/pypi/imagen,
and can be installed along with dependencies via `pip install imagen` or
`easy_install imagen`.

Alternatively, after separately installing the dependencies, ImaGen can be
installed via `python setup.py install` (Windows users can download and run
an exe from the preceding link).

Support
=======

Questions and comments are welcome at https://github.com/ioam/imagen/issues.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


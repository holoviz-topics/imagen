"""
ImaGen analysis provides common analysis functions, which can be applied to
any SheetView or SheetStack. This allows the user to perform analyses on their
input patterns or any other arrays embedded within a SheetView and display
the output of the analysis alongside the original patterns.

Currently this module provides FFT, auto-correlation and gradient analyses as
well the analysis baseclass, which will apply any TransferFn to the data.
"""

import numpy as np
from numpy.fft.fftpack import fft2
from numpy.fft.helper import fftshift

import param
from param import ParamOverrides

from . import wrap
from views import SheetView, SheetStack
from sheetcoords import BoundingBox
from transferfn import TransferFn


class analysis(param.ParameterizedFunction):
    """
    The analysis baseclass provides support for processing SheetStacks,
    SheetViews and lists of SheetView objects. The actual transformation is
    performed by the _analysis method, which can be subclassed to provide any
    desired transformation, however by default it will apply the supplied
    transfer_fn.
    """

    transfer_fn = param.ClassSelector(class_=TransferFn, default=None)

    __abstract = True

    def __call__(self, view, **params):
        p = ParamOverrides(self, params)

        if isinstance(view, SheetView):
            return self._analysis(p, view)
        elif isinstance(view, SheetStack):
            return view.clone([(k, self._analysis(p, sv))
                               for k, sv in view.items()], bounds=None)
        elif isinstance(view, list):
            return [self._analysis(p, sv) for sv in view]


    def _analysis(self, p, sheetview):
        data = sheetview.data.copy()
        if p.transfer_fn is not None:
            p.transfer_fn(data)
        return SheetView(data, sheetview.bounds, metadata=sheetview.metadata)



class fft(analysis):
    """
    Compute the 2D Fast Fourier Transform (FFT) of the supplied sheet view.

    Example:: fft(topo.sim.V1.views.maps.OrientationPreference)
    """

    peak_val = param.Number(default=1.0)

    def _analysis(self, p, sheetview):
        cr = sheetview.cyclic_range
        data = sheetview.data if cr is None else sheetview.data/cr
        fft_spectrum = abs(fftshift(fft2(data - 0.5, s=None, axes=(-2, -1))))
        fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
        zero_min_spectrum = fft_spectrum - fft_spectrum.min()
        spectrum_range = fft_spectrum.max() - fft_spectrum.min()
        normalized_spectrum = (p.peak_val * zero_min_spectrum) / spectrum_range

        l, b, r, t = sheetview.bounds.lbrt()
        density = sheetview.xdensity
        bb = BoundingBox(radius=(density/2)/(r-l))

        return SheetView(normalized_spectrum, bb, metadata=sheetview.metadata)



class gradient(analysis):
    """
    Compute the gradient plot of the supplied SheetView or SheetStack.
    Translated from Octave code originally written by Yoonsuck Choe.

    If the SheetView has a cyclic_range, negative differences will
    be wrapped into the range.

    Example:: gradient(topo.sim.V1.views.maps.OrientationPreference)
    """

    cyclic_range = param.Number(default=None, allow_None=True)

    def _analysis(self, p, sheetview):
        data = sheetview.data
        r, c = data.shape
        dx = np.diff(data, 1, axis=1)[0:r-1, 0:c-1]
        dy = np.diff(data, 1, axis=0)[0:r-1, 0:c-1]

        cyclic_range = 1.0 if sheetview.cyclic_range is None else sheetview.cyclic_range
        if cyclic_range is not None: # Wrap into the specified range
            # Convert negative differences to an equivalent positive value
            dx = wrap(0, cyclic_range, dx)
            dy = wrap(0, cyclic_range, dy)
            #
            # Make it increase as gradient reaches the halfway point,
            # and decrease from there
            dx = 0.5 * cyclic_range - np.abs(dx - 0.5 * cyclic_range)
            dy = 0.5 * cyclic_range - np.abs(dy - 0.5 * cyclic_range)

        return SheetView(np.sqrt(dx*dx + dy*dy), sheetview.bounds,
                         metadata=sheetview.metadata)



class autocorrelation(analysis):
    """
    Compute the 2D autocorrelation of the supplied data. Requires the external
    SciPy package.

    Example:: autocorrelation(topo.sim.V1.views.maps.OrientationPreference)
    """

    def _analysis(self, p, sheetview):
        import scipy.signal
        data = sheetview.data
        autocorr_data = scipy.signal.correlate2d(data, data)
        return SheetView(autocorr_data, sheetview.bounds,
                         metadata=sheetview.metadata)

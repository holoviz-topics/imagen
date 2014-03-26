"""
The ImaGen analysis module provides common analysis functions, which
can be applied to any SheetView or SheetStack. This allows the user to
perform analyses on their input patterns or any other arrays embedded
within a SheetView and display the output of the analysis alongside
the original patterns.

Currently this module provides FFT, auto-correlation and gradient
analyses.
"""

import numpy as np
from numpy.fft.fftpack import fft2
from numpy.fft.helper import fftshift

from matplotlib import pyplot as plt

import param
from param import ParamOverrides

from dataviews import SheetView, SheetStack,  SheetLayer, SheetLines, SheetOverlay
from dataviews.sheetcoords import BoundingBox

from imagen import wrap



class SheetOperation(param.ParameterizedFunction):
    """
    A SheetOperation is a transformation that operates on the
    SheetLayer level.
    """

    def _process(self, view):
        """
        A single SheetLayer may be returned but multiple SheetLayer
        outputs may be returned as a tuple.
        """
        raise NotImplementedError

    def __call__(self, view, **params):
        self.p = ParamOverrides(self, params)

        if isinstance(view, SheetLayer):
            return self._process(view)
        elif isinstance(view, SheetStack):
            return view.map(lambda el, k: self._process(el))
        else:
            raise TypeError("Not a SheetLayer or SheetStack.")



class fft_power_spectrum(SheetOperation):
    """
    Compute the 2D Fast Fourier Transform (FFT) of the supplied sheet view.

    Example::
    fft_power_spectrum(topo.sim.V1.views.maps.OrientationPreference)
    """

    peak_val = param.Number(default=1.0)

    def _process(self, sheetview):
        cr = sheetview.cyclic_range
        data = sheetview.data if cr is None else sheetview.data/cr
        fft_spectrum = abs(fftshift(fft2(data - 0.5, s=None, axes=(-2, -1))))
        fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
        zero_min_spectrum = fft_spectrum - fft_spectrum.min()
        spectrum_range = fft_spectrum.max() - fft_spectrum.min()
        normalized_spectrum = (self.p.peak_val * zero_min_spectrum) / spectrum_range

        l, b, r, t = sheetview.bounds.lbrt()
        density = sheetview.xdensity
        bb = BoundingBox(radius=(density/2)/(r-l))

        return SheetView(normalized_spectrum, bb, metadata=sheetview.metadata)



class gradient(SheetOperation):
    """
    Compute the gradient plot of the supplied SheetView or SheetStack.
    Translated from Octave code originally written by Yoonsuck Choe.

    If the SheetView has a cyclic_range, negative differences will be
    wrapped into the range.

    Example:: gradient(topo.sim.V1.views.maps.OrientationPreference)
    """

    def _process(self, sheetview):
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



class autocorrelation(SheetOperation):
    """
    Compute the 2D autocorrelation of the supplied data. Requires the
    external SciPy package.

    Example::
    autocorrelation(topo.sim.V1.views.maps.OrientationPreference)
    """

    def _process(self, sheetview):
        import scipy.signal
        data = sheetview.data
        autocorr_data = scipy.signal.correlate2d(data, data)
        return SheetView(autocorr_data, sheetview.bounds,
                         metadata=sheetview.metadata)



class contours(SheetOperation):
    """
    Given a SheetView with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is a overlay with a SheetLines layer for each given
    level, overlaid on top of the input SheetView.
    """

    levels = param.NumericTuple(default=(0.5,), doc="""
         A list of scalar values used to specify the contour levels.""")

    colors = param.List(default=[], doc="""
         If not empty, this is a list of color strings to associate
         with each contour level. This list must have the same length
         as the levels parameter.""")

    def _process(self, sheetview):

        if self.p.colors and len(self.p.colors) != len(self.p.levels):
            raise Exception("List of colors must match number of levels.")

        colors = self.p.colors if self.p.colors else [None] * len(self.p.levels)

        figure_handle = plt.figure()
        (l,b,r,t) = sheetview.bounds.lbrt()
        contour_set = plt.contour(sheetview.data,
                                  extent=(l,r,t,b),
                                  levels=self.p.levels)

        sheetlines = []
        for col, level, cset in zip(colors, self.p.levels, contour_set.collections):
            paths = cset.get_paths()
            lines = [path.vertices for path in paths]
            sheetline = SheetLines(lines,
                                   sheetview.bounds,
                                   metadata={'level':level})
            if col is not None:
                sheetline.style['color']=col
            sheetlines.append(sheetline)

        plt.close(figure_handle)

        if len(sheetlines) == 1:
            return (sheetview * sheetlines[0])
        else:
            return sheetview * SheetOverlay(sheetlines, sheetview.bounds)

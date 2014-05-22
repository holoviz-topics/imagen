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

import param
from dataviews import SheetView
from dataviews.sheetviews import BoundingBox
from dataviews.options import options, GrayNearest
from dataviews.operation  import ViewOperation

from imagen import wrap




class fft_power_spectrum(ViewOperation):
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

        return [SheetView(normalized_spectrum, bb,
                          metadata=sheetview.metadata,
                          label=sheetview.label+' FFT Power Spectrum')]



class gradient(ViewOperation):
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

        return [SheetView(np.sqrt(dx*dx + dy*dy), sheetview.bounds,
                          metadata=sheetview.metadata,
                          label=sheetview.label+' Gradient')]



class autocorrelation(ViewOperation):
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
        return [SheetView(autocorr_data, sheetview.bounds,
                          metadata=sheetview.metadata,
                          label=sheetview.label+' AutoCorrelation')]





class cyclic_similarity_index(ViewOperation):
    """
    The similarity index between any two cyclic maps. By default, a
    zero value indicates uncorrelated SheetView data. The similarity
    index may be useful for quantifying the stability of some cyclic
    quantity over time by comparing each sample in a SheetStack to the
    final SheetView element.
    """

    unit_range = param.Boolean(default=True, doc="""
        Whether to scale the similarity values linearly so that
        uncorrelated values have a value of zero and exactly matching
        elements are indicated with a value of 1.0. Negative values
        are then used to indicate anticorrelation.""")

    def _process(self, overlay):

        if len(overlay) != 2:
             raise Exception("The similarity index may only be computed using overlays of SheetViews.")

        if any(el.cyclic_range is None for el in overlay):
             raise Exception("The SheetViews in each  overlay must have a defined cyclic range.")

        prefA_data = overlay[0].N.data
        prefB_data = overlay[1].N.data
        # Ensure difference is symmetric distance.
        difference = abs(prefA_data - prefB_data)
        greaterHalf = (difference >= 0.5)
        difference[greaterHalf] = 1.0 - difference[greaterHalf]
        # Difference [0,0.5] so 2x normalizes...
        similarity = 1 - difference * 2.0
        # Subtracted from 1.0 as low difference => high stability
        # As this is made into a unit metric, uncorrelated has value zero.
        similarity = (2 * (similarity - 0.5)) if self.p.unit_range else similarity
        return [SheetView(similarity, bounds=overlay.bounds,
                          label=overlay[0].label+' Cyclic Similarity')]


options.CyclicSimilarity_SheetView    = GrayNearest
options.AutoCorrelation_SheetView     = GrayNearest
options.Gradient_SheetView            = GrayNearest
options.FFTPowerSpectrum_SheetView    = GrayNearest

"""
PatternGenerator abstract class, basic example concrete class, and
multichannel support.

PatternGenerators support both single-channel patterns, i.e. bare
arrays, and multiple channels, such as for color images.  See
``PatternGenerator.__call__`` and ``PatternGenerator.channels`` for
more information.
"""

import numpy as np
from numpy import pi
import collections

import param
from param.parameterized import ParamOverrides

from holoviews import ViewMap, Matrix
from holoviews.core import BoundingBox, BoundingRegionParameter, SheetCoordinateSystem
from holoviews.core.options import options, StyleOpts

from .transferfn import TransferFn


# CEBALERT: PatternGenerator has become a bit of a monster abstract
# class.  Can it be split into the minimum required to specify the
# interface, with a subclass implementing the rest (this subclass
# still being above the rest of the PatternGenerators)?  We want to
# make it easy to add new types of PatternGenerator that don't match
# the assumptions of the current ones (OneDPowerSpectrum is an example
# of a PG that doesn't match the current assumptions), but still lets
# them be used like the current ones.
# (PatternGenerator-->TwoDPatternGenerator?)

# JLALERT: PatternGenerator should have
# override_plasticity_state/restore_plasticity_state functions which
# can override the plasticity of any output_fn that has state, in case
# anyone ever uses such an object in a PatternGenerator.  Will also
# need to support Composite patterns.


class PatternGenerator(param.Parameterized):
    """
    A class hierarchy for callable objects that can generate 2D patterns.

    Once initialized, PatternGenerators can be called to generate a
    value or a matrix of values from a 2D function, typically
    accepting at least x and y.

    A PatternGenerator's Parameters can make use of Parameter's
    precedence attribute to specify the order in which they should
    appear, e.g. in a GUI. The precedence attribute has a nominal
    range of 0.0 to 1.0, with ordering going from 0.0 (first) to 1.0
    (last), but any value is allowed.

    The orientation and layout of the pattern matrices is defined by
    the SheetCoordinateSystem class, which see.

    Note that not every parameter defined for a PatternGenerator will
    be used by every subclass.  For instance, a Constant pattern will
    ignore the x, y, orientation, and size parameters, because the
    pattern does not vary with any of those parameters.  However,
    those parameters are still defined for all PatternGenerators, even
    Constant patterns, to allow PatternGenerators to be scaled, rotated,
    translated, etc. uniformly.
    """
    __abstract = True

    bounds  = BoundingRegionParameter(
        default=BoundingBox(points=((-0.5,-0.5), (0.5,0.5))),precedence=-1,
        doc="BoundingBox of the area in which the pattern is generated.")

    xdensity = param.Number(default=256,bounds=(0,None),precedence=-1,doc="""
        Density (number of samples per 1.0 length) in the x direction.""")

    ydensity = param.Number(default=256,bounds=(0,None),precedence=-1,doc="""
        Density (number of samples per 1.0 length) in the y direction.
        Typically the same as the xdensity.""")

    x = param.Number(default=0.0,softbounds=(-1.0,1.0),precedence=0.20,doc="""
        X-coordinate location of pattern center.""")

    y = param.Number(default=0.0,softbounds=(-1.0,1.0),precedence=0.21,doc="""
        Y-coordinate location of pattern center.""")


    position = param.Composite(attribs=['x','y'],precedence=-1,doc="""
        Coordinates of location of pattern center.
        Provides a convenient way to set the x and y parameters together
        as a tuple (x,y), but shares the same actual storage as x and y
        (and thus only position OR x and y need to be specified).""")

    orientation = param.Number(default=0.0,softbounds=(0.0,2*pi),precedence=0.40,doc="""
        Polar angle of pattern, i.e., the orientation in the Cartesian coordinate
        system, with zero at 3 o'clock and increasing counterclockwise.""")

    size = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,6.0),
        precedence=0.30,doc="""Determines the overall size of the pattern.""")

    scale = param.Number(default=1.0,softbounds=(0.0,2.0),precedence=0.10,doc="""
        Multiplicative strength of input pattern, defaulting to 1.0""")

    offset = param.Number(default=0.0,softbounds=(-1.0,1.0),precedence=0.11,doc="""
        Additive offset to input pattern, defaulting to 0.0""")

    mask = param.Parameter(default=None,precedence=-1,doc="""
        Optional object (expected to be an array) with which to multiply the
        pattern array after it has been created, before any output_fns are
        applied. This can be used to shape the pattern.""")

    # Note that the class type is overridden to PatternGenerator below
    mask_shape = param.ClassSelector(param.Parameterized,default=None,precedence=0.06,doc="""
        Optional PatternGenerator used to construct a mask to be applied to
        the pattern.""")

    output_fns = param.HookList(default=[],class_=TransferFn,precedence=0.08,doc="""
        Optional function(s) to apply to the pattern array after it has been created.
        Can be used for normalization, thresholding, etc.""")


    def __init__(self,**params):
        super(PatternGenerator, self).__init__(**params)
        self.set_matrix_dimensions(self.bounds, self.xdensity, self.ydensity)


    def __call__(self,**params_to_override):
        """
        Call the subclass's 'function' method on a rotated and scaled
        coordinate system.

        Creates and fills an array with the requested pattern.  If
        called without any params, uses the values for the Parameters
        as currently set on the object. Otherwise, any params
        specified override those currently set on the object.
        """
        if 'output_fns' in params_to_override:
            self.warning("Output functions specified through the call method will be ignored.")

        p=ParamOverrides(self,params_to_override)

        # CEBERRORALERT: position parameter is not currently
        # supported. We should delete the position parameter or fix
        # this.
        #
        # position=params_to_override.get('position',None) if position
        # is not None: x,y = position

        self._setup_xy(p.bounds,p.xdensity,p.ydensity,p.x,p.y,p.orientation)
        fn_result = self.function(p)
        self._apply_mask(p,fn_result)
        if p.scale != 1.0:
            result = p.scale * fn_result
        else:
            result = fn_result
        if p.offset != 0.0:
            result += p.offset

        for of in p.output_fns:
            of(result)

        return result


    def __getitem__(self, coords):
        arr = (np.dstack(self.channels().values()[1:])
               if self.num_channels() in [3,4] else self())
        return Matrix(arr, self.bounds,
                      label=self.__class__.__name__+ ' Pattern')[coords]


    def channels(self, use_cached=False, **params_to_override):
        """
        Channels() adds a shared interface for single channel and
        multichannel structures.  It will always return an ordered
        dict: its first element is the single channel of the pattern
        (if single-channel) or the channel average (if multichannel);
        the successive elements are the individual channels' arrays
        (key: 0,1,..N-1).
        """
        return collections.OrderedDict({ 'default':self.__call__(**params_to_override) })


    def num_channels(self):
        """
        Query the number of channels implemented by the
        PatternGenerator. In case of single-channel generators this
        will return 1; in case of multichannel, it will return the
        number of channels (eg, in the case of RGB images it would
        return '3', Red-Green-Blue, even though the OrderedDict
        returned by channels() will have 4 elements -- the 3 channels
        + their average).
        """
        return 1


    def _setup_xy(self,bounds,xdensity,ydensity,x,y,orientation):
        """
        Produce pattern coordinate matrices from the bounds and
        density (or rows and cols), and transforms them according to
        x, y, and orientation.
        """
        self.debug(lambda:"bounds=%s, xdensity=%s, ydensity=%s, x=%s, y=%s, orientation=%s"%(bounds,xdensity,ydensity,x,y,orientation))
        # Generate vectors representing coordinates at which the pattern
        # will be sampled.

        # CB: note to myself - use slice_._scs if supplied?
        x_points,y_points = SheetCoordinateSystem(bounds,xdensity,ydensity).sheetcoordinates_of_matrixidx()

        # Generate matrices of x and y sheet coordinates at which to
        # sample pattern, at the correct orientation
        self.pattern_x, self.pattern_y = self._create_and_rotate_coordinate_arrays(x_points-x,y_points-y,orientation)


    def function(self,p):
        """
        Function to draw a pattern that will then be scaled and rotated.

        Instead of implementing __call__ directly, PatternGenerator
        subclasses will typically implement this helper function used
        by __call__, because that way they can let __call__ handle the
        scaling and rotation for them.  Alternatively, __call__ itself
        can be reimplemented entirely by a subclass (e.g. if it does
        not need to do any scaling or rotation), in which case this
        function will be ignored.
        """
        raise NotImplementedError


    def _create_and_rotate_coordinate_arrays(self, x, y, orientation):
        """
        Create pattern matrices from x and y vectors, and rotate them
        to the specified orientation.
        """
        # Using this two-liner requires that x increase from left to
        # right and y decrease from left to right; I don't think it
        # can be rewritten in so little code otherwise - but please
        # prove me wrong.
        pattern_y = np.subtract.outer(np.cos(orientation)*y, np.sin(orientation)*x)
        pattern_x = np.add.outer(np.sin(orientation)*y, np.cos(orientation)*x)
        return pattern_x, pattern_y


    def _apply_mask(self,p,mat):
        """Create (if necessary) and apply the mask to the given matrix mat."""
        mask = p.mask
        ms=p.mask_shape
        if ms is not None:
            mask = ms(x=p.x+p.size*(ms.x*np.cos(p.orientation)-ms.y*np.sin(p.orientation)),
                      y=p.y+p.size*(ms.x*np.sin(p.orientation)+ms.y*np.cos(p.orientation)),
                      orientation=ms.orientation+p.orientation,size=ms.size*p.size,
                      bounds=p.bounds,ydensity=p.ydensity,xdensity=p.xdensity)
        if mask is not None:
            mat*=mask


    def set_matrix_dimensions(self, bounds, xdensity, ydensity):
        """
        Change the dimensions of the matrix into which the pattern
        will be drawn.  Users of this class should call this method
        rather than changing the bounds, xdensity, and ydensity
        parameters directly.  Subclasses can override this method to
        update any internal data structures that may depend on the
        matrix dimensions.
        """
        self.bounds = bounds
        self.xdensity = xdensity
        self.ydensity = ydensity
        scs = SheetCoordinateSystem(bounds, xdensity, ydensity)
        for of in self.output_fns:
            of.initialize(SCS=scs, shape=scs.shape)

    def state_push(self):
        "Save the state of the output functions, to be restored with state_pop."
        for of in self.output_fns:
            if hasattr(of,'state_push'):
                of.state_push()
        super(PatternGenerator, self).state_push()


    def state_pop(self):
        "Restore the state of the output functions saved by state_push."
        for of in self.output_fns:
            if hasattr(of,'state_pop'):
                of.state_pop()
        super(PatternGenerator, self).state_pop()


    def anim(self, frames, offset=0, timestep=1,
             dimension='Frames', time_fn=param.Dynamic.time_fn):
        """
        frames: The number of frames generated relative to offset.

        time_fn: The time object shared across the time-varying
        objects that are to be sampled.

        offset: The time offset from which frames are generated given
        the supplied pattern

        timestep: The time interval between successive layers. Valid
        time values must be an integer multiple of this timestep value
        (which may be a float or some other numeric type).

        dimension: Animations are indexed by time. This may be by
        integer frame number or some continuous (e.g. floating point
        or rational) representation of time.

        Note that the offset, timestep, dimensions and time_fn only
        affect patterns parameterized by time-dependent number
        generators. Otherwise, the frames are generated by successive
        call to the pattern which may or may not be varying (e.g to
        view the patterns contained within a Selector).
        """
        vmap = ViewMap(dimensions=[dimension])
        self.state_push()
        with time_fn as t:
            t(offset)
            for i in range(frames):
                vmap[t()] = self[:]
                t += timestep
        self.state_pop()
        return vmap

# Override class type; must be set here rather than when mask_shape is declared,
# to avoid referring to class not yet constructed
PatternGenerator.params('mask_shape').class_=PatternGenerator


# Trivial example of a PatternGenerator, provided for when a default is
# needed.  The other concrete PatternGenerator classes are stored
# elsewhere, to be imported as needed.

class Constant(PatternGenerator):
    """Constant pattern generator, i.e., a solid, uniform field of the same value."""

    # The orientation is ignored, so we don't show it in
    # auto-generated lists of parameters (e.g. in the GUI)
    orientation = param.Number(precedence=-1)

    # Optimization: We use a simpler __call__ method here to skip the
    # coordinate transformations (which would have no effect anyway)
    def __call__(self,**params_to_override):
        p = ParamOverrides(self,params_to_override)

        shape = SheetCoordinateSystem(p.bounds,p.xdensity,p.ydensity).shape

        result = p.scale*np.ones(shape, np.float)+p.offset
        self._apply_mask(p,result)

        for of in p.output_fns:
            of(result)

        return result



class ChannelTransform(param.Parameterized):
    """
    A ChannelTransform is a callable object that takes channels as
    input (an ordered dictionary of arrays) and transforms their
    contents in some way before returning them.
    """

    __abstract = True

    def __call__(self, channels):
        raise NotImplementedError



# Example of a ChannelTransform
class CorrelateChannels(ChannelTransform):
    """
    Correlate channels by mixing a fraction of one channel into another.
    """

    from_channel = param.Number(default=1, doc="""
        Name of the channel to take data from.""")

    to_channel = param.Number(default=2, doc="""
        Name of the channel to change data of.""")

    strength = param.Number(default=0, doc="""
        Strength of the correlation to add, with 0 being no change,
        and 1.0 overwriting to_channel with from_channel.""")

    def __call__(self, channel_data):
        channel_data[self.to_channel] = \
            self.strength*channel_data[self.from_channel] + \
            (1-self.strength)*channel_data[self.to_channel]

        return channel_data



class ChannelGenerator(PatternGenerator):
    """
    Abstract base class for patterns supporting multiple channels natively.
    """

    __abstract = True

    channel_transforms = param.HookList(class_=ChannelTransform,default=[],doc="""
        Optional functions to apply post processing to the set of channels.""")


    def __init__(self, **params):
        self._original_channel_data = [] # channel data before processing
        self._channel_data = []          # channel data after processing
        super(ChannelGenerator, self).__init__(**params)


    def channels(self, use_cached=False, **params_to_override):
        res = collections.OrderedDict()

        if not use_cached:
            default = self(**params_to_override)
            res['default'] = default
        else:
            res['default'] = None

        for i in range(len(self._channel_data)):
            res[i] = self._channel_data[i]

        return res

    def num_channels(self):
        return len(self._channel_data)


class ComposeChannels(ChannelGenerator):
    """
    Create a multi-channel PatternGenerator from another
    PatternGenerator.

    If the specified generator itself already posseses more than one
    channel, will use its channels' data; otherwise, will synthesize
    the channels from the single channel of the generator.

    After finding or synthesizing the channels, they are scaled
    according to the corresponding channel_factors.
    """

    generators = param.List(class_=PatternGenerator,default=[Constant(scale=0.0)],
                            bounds=(1,None), doc="""
        List of patterns to use for each channel. Generators which already have more than one
        channel will only contribute to a single channel of ComposeChannels.""")


    def __init__(self,**params):
        super(ComposeChannels,self).__init__(**params)

        for i in range(len(self.generators)):
            self._channel_data.append( None )


    def __call__(self,**params):
        # Generates all channels, then returns the default channel

        p = param.ParamOverrides(self,params)

        params['xdensity']=p.xdensity
        params['ydensity']=p.ydensity
        params['bounds']=p.bounds

        # (not **p)
        for i in range(len(p.generators)):
            self._channel_data[i] = p.generators[i]( **params )


        for c in self.channel_transforms:
            self._channel_data = c(self._channel_data)

        return sum(act for act in self._channel_data)/len(self._channel_data)



options.Pattern_SheetView = StyleOpts(cmap='gray')

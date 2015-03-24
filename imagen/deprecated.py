"""
Old patterns not intended for new code.

These patterns are expected to be deleted eventually.
"""

import param
import numpy as np
from numpy import pi

from param.parameterized import ParamOverrides
from param import ClassSelector

from .patterngenerator import Constant, PatternGenerator
from . import Gaussian


#JABALERT: Can't this be replaced with a Composite?
class TwoRectangles(PatternGenerator):
    """Two 2D rectangle pattern generator."""

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,6.0),
        precedence=0.31,doc=
        "Ratio of width to height; size*aspect_ratio gives the width of the rectangle.")

    x1 = param.Number(default=-0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="X center of rectangle 1.")

    y1 = param.Number(default=-0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="Y center of rectangle 1.")

    x2 = param.Number(default=0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="X center of rectangle 2.")

    y2 = param.Number(default=0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="Y center of rectangle 2.")

    size = param.Number(default=0.5)

    # YC: Maybe this can be implemented much more cleanly by calling
    # the parent's function() twice, but it's hard to see how to
    # set the (x,y) offset for the parent.
    def function(self,p):
        height = p.size
        width = p.aspect_ratio*height

        return np.bitwise_or(
               np.bitwise_and(np.bitwise_and(
                        (self.pattern_x-p.x1)<=p.x1+width/4.0,
                        (self.pattern_x-p.x1)>=p.x1-width/4.0),
                      np.bitwise_and(
                        (self.pattern_y-p.y1)<=p.y1+height/4.0,
                        (self.pattern_y-p.y1)>=p.y1-height/4.0)),
               np.bitwise_and(np.bitwise_and(
                        (self.pattern_x-p.x2)<=p.x2+width/4.0,
                        (self.pattern_x-p.x2)>=p.x2-width/4.0),
                      np.bitwise_and(
                        (self.pattern_y-p.y2)<=p.y2+height/4.0,
                        (self.pattern_y-p.y2)>=p.y2-height/4.0)))


### JABALERT: This class should be eliminated if at all possible; it
### is just a specialized version of Composite, and should be
### implementable directly using what is already in Composite.
class GaussiansCorner(PatternGenerator):
    """
    Two Gaussian pattern generators with a variable intersection point,
    appearing as a corner or cross.
    """

    x = param.Number(default=-0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="X center of the corner")

    y = param.Number(default=-0.15,bounds=(-1.0,1.0),softbounds=(-0.5,0.5),
                doc="Y center of the corner")

    size = param.Number(default=0.5,bounds=(0,None), softbounds=(0.1,1),
                doc="The size of the corner")

    aspect_ratio = param.Number(default=1/0.31, bounds=(0,None), softbounds=(1,10),
                doc="Ratio of the width to the height for both Gaussians")

    angle = param.Number(default=0.5*pi,bounds=(0,pi), softbounds=(0.01*pi,0.99*pi),
                doc="The angle of the corner")

    cross = param.Number(default=0.4, bounds=(0,1), softbounds=(0,1),
                doc="Where the two Gaussians cross, as a fraction of their half length")


    def __call__(self,**params_to_override):
        p = ParamOverrides(self,params_to_override)

        g_1 = Gaussian()
        g_2 = Gaussian()

        x_1 = g_1(orientation = p.orientation, bounds = p.bounds, xdensity = p.xdensity,
                            ydensity = p.ydensity, offset = p.offset, size = p.size,
                            aspect_ratio = p.aspect_ratio,
                            x = p.x + 0.7 * np.cos(p.orientation) * p.cross * p.size * p.aspect_ratio,
                            y = p.y + 0.7 * np.sin(p.orientation) * p.cross * p.size * p.aspect_ratio)
        x_2 = g_2(orientation = p.orientation+p.angle, bounds = p.bounds, xdensity = p.xdensity,
                            ydensity = p.ydensity, offset = p.offset, size = p.size,
                            aspect_ratio = p.aspect_ratio,
                            x = p.x + 0.7 * np.cos(p.orientation+p.angle) * p.cross * p.size * p.aspect_ratio,
                            y = p.y + 0.7 * np.sin(p.orientation+p.angle) * p.cross * p.size * p.aspect_ratio)

        return np.maximum( x_1, x_2 )



class Translator(PatternGenerator):
    """
    PatternGenerator that translates another PatternGenerator over
    time.

    This PatternGenerator will create a series of episodes, where in
    each episode the underlying generator is moved in a fixed
    direction at a fixed speed.  To begin an episode, the Translator's
    x, y, and direction are evaluated (e.g. from random
    distributions), and the underlying generator is then drawn at
    those values plus changes over time that are determined by the
    speed.  The orientation of the underlying generator should be set
    to 0 to get motion perpendicular to the generator's orientation
    (which is typical).

    Note that at present the parameter values for x, y, and direction
    cannot be passed in when the instance is called; only the values
    set on the instance are used.
    """
    generator = param.ClassSelector(default=Gaussian(),
        class_=PatternGenerator,doc="""Pattern to be translated.""")

    direction = param.Number(default=0.0,softbounds=(-pi,pi),doc="""
        The direction in which the pattern should move, in radians.""")

    speed = param.Number(default=0.01,bounds=(0.0,None),doc="""
        The speed with which the pattern should move,
        in sheet coordinates per time_fn unit.""")

    reset_period = param.Number(default=1,bounds=(0.0,None),doc="""
        Period between generating each new translation episode.""")

    episode_interval = param.Number(default=0,doc="""
        Interval between successive translation episodes.

        If nonzero, the episode_separator pattern is presented for
        this amount of time_fn time after each episode, e.g. to
        allow processing of the previous episode to complete.""")

    episode_separator = param.ClassSelector(default=Constant(scale=0.0),
         class_=PatternGenerator,doc="""
         Pattern to display during the episode_interval, if any.
         The default is a blank pattern.""")

    time_fn = param.Callable(default=param.Dynamic.time_fn,doc="""
        Function to generate the time used as a base for translation.""")

    def _advance_params(self):
        """
        Explicitly generate new values for these parameters only
        when appropriate.
        """
        for p in ['x','y','direction']:
            self.force_new_dynamic_value(p)
        self.last_time = self.time_fn()


    def __init__(self,**params):
        super(Translator,self).__init__(**params)
        self._advance_params()


    def __call__(self,**params_to_override):
        p=ParamOverrides(self,params_to_override)

        if self.time_fn() >= self.last_time + p.reset_period:
            ## Returns early if within episode interval
            if self.time_fn()<self.last_time+p.reset_period+p.episode_interval:
                return p.episode_separator(xdensity=p.xdensity,
                                           ydensity=p.ydensity,
                                           bounds=p.bounds)
            else:
                self._advance_params()

        # JABALERT: Does not allow x, y, or direction to be passed in
        # to the call; fixing this would require implementing
        # inspect_value and force_new_dynamic_value (for
        # use in _advance_params) for ParamOverrides.
        #
        # Access parameter values without giving them new values
        assert ('x' not in params_to_override and
                'y' not in params_to_override and
                'direction' not in params_to_override)
        x = self.inspect_value('x')
        y = self.inspect_value('y')
        direction = self.inspect_value('direction')

        # compute how much time elapsed from the last reset
        # float(t) required because time could be e.g. gmpy.mpq
        t = float(self.time_fn()-self.last_time)

        ## CEBALERT: mask gets applied twice, both for the underlying
        ## generator and for this one.  (leads to redundant
        ## calculations in current lissom_oo_or usage, but will lead
        ## to problems/limitations in the future).
        return p.generator(
            xdensity=p.xdensity,ydensity=p.ydensity,bounds=p.bounds,
            x=x+t*np.cos(direction)*p.speed+p.generator.x,
            y=y+t*np.sin(direction)*p.speed+p.generator.y,
            orientation=(direction-pi/2)+p.generator.orientation)




# Legacy Sweeper class which is used in lissom.ty, should be deleted
# once lissom.ty is deprecated
#
# CB: I removed motion_sign from this class because I think it is
# unnecessary. But maybe I misunderstood the original author's
# intention?
#
# In any case, the original implementation was incorrect - it was not
# possible to get some motion directions (directions in one whole
# quadrant were missed out).
#
# Note that to get a 2pi range of directions, one must use a 2pi range
# of orientations (there are two directions for any given
# orientation).  Alternatively, we could generate a random sign, and
# use an orientation restricted to a pi range.

class OldSweeper(PatternGenerator):
    """
    PatternGenerator that sweeps a supplied PatternGenerator in a
    direction perpendicular to its orientation.
    """

    generator = param.Parameter(default=Gaussian(),precedence=0.97, doc="Pattern to sweep.")

    speed = param.Number(default=0.25,bounds=(0.0,None),doc="""
        Sweep speed: number of sheet coordinate units per unit time.""")

    step = param.Number(default=1,doc="""
        Number of steps at the given speed to move in the sweep direction.
        The distance moved is speed*step.""")

    # Provide access to value needed for measuring maps
    def __get_phase(self): return self.generator.phase
    def __set_phase(self,new_val): self.generator.phase = new_val
    phase = property(__get_phase,__set_phase)

    def function(self,p):
        """Selects and returns one of the patterns in the list."""
        pg = p.generator
        motion_orientation=p.orientation+pi/2.0

        new_x = p.x+p.size*pg.x
        new_y = p.y+p.size*pg.y

        image_array = pg(xdensity=p.xdensity,ydensity=p.ydensity,bounds=p.bounds,
                         x=new_x + p.speed*p.step*np.cos(motion_orientation),
                         y=new_y + p.speed*p.step*np.sin(motion_orientation),
                         orientation=p.orientation,
                         scale=pg.scale*p.scale,offset=pg.offset+p.offset)

        return image_array



_public = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and issubclass(_v,PatternGenerator)]))
__all__ = _public

"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'

import param
from sheetcoords import SheetCoordinateSystem, Slice
from boundingregion import BoundingBox, BoundingRegion
from ndmapping import NdMapping


class SheetView(param.Parameterized, SheetCoordinateSystem):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.
    """

    cyclic_range = param.Number(default=None, bounds=(0, None), doc="""
        For a cyclic quantity, the range over which the values repeat. For
        instance, the orientation of a mirror-symmetric pattern in a plane is
        pi-periodic, with orientation x the same as orientation x+pi (and
        x+2pi, etc.) A cyclic_range of None declares that the data are not
        cyclic. This parameter is metadata, declaring properties of the data
        that can be useful for automatic plotting and/or normalization, and is
        not used within this class itself.""")

    roi_bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data.""")

    _deep_indexable = True

    def __init__(self, data, bounds, **kwargs):
        self.data = data

        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = data.shape
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)

        param.Parameterized.__init__(self, **kwargs)
        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)

        if self.roi_bounds is None:
            self.roi_bounds = self.bounds


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords is ():
            return self

        if all([isinstance(c, slice) for c in coords]):
            l, b, r, t = self.bounds.lbrt()
            xcoords, ycoords = coords
            xstart = l if xcoords.start is None else xcoords.start
            xend = b if xcoords.end is None else xcoords.end
            ystart = r if ycoords.start is None else ycoords.start
            yend = t if ycoords.end is None else ycoords.end
            bounds = BoundingBox(points=((xstart, ystart), (xend, yend)))
        else:
            raise IndexError('Indexing requires x- and y-slice ranges.')

        return Slice(bounds, self).submatrix(self.data)


    @property
    def roi(self):
        return Slice(self.roi_bounds, self).submatrix(self.data)



class SheetStack(NdMapping):
    """
    A SheetStack is a stack of SheetViews over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions along
    which SheetViews may vary.
    """

    enforced_type = param.Parameter(default=SheetView, constant=True)

    def _element_check(self, data):
        super(SheetStack, self)._element_check(data)
        if not hasattr(self, 'bounds'): self.bounds = data.bounds
        if not data.bounds.lbrt() == self.bounds.lbrt():
            raise AssertionError("All SheetView elements must have matching bounds.")



class ProjectionGrid(NdMapping, SheetCoordinateSystem):
    """
    ProjectionView indexes other NdMapping objects, containing projections
    onto coordinate systems. The X and Y dimensions are mapped onto the bounds
    object, allowing for bounds checking and grid-snapping.
    """

    dimension_labels = param.List(default=['X', 'Y'])

    def __init__(self, bounds=None, shape=None, **kwargs):
        if bounds is None or shape is None:
            raise Exception("Specify bounds and shape to initialize ProjectionGrid.")

        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = shape
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(ProjectionGrid, self).__init__(**kwargs)


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        if not self.bounds.contains(*coords):
            self.warning('Specified coordinate is outside grid bounds,'
                         ' data could not be added')
        self._element_check(data)
        coords = self._transform_indices(coords)
        super(ProjectionGrid, self)._add_item(coords, data, sort=sort)


    def _transform_indices(self, coords):
        return tuple([self._transform_index(i, coord) for (i, coord) in enumerate(coords)])


    def _transform_index(self, dim, index):
        if isinstance(index, slice):
            [start, stop] = [self._transform_value(el, dim)
                             for el in (index.start, index.stop)]
            return slice(start, stop)
        else:
            return self._transform_value(index, dim)


    def _transform_value(self, val, dim):
        """
        Subclassed to discretize grid spacing.
        """
        if val is None: return None
        return self.closest_cell_center(*((0, val) if dim else (val, 0)))[dim]


    def update(self, other):
        """
        Adds bounds checking to the default update behavior.
        """
        if self.bounds.lbrt() != other.bounds.lbrt():
            raise Exception('Cannot combine %ss with different'
                            ' bounds.' % self.__class__)
        super(ProjectionGrid, self).update(other)


    def empty(self):
        """
        Returns an empty duplicate of itself with all parameter values and
        metadata copied across.
        """
        settings = dict(self.get_param_values(), **self.metadata)
        return self.__class__(bounds=self.bounds, shape=self.shape, **settings)



class Timeline(param.Parameterized):
    """
    A Timeline object takes a time function of type param.Time and
    uses this object's context-manager facilities to sample
    time-varying objects over a given portion of the timeline. The
    return values are ImaGen data structures which may then be
    manipulated, stored or visualized.

    For instance, an ImaGen pattern with time-varying Dynamic
    parameters may be sampled using the 'stack' method. Each snapshot
    of the pattern becomes a SheetView held in the returned
    SheetStack. Each of the spatial discretizations in each SheetView
    (a numpy array) is generated at a particular time according to the
    time_fn used by the pattern and its parameter. For this to work
    correctly, the time_fn of the objects sampled must match the
    'time' parameter, which is set to param.Dynamic.time_fn by
    default. The returned SheetStack is then a spatiotemporal sampling
    of the pattern, i.e., as a stack of images over time.

    A Timeline is also a context manager that extends the behaviour of
    Time by surrounding the block with state_push and state_pop for a
    given parameterized object or list of parameterized objects. The
    objects to state_push and pop are supplied to the __call__ method
    which returns the context manager. For an example for how Timeline
    and Time may be used as a context manager, consult the example
    given in the docstring of the Time class.
    """

    time = param.ClassSelector(default=param.Dynamic.time_fn,
                               class_=param.Time,doc="""
        The time object shared across the time-varying objects
        that are to be sampled.""")


    def __init__(self, time=None, **kwargs):
        if time is None:
            time = param.Dynamic.time_fn
        super(Timeline, self).__init__(time=time, **kwargs)


    def __enter__(self):
        if self._objs is None:
            raise Exception("Call needs to be supplied with the object(s) that require state push and pop.")
        for obj in self._objs:
            obj.state_push()
        return self.time.__enter__()


    def __exit__(self, exc, *args):
        for obj in self._objs:
            obj.state_pop()
        self._objs = None
        self.time.__exit__(exc, *args)


    def __call__(self, objs):
        """
        Returns a context manager which acts like the Time context
        manager but also pushes and pops the state of the specified
        object or collection of objects.
        """
        try:
            self._objs = list(iter(objs))
        except TypeError:
            self._objs = [objs]
        return self


    def ndmap(self, obj, until, offset=0, timestep=None,
              value_fn=lambda obj: obj()):
        """
        Builds an NDMapping across time, taking snapshots of some
        time-varying object. The value stored in the NDMapping is
        generated by evaluating value_fn from the time 'offset' till
        the time 'until' in steps of 'timestep'.
        """

        ndmap = NdMapping(dimension_labels=['Time'])
        with self(obj) as t:
            t.until = until
            if timestep is not None:
                t.timestep = timestep
            t(offset)
            for time in t:
                val = value_fn(obj)
                ndmap[time] = val
        return ndmap


    def stack(self, obj, steps, offset=0, timestep=None, bounds=None,
              array_fn=lambda obj: obj(), **kwargs):
        """
        Builds a SheetStack from some time-varying object with bounds
        and representation as a numpy array. The bounds are used
        together with the numpy arrays returned by array_fn to build a
        series of SheetViews into a SheetStack.

        This method accepts time-varying Imagen patterns directly,
        without needing to specify array_fn.
        """
        bounds = obj.bounds if (bounds is None) else bounds
        ndmap = self.ndmap(obj, steps, offset=offset, timestep=timestep,
                           value_fn=lambda obj: SheetView(array_fn(obj),
                                                          bounds))
        return SheetStack(ndmap, metadata=kwargs)

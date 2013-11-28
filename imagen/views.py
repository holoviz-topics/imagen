"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'

import param
from sheetcoords import SheetCoordinateSystem
from ndmapping import NdMapping

class SheetIndexing(param.Parameterized):
    """
    SheetIndexing provides methods to transform indices and slices from one
    coordinate system to another. By default it maps from sheet coordinates
    to matrix coordinates using the _transform_indices method. However,
    subclasses can change this behavior by subclassing the _transform_values
    method, which takes an index value along a certain dimension along with the
    dimension number as input and returns the transformed value.
    """

    bounds = param.Parameter(default=None, doc="""
        The bounds of the two dimensional coordinate system in which the data
        resides.""")

    roi = param.NumericTuple(default=(0, 0, 0, 0), doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data. NOT YET IMPLEMENTED.""")

    __abstract = True

    _deep_indexable = True

    def _create_scs(self):
        (l, b, r, t) = self.bounds.lbrt()
        (dim1, dim2) = self.shape
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)
        return SheetCoordinateSystem(self.bounds, xdensity, ydensity)


    def _transform_indices(self, coords):
        if type(coords) == int:
            raise Exception("***** %s" % coords)
            #coords = [coords]
        return tuple([self._transform_index(i, coord) for (i, coord) in enumerate(coords)][::-1])


    def _transform_index(self, dim, index):
        if isinstance(index, slice):
            [start, stop] = [self._transform_value(el, dim)
                             for el in (index.start, index.stop)]
            if None not in [start, stop]:
                [start, stop] = [start, stop+1] if start < stop else [stop, start+1]
            return slice(start, stop, index.step)
        else:
            return self._transform_value(index, dim)


    def _transform_value(self, val, dim):
        if val is None: return None
        (ind1, ind2) = self.scs.sheet2matrixidx(*((0, val) if dim else (val, 0)))
        return ind1 if dim else ind2


    def matrixidx2coord(self, *args):
        return self.scs.matrixidx2sheet(*args)



class SheetView(SheetIndexing):
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

    def __init__(self, data, bounds, **kwargs):
        super(SheetView, self).__init__(bounds=bounds, **kwargs)
        if 'roi' not in kwargs:
            self.roi = self.bounds.lbrt()
        self.data = data
        self.shape = data.shape
        self.scs = self._create_scs()


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords is ():
            return self

        coord1, coord2 = self._transform_indices(coords)

        return self.data[coord1, coord2]



class SheetStack(SheetView):
    """
    A SheetStack is a stack of SheetViews over some dimension. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other single dimension along
    which SheetViews may vary.

    A SheetStack extends the functionality of SheetView and behaves
    like a single SheetView at a given layer index. This layer index
    may be then modified using the set_layer method.

    A single NDMapping over a single dimension is used to constract a
    SheetStack where all of the maps values must be SheetViews with
    common bounds.
    """

    metadata = param.Dict(default={}, doc="""
        Additional labels to be associated with the SheetStack such as
        hints that control how the SheetStack is displayed.""")

    ndmap = param.ClassSelector(class_=NdMapping, constant=True, doc="""
        The underlying NdMapping object that defines the SheetStack.""")

    def __init__(self, ndmap, layer=0, **kwargs):

        self.sheetviews = ndmap.values()
        if not all(isinstance(v, SheetView) for v in self.sheetviews):
            raise TypeError("All values supplied must be SheetViews")

        super(SheetStack, self).__init__(self.sheetviews[layer].data,
                                         self.sheetviews[layer].bounds,
                                         ndmap=ndmap,**kwargs)

        if len(ndmap.dimension_labels) != 1:
            raise AssertionError("NDMapping must have only one dimension_label")
        if not all(sv.bounds.lbrt() == self.sheetviews[0].bounds.lbrt() for sv in self.sheetviews):
            raise AssertionError("All SheetView must have matching bounds.")

        self.dimension_label = ndmap.dimension_labels[0]
        self.sheetviews = ndmap.values()
        self.labels = ndmap.keys()

        self._layer = None
        self.layer = layer


    @property
    def layer(self):
        """
        Returns the active layer index.
        """
        return self._layer


    @layer.setter
    def layer(self, ind):
        """
        Sets the active layer of the stack by index.
        """
        self._layer = ind
        sview = self.sheetviews[ind]
        self.data = sview.data
        self.cyclic_range = sview.cyclic_range


    def slice(self, start=0, end=None):
        """
        Continuous slice of the stack bases on the supplied NdMapping
        object. Returns a new SheetStack.
        """
        items = self.ndmap[slice(start, end)]
        return SheetStack(NdMapping(initial_items = items))


    def __iter__(self):
        """
        Iterates through the stack from the current layer
        onwards. Returns a SheetView on each iteration and restores
        the layer index at the end.
        """
        start_layer = self.layer
        layer = self.layer
        while layer < self.depth:
            yield self.sheetviews[layer]
            layer += 1
        self.layer = start_layer

    def view(self):
        """
        Return the sheetview corresponding to the active layer.
        """
        return self.sheetviews[self.layer]

    @property
    def depth(self):
        """
        Returns the depth of the stack.
        """
        return len(self.sheetviews)



class ProjectionGrid(SheetIndexing, NdMapping):
    """
    ProjectionView indexes other NdMapping objects, containing projections
    onto coordinate systems. The X and Y dimensions are mapped onto the bounds
    object, allowing for bounds checking and grid-snapping.
    """

    bounds = param.Parameter(default=None, constant=True, doc="""
        The bounds of the coordinate system in which the data items reside.""")

    dimension_labels = param.List(default=['X', 'Y'])

    shape = param.NumericTuple(default=(0, 0), doc="""
        The shape of the grid is required to determine grid spacing and to
        create a SheetCoordinateSystem.""")

    sorted = param.Boolean(default=False, doc="""
        No need to keep ProjectionGrid sorted, since order of keys is
        irrelevant.""")

    def __init__(self, *args, **kwargs):
        super(ProjectionGrid, self).__init__(*args, **kwargs)
        self.scs = self._create_scs()


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        if not self.bounds.contains(*coords):
            self.warning('Specified coordinate is outside grid bounds,'
                         ' data could not be added')
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
        return self.scs.closest_cell_center(*((0, val) if dim
                                              else (val, 0)))[dim]


    def _update_item(self, coords, data):
        """
        Subclasses default method to allow updating of nested data structures
        rather than simply overriding them.
        """
        if coords in self._data:
            self._data[coords].update(data)
        else:
            self._data[coords] = data


    def update(self, other):
        """
        Adds bounds checking to the default update behavior.
        """
        if self.bounds.lbrt() != other.bounds.lbrt():
            raise Exception('Cannot combine %ss with different'
                            ' bounds.' % self.__class__)
        super(ProjectionGrid, self).update(other)



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


    def __init__(self,time=None,**kwargs):
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


    def ndmap(self,obj,until,offset=0,timestep=None,
              value_fn = lambda obj: obj()):
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
              array_fn = lambda obj: obj(), **kwargs):
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
                           value_fn = lambda obj: SheetView(array_fn(obj),
                                                            bounds))
        return SheetStack(ndmap, metadata=kwargs)

"""
Views objects used to hold data and provide indexing into different coordinate
systems.
"""

__version__='$Revision$'


import math
from collections import defaultdict
import numpy as np
import param

from sheetcoords import SheetCoordinateSystem, Slice
from boundingregion import BoundingBox, BoundingRegion
from ndmapping import NdMapping, AttrDict, map_type


class View(param.Parameterized):
    """
    A view is a data structure for holding data, which may be plotted using
    matplotlib. Views have an associated title, style and metadata and can
    be composed together into a GridLayout using the plus operator.
    """

    title = param.String(default=None, allow_None=True, doc="""
       A short description of the layer that may be used as a title.""")

    style = param.Dict(default=AttrDict(), doc="""
        Optional keywords for specifying the display style.""")

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional information to be associated with the Layer.""")


    def __init__(self, data, **kwargs):
        self.data = data
        super(View, self).__init__(**kwargs)


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_grid=[[self, obj]])



class Overlay(View):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """

    _abstract = True

    def __init__(self, overlays, **kwargs):
        super(Overlay, self).__init__([], **kwargs)
        self.set(overlays)


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        self.data.append(layer)


    def set(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        self.data = []
        for layer in layers:
            self.add(layer)
        return self


    def __getitem__(self, ind):
        return self.data[ind]


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1



class SheetLayer(View):
    """
    A SheetLayer is a data structure for holding one or more numpy
    arrays embedded within a two-dimensional space. The array(s) may
    correspond to a discretisation of an image (i.e. a rasterisation)
    or vector elements such as points or lines. Lines may be linearly
    interpolated or correspond to control nodes of a smooth vector
    representation such as Bezier splines.
    """

    bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
       The bounding region in sheet coordinates containing the data.""")

    roi_bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data.""")

    _abstract = True

    def __init__(self, data, bounds, **kwargs):
        kwargs['bounds'] = bounds
        super(SheetLayer, self).__init__(data, **kwargs)

        if self.roi_bounds is None:
            self.roi_bounds = self.bounds


    def __mul__(self, other):

        if isinstance(other, SheetStack):
            items = [(k, self * v) for (k,v) in  other.items()]
            return other.clone(items=items)
        elif isinstance(self, SheetOverlay):
            if isinstance(other, SheetOverlay):
                overlays = self.data + other.data
            else:
                overlays = self.data + [other]
        elif isinstance(other, SheetOverlay):
            overlays = [self] + other.data
        elif isinstance(other, SheetLayer):
            overlays = [self, other]
        else:
            raise TypeError('Can only create an overlay of SheetLayers.')

        return SheetOverlay(overlays, self.bounds,
                            style=self.style, metadata=self.metadata,
                            roi_bounds=self.roi_bounds)



class SheetOverlay(SheetLayer, Overlay):
    """
    SheetOverlay extends a regular Overlay with bounds checking and an
    ROI property, which applies the roi_bounds to all SheetLayer
    objects it contains.

    A SheetOverlay may be used to overlay lines or points over a
    SheetView. In addition, if an overlay consists of three or four
    SheetViews of depth 1, the overlay may be converted to an RGB(A)
    SheetView via the rgb property.
    """

    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if layer.bounds.lbrt() != self.bounds.lbrt():
            raise Exception("Layer must have same bounds as SheetOverlay")
        self.data.append(layer)

    @property
    def rgb(self):
        """
        Convert an overlay of three or four SheetViews into a
        SheetView in RGB(A) mode.
        """
        if len(self) not in [3,4]:
            raise Exception("Requires 3 or 4 layers to convert to RGB(A)")
        if not all(isinstance(el, SheetView) for el in self.data):
            raise Exception("All layers must be SheetViews to convert to RGB(A) format")
        if not all(el.depth==1 for el in self.data):
            raise Exception("All SheetViews must have a depth of one for conversion to RGB(A) format")
        mode = 'rgb' if len(self)==3 else 'rgba'
        return SheetView(np.dstack([el.data for el in self.data]), self.bounds, mode=mode)

    @property
    def roi(self):
        "Apply the roi_bounds to all elements in the SheetOverlay"
        return SheetOverlay([el.roi for el in self.data],
                            bounds=self.roi_bounds,
                            style=self.style, metadata=self.metadata)

    def __len__(self):
        return len(self.data)



class SheetView(SheetLayer, SheetCoordinateSystem):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.

    Arrays with a shape of (X,Y) or (X,Y,Z) are valid. In the case of
    3D arrays, each depth layer is interpreted as a channel of the 2D
    representation.
    """

    cyclic_range = param.Number(default=None, bounds=(0, None), doc="""
        For a cyclic quantity, the range over which the values repeat. For
        instance, the orientation of a mirror-symmetric pattern in a plane is
        pi-periodic, with orientation x the same as orientation x+pi (and
        x+2pi, etc.) A cyclic_range of None declares that the data are not
        cyclic. This parameter is metadata, declaring properties of the data
        that can be useful for automatic plotting and/or normalization, and is
        not used within this class itself.""")

    _deep_indexable = True

    def __init__(self, data, bounds, **kwargs):

        data = np.array([[0]]) if data is None else data
        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = data.shape[0], data.shape[1]
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)

        self._mode = kwargs.pop('mode', None)
        SheetLayer.__init__(self, data, bounds, **kwargs)
        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)


    def __getitem__(self, coords):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        if coords is () or coords == slice(None, None):
            return self

        if not any([isinstance(el, slice) for el in coords]):
            return self.data[self.sheet2matrixidx(*coords)]
        if all([isinstance(c, slice) for c in coords]):
            l, b, r, t = self.bounds.lbrt()
            xcoords, ycoords = coords
            xstart = l if xcoords.start is None else max(l,xcoords.start)
            xend = r if xcoords.stop is None else min(r, xcoords.stop)
            ystart = b if ycoords.start is None else max(b,ycoords.start)
            yend = t if ycoords.stop is None else min(t, ycoords.stop)
            bounds = BoundingBox(points=((xstart, ystart), (xend, yend)))
        else:
            raise IndexError('Indexing requires x- and y-slice ranges.')

        return SheetView(Slice(bounds, self).submatrix(self.data),
                         bounds, cyclic_range=self.cyclic_range,
                         style=self.style, metadata=self.metadata)


    def normalize(self, min=0.0, max=1.0, norm_factor=None):
        norm_factor = self.cyclic_range if norm_factor is None else norm_factor
        if norm_factor is None:
            norm_factor = self.data.max() - self.data.min()
        else:
            min, max = (0.0, 1.0)
        norm_data = (((self.data - self.data.min()) / norm_factor) * abs((max-min))) + min
        return SheetView(norm_data, self.bounds, cyclic_range=self.cyclic_range,
                         metadata=self.metadata, roi_bounds=self.roi_bounds,
                         style=self.style)


    @property
    def depth(self):
        return 1 if len(self.data.shape)==2 else self.data.shape[2]


    @property
    def mode(self):
        """
        Mode specifying the color space for visualizing the array
        data. The string returned corresponds to the matplotlib colour
        map name unless depth is 3 or 4 with modes 'rgb' or 'rgba'
        respectively.

        If not explicitly specified, the mode defaults to 'gray'
        unless the cyclic_range is set, in which case 'hsv' is
        returned.
        """
        if self._mode is not None:
            return self._mode
        return 'gray' if (self.cyclic_range is None) else 'hsv'

    @mode.setter
    def mode(self, val):
        self._mode = val

    @property
    def N(self):
        return self.normalize()


    @property
    def roi(self):
        return SheetView(Slice(self.roi_bounds, self).submatrix(self.data),
                         self.roi_bounds, cyclic_range=self.cyclic_range,
                         style=self.style, metadata=self.metadata)



class SheetPoints(SheetLayer):
    """
    Allows sets of points to be positioned over a sheet coordinate
    system.

    The input data is an Nx2 Numpy array where each point in the numpy
    array corresponds to an X,Y coordinate in sheet coordinates,
    within the declared bounding region.
    """

    def __init__(self, data, bounds, **kwargs):
        data = np.array([[],[]]).T if data is None else data
        super(SheetPoints, self).__init__(data, bounds, **kwargs)


    def resize(self, bounds):
        return SheetPoints(self.points, bounds, style=self.style)

    def __len__(self):
        return self.data.shape[0]

    @property
    def roi(self):
        (N,_) = self.data.shape
        roi_data = self.data[[n for n in range(N) if self.data[n,:] in self.roi_bounds]]
        return SheetPoints(roi_data, self.roi_bounds,
                           style=self.style, metadata=self.metadata)

    def __iter__(self):
        i = 0
        while i < len(self):
            yield tuple(self.data[i,:])
            i+=1



class SheetLines(SheetLayer):
    """
    Allows sets of contour lines to be defined over a
    SheetCoordinateSystem.

    The input data is a list of Nx2 numpy arrays where each array
    corresponds to a contour in the group. Each point in the numpy
    array corresponds to an X,Y coordinate.
    """

    def __init__(self, data, bounds, **kwargs):
        data = [] if data is None else data
        super(SheetLines, self).__init__(data, bounds, **kwargs)

    def resize(self, bounds):
        return SheetLines(self.contours, bounds, style=self.style)

    def __len__(self):
        return self.data.shape[0]

    @property
    def roi(self):
        # Note: Data returned is not sliced to ROI because vertices
        # outside the bounds need to be snapped to the bounding box
        # edges.
        return SheetLines(self.data, self.roi_bounds,
                             style=self.style, metadata=self.metadata)



class SheetStack(NdMapping):
    """
    A SheetStack is a stack of SheetLayers over some dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other dimensions along
    which SheetLayers may vary.
    """

    data_type = param.Parameter(default=SheetLayer, constant=True)

    title = param.String(default=None, doc="""
       A short description of the stack that may be used as a title
       (e.g. the title of an animation) but may also accept a
       formatting string to generate a unique title per layer. For
       instance the format string '{label0} = {value0}' will generate
       a title using the first dimension label and corresponding key
       value. Numbering is by dimension position and extends across
       all available dimensions e.g. {label1}, {value2} and so on.""")

    bounds = param.ClassSelector(class_=BoundingRegion, default=None, doc="""
       The bounding region in sheet coordinates containing the data""")

    @property
    def type(self):
        "The type of elements stored in the stack."
        return None if len(self) == 0 else self.top.__class__

    @property
    def N(self):
        return self.normalize()

    @property
    def roi(self):
        return self.map(lambda x: x.roi)

    @property
    def rgb(self):
        if self.type == SheetOverlay:
            return self.map(lambda x: x.rgb)
        else:
            raise Exception("Can only convert SheetStack of overlays to RGB(A)")


    def _item_check(self, dim_vals, data):
        if self.bounds is None:
            self.bounds = data.bounds
        if not data.bounds.lbrt() == self.bounds.lbrt():
            raise AssertionError("All SheetLayer elements must have matching bounds.")
        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of SheetLayer." % self.__class__.__name__)
        super(SheetStack, self)._item_check(dim_vals, data)


    def map(self, map_fn, **kwargs):
        mapped_items = [(k, map_fn(el)) for k,el in self.items()]
        bounds = mapped_items[0][1].bounds # Bounds of first mapped item
        return self.clone(mapped_items, bounds=bounds, **kwargs)


    def normalize_elements(self, **kwargs):
        return self.map(lambda x: x.normalize(**kwargs))


    def normalize(self, min=0.0, max=1.0):
        data_max = np.max([el.data.max() for el in self.values()])
        data_min = np.min([el.data.min() for el in self.values()])
        norm_factor = data_max-data_min
        return self.map(lambda x: x.normalize(min=min, max=max,
                                              norm_factor=norm_factor))

    def split(self):
        """
        Given a SheetStack of SheetOverlays of N layers, split out the
        layers into N separate SheetStacks.
        """
        if self.type is not SheetOverlay:
            return self.clone(self.items())

        stacks = []
        item_stacks = defaultdict(list)
        for k, overlay in self.items():
            for i, el in enumerate(overlay):
                item_stacks[i].append((k,el))

        for k in sorted(item_stacks.keys()):
            stacks.append(self.clone(item_stacks[k]))
        return stacks


    def __mul__(self, other):
        if isinstance(other, SheetStack):
            all_keys = sorted(set(self.keys() + other.keys()))
            items = []
            for key in all_keys:
                if (key in self) and (key in other):
                    items.append((key, self[key] * other[key]))
                elif (key in self):
                    items.append((key, self[key] * other.type(None, self.bounds)))
                else:
                    items.append((key, self.type(None, self.bounds) * other[key]))
            return self.clone(items=items)
        elif isinstance(other, SheetLayer):
            items = [(k, v * other) for (k,v) in  self.items()]
            return self.clone(items=items)
        else:
            raise Exception("Can only overlay with SheetLayer of SheetStack.")


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_grid=[[self, obj]])



class ProjectionGrid(NdMapping, SheetCoordinateSystem):
    """
    ProjectionView indexes other NdMapping objects, containing projections
    onto coordinate systems. The X and Y dimensions are mapped onto the bounds
    object, allowing for bounds checking and grid-snapping.
    """

    dimension_labels = param.List(default=['X', 'Y'])

    def __init__(self, bounds, shape, initial_items=None, **kwargs):
        (l, b, r, t) = bounds.lbrt()
        (dim1, dim2) = shape
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)

        SheetCoordinateSystem.__init__(self, bounds, xdensity, ydensity)
        super(ProjectionGrid, self).__init__(initial_items, **kwargs)


    def _add_item(self, coords, data, sort=True):
        """
        Subclassed to provide bounds checking.
        """
        if not self.bounds.contains(*coords):
            self.warning('Specified coordinate is outside grid bounds,'
                         ' data could not be added')
        self._item_check(coords, data)
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
        if hasattr(other, 'bounds') and (self.bounds.lbrt() != other.bounds.lbrt()):
            raise Exception('Cannot combine %ss with different'
                            ' bounds.' % self.__class__)
        super(ProjectionGrid, self).update(other)


    def clone(self, items=None, **kwargs):
        """
        Returns an empty duplicate of itself with all parameter values and
        metadata copied across.
        """
        settings = dict(self.get_param_values(), **kwargs)
        settings.pop('metadata', None)
        return ProjectionGrid(bounds=self.bounds, shape=self.shape,
                              initial_items=items,
                              metadata=self.metadata, **settings)

    def __mul__(self, other):
        if isinstance(other, SheetStack) and len(other) == 1:
            other = other.top
        overlayed_items = [(k, el * other) for k,el in self.items()]
        return self.clone(overlayed_items)


    @property
    def top(self):
        """
        The top of a ProjectionGrid is another ProjectionGrid
        constituted of the top of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """

        top_items=[(k, v.clone(items=(v.keys()[-1], v.top)))
                   for (k,v) in self.items()]
        return self.clone(top_items)

    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by SheetStack. For the total number of
        elements, count the full set of keys.
        """
        return  max(len(v) for v in self.values())

    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_grid=[[self, obj]])


class GridLayout(NdMapping):

    key_type = param.List(default=[int, int], constant=True)

    dimension_labels = param.List(default=['Row', 'Column'])

    def __init__(self, initial_grid=[], **kwargs):

        self._max_cols = 4
        initial_grid = [[]] if initial_grid == [] else initial_grid
        items = self._grid_to_items(initial_grid)
        super(GridLayout, self).__init__(initial_items=items, **kwargs)

    @property
    def shape(self):
        rows, cols = zip(*self.keys())
        return max(rows)+1, max(cols)+1

    @property
    def coords(self):
        """
        Compute the list of (row,column,view) elements from the
        current set of items (i.e. tuples of form ((row, column), view))
        """
        if self.keys() == []:  return []
        return [(r,c,v) for ((r,c),v) in zip(self.keys(), self.values())]

    @property
    def max_cols(self):
        return self._max_cols

    @max_cols.setter
    def max_cols(self, n):
        self._max_cols = n
        self.update({}, n)

    def cols(self, n):
        self.update({}, n)
        return self

    def _grid_to_items(self, grid):
        """
        Given a grid (i.e. a list of lists), compute the list of
        items.
        """
        items = []  # Flatten this method to single list comprehension.
        for rind, row in enumerate(grid):
            for cind, view in enumerate(row):
                items.append(((rind, cind), view))
        return items


    def update(self, other, cols=None):
        """
        Given a mapping or iterable of additional views, extend the
        grid in scanline order, obeying max_cols (if applicable).
        """
        values = other if isinstance(other, list) else other.values()
        grid = [[]] if self.coords == [] else self._grid(self.coords)
        new_grid = grid[:-1] + ([grid[-1]+ values])
        cols = self.max_cols if cols is None else cols
        reshaped_grid = self._reshape_grid(new_grid, cols)
        self._data = map_type(self._grid_to_items(reshaped_grid))


    def __call__(self, cols=None):
        """
        Recompute the grid layout of the views based on precedence and
        row_precendence value metadata. Formats the grid to a maximum
        of cols columns if specified.
        """
        # Plots are sorted first by precedence, then grouped by row_precedence
        values = sorted(self.values(), key=lambda x: x.metadata.get('precedence', 0.5))
        precedences = sorted(set(v.metadata.get('row_precedence', 0.5) for v in values))

        coords=[]
        # Can use collections.Counter in Python >= 2.7
        column_counter = dict((i, 0) for i, _ in enumerate(precedences))
        for view in values:
            # Find the row number based on the row_precedences
            row = precedences.index(view.metadata.get('row_precedence', 0.5))
            # Look up the current column position of the row
            col = column_counter[row]
            # The next view on this row will have to be in the next column
            column_counter[row] += 1
            coords.append((row, col, view))

        grid = self._reshape_grid(self._grid(coords), cols)
        self._data = map_type(self._grid_to_items(grid))
        return self

    def _grid(self, coords):
        """
        From a list of coordinates of form [<(row, col, view)>] build
        a corresponding list of lists grid.
        """
        rows = max(r for (r,_,_) in coords) + 1 if coords != [] else 0
        unpadded_grid = [[p for (r,_, p) in coords if r==row] for row in range(rows)]
        return unpadded_grid


    def _reshape_grid(self, grid, cols):
        """
        Given a grid (i.e. a list of lists) , reformat it to a layout
        with a maximum of cols columns (if not None).
        """
        if cols is None: return grid
        flattened = [view for row in grid for view in row if (view is not None)]
        row_num = int(math.ceil(len(flattened) / float(cols)))

        reshaped_grid = []
        for rind in range(row_num):
            new_row = flattened[rind*cols:cols*(rind+1)]
            reshaped_grid.append(new_row)

        return reshaped_grid

    def __add__(self, other):
        new_values = other.values() if isinstance(other, GridLayout) else [other]
        self.update(new_values)
        return self

    def __len__(self):
        return max([len(v) for v in self.values() if isinstance(v, SheetStack)]+[1])


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

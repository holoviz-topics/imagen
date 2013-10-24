import param
from sheetcoords import SheetCoordinateSystem
from ndmapping import NdMapping, AttrDict


class SheetIndexing(param.Parameterized):
    """
    SheetIndexing provides methods to indexes and slices from one coordinate
    system to another. By default it maps from sheet coordinates to matrix
    coordinates using the _transform_indices method. However subclasses can
    change this behavior by subclassing the _transform_values method,
    which takes an index value along a certain dimension along with the
    dimension number as input returns the transformed value.
    """

    bounds = param.Parameter(default=None, doc="""
        The bounds of the two dimensional coordinate system in which the data
        resides.""")

    roi = param.NumericTuple(default=(0, 0, 0, 0), doc="""
        The ROI can be specified to select only a sub-region of the bounds to
        be stored as data. NOT YET IMPLEMENTED.""")

    __abstract = True

    def _create_scs(self):
        (l, b, r, t) = self.bounds.lbrt()
        (dim1, dim2) = self.shape
        xdensity = dim1 / (r - l)
        ydensity = dim2 / (t - b)
        return SheetCoordinateSystem(self.bounds, xdensity, ydensity)


    def _transform_indices(self, coords):
        return tuple([self._transform_index(i, coord)
                      for (i, coord) in enumerate(coords)])


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
        (ind1, ind2) = self.scs.sheet2matrixidx(*((val, 0) if dim else (0, val)))
        return ind2 if dim else ind1


    def matrixidx2coord(self, *args):
        return self.scs.matrixidx2sheet(*args)



class SheetView(SheetIndexing):
    """
    SheetView is the atomic unit as which 2D data is stored, along with its
    bounds object. Allows slicing operations of the data in sheet coordinates or
    direct access to the data, via the .data attribute.
    """

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


class ProjectionGrid(SheetIndexing, NdMapping):
    """
    ProjectionView indexes other RangeMap objects, containing projections
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
                            ' bounds.' % self._class__)
        super(ProjectionGrid, self).update(other)

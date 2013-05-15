import param
import bisect

from imagen.sheetcoords import SheetCoordinateSystem

try:     from collections import OrderedDict
except:  OrderedDict = None

class DataView(param.Parameterized):
    """
    A dataview associates data with the appropriate bounds.
    """

    timestamp = param.Number(default=None, doc=
        """ The initial timestamp. If None, the DataView will not all slicing of
            a time interval and record method will be disabled.""")

    roi = param.Parameter(default=None, doc=
        """ The bounds of the region of interest (if any). Any attempt to access
            data outside this region will generate a warning. """)

    bounds = param.Parameter(default=None, doc=
        """ The bounds of the two dimensional coordinate system in which the data resides.""")

    cyclic_interval = param.Number(default=None, allow_None=True, doc=
        """ If None, the data is in a non-cyclic dimension.  Otherwise, the data
            is in a cyclic dimension over the given interval.""")

    interval_dtype = param.Parameter(default=OrderedDict if OrderedDict else list, doc=
       """ The return type for a matched time interval that has been
           specified by slicing.  Constructor of the data type should
           accept a list of (key, value) tuples. When available,
           ordered dictionaries are appropriate.""")

    metadata = param.Dict(default={}, doc=
       """ Additional metadata to be associated with the Dataview """)

    def __init__(self, data, bounds, **kwargs):
        super(DataView,self).__init__(bounds=bounds, **kwargs)
        self._sorted_stack = [data]             # Sorted by timestamp
        self._timestamps = [self.timestamp]     # All the timestamps

    def _index_timestamp(self, timestamp):
        """
        Locate the leftmost value of the stack matching the timestamp.
        If not found, raises a ValueError. See Python documentation on
        bisect module for more information about this function.
        """
        i = bisect.bisect_left(self._timestamps, timestamp)
        if i != len(self._timestamps) and self._timestamps[i] == timestamp: return i
        raise ValueError


    def __getitem__(self, timeslice):
        """
        DataViews support slicing the available data over a time
        interval.  Supports the usual slicing semantics with an
        inclusive lower bound and an exclusive upper bound.
        """
        if self.timestamp is None:
            raise Exception('Time slicing disabled as initial timestamp not set.')

        if not isinstance(timeslice, slice):
            try:
                matched_index = self._index_timestamp(timeslice)
                return self._sorted_stack[matched_index]
            except:
                raise ValueError
        else:
            start, stop, step = timeslice.start, timeslice.stop, timeslice.step
            start_ind = None if (start is None) else bisect.bisect_left(self._timestamps, start)
            stop_ind = None if (stop is None) else bisect.bisect_left(self._timestamps, stop)
            timeslice = slice(start_ind, stop_ind, step)
            return self.interval_dtype(zip(self._timestamps[timeslice],
                                           self._sorted_stack[timeslice]))

    def __len__(self):
        return len(self._timestamps) if hasattr(self, '_timestamps') else 1


    def _copy_data(self):
        # Need to copy data but not sure of type.
        pass

    def record(self, data, timestamp):
        """
        Records the data with the given timestamp into the DataView.
        Data and timestamp may be provided as lists which will be
        zipped together to record multiple items of data at once.
        """

        try:     pairs = zip(data, timestamp)
        except:  pairs = [(data.copy(), timestamp)]

        for data, timestamp in pairs:
            # Keep _sorted_stack and _timestamp in sorted order.
            insert_index = bisect.bisect_left(self._timestamps, timestamp)
            self._sorted_stack.insert(insert_index, data.copy())
            self._timestamps.insert(insert_index, timestamp)


    def sample(self, coords):
        """
        Return the data at a specified coordinate point over the
        appropriate time slice.
        """
        return [data[self._coords2matrixidx(coords, data)] for data in self.data]

    def _coords2matrixidx(self, coords):
        raise NotImplementedError

    def view(self):
        """
        Return the requested view as a (data, bbox) tuple.  Provided
        for backward compatibility with the original Topographica
        SheetView model. It is now easier to access the data and
        bounds attributes directly.
        """
        return (self._sorted_stack[-1], self.bounds)



class Cartesian2D(DataView):
    """
    A dataview for data situated in a 2-dimensional Cartesian
    coordinate system.
    """

    def _coords2matrixidx(self, coords, arr):
        (l,b,r,t) = self.bounds.lbrt()
        (dim1, dim2) = arr.shape
        xdensity = dim1 / (r-l)
        ydensity = dim2 / (t-b)
        return SheetCoordinateSystem(self.bounds, xdensity, ydensity).sheet2matrixidx(*coords)

"""
NdIndexableMapping and NdMapping objects.
"""

__version__='$Revision$'

import param

try:
    from collections import OrderedDict
except:
    from odict import OrderedDict # pyflakes:ignore (try/except import)

map_type = OrderedDict

class AttrDict(dict):
    """
    A dictionary type object that supports attribute access (e.g. for IPython
    tab completion).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NdIndexableMapping(param.Parameterized):
    """
    An NdIndexableMapping is a type of mapping (like a dictionary or array)
    that uses fixed-length multidimensional keys. The effect is like an
    N-dimensional array, without requiring that the entire multidimensional
    space be populated.

    If the underlying type of data for each (key,value) pair also supports
    indexing (such as a dictionary, array, or list), fully qualified indexing
    can be used from the top level, with the first N dimensions of the index
    selecting a particular piece of data stored in the NdIndexableMapping
    object, and the remaining dimensions used to index into the underlying data.

    For instance, for an NdIndexableMapping x with dimensions "Year" and
    "Month" and an underlying data type that is a 2D floating-point array
    indexed by (r,c), a 2D array can be indexed with x[2000,3] and a single
    floating-point number may be indexed as x[2000,3,1,9].

    In practice, this class is typically only used as an abstract base class,
    because the NdMapping subclass extends it with a range of useful slicing
    methods for selecting subsets of the data. Even so, keeping the slicing
    support separate from the indexing and data storage methods helps make both
    classes easier to understand.
    """

    dimension_labels = param.List(default=[None], constant=True, doc="""
        The dimension_labels parameter accepts a list of the features
        along which the data will indexed.""")

    enforced_type = param.Parameter(default=None, constant=True)

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional labels to be associated with the Dataview.""")

    sorted = param.Boolean(default=True, doc="""
        Determines whether to keep internal data structure sorted, using
        all dimensions indices to sort on. Important if data is not added in
        a consistently increasing order but the order matters for plotting
        or other purposes.""")

    _deep_indexable = True

    def __init__(self, initial_items=None, **kwargs):
        self._data = map_type()

        kwargs, metadata = self.write_metadata(kwargs)

        super(NdIndexableMapping, self).__init__(metadata=metadata, **kwargs)

        self._next_ind = 0

        if isinstance(initial_items, tuple):
            self._add_item(initial_items[0], initial_items[1])
        elif initial_items is not None:
            self.update(map_type(initial_items))


    def write_metadata(self, kwargs):
        """
        Acts as a param override, placing any kwargs, which are not parameters
        into the metadata dictionary.
        """
        items = kwargs.items()
        if 'metadata' in kwargs:
            items = kwargs.pop('metadata').items() + items
        metadata = AttrDict(self.metadata, **dict([(k, v) for k, v in items
                                                   if k not in self.params()]))
        for key in metadata:
            kwargs.pop(key, None)
        return kwargs, metadata


    @property
    def timestamp(self):
        """
        Looks for the latest timestamp along the indexed dimensions, if 'Time'
        is one of the dimension labels, otherwise looks in the metadata or
        returns None, if none is defined.
        """
        if 'Time' in self.dimension_labels:
            return max([k[self.dimension_labels.index('Time')] if type(k) is tuple
                        else k for k in self.keys()])
        elif 'timestamp' in self.metadata:
            return self.metadata.timestamp
        else:
            return None


    @property
    def ndim(self):
        return len(self.dimension_labels)


    def _element_check(self, data):
        """
        Applies checks to individual data elements before they are inserted
        ensuring that they are of a certain type. Can be subclassed to implement
        further element restrictions.
        """
        if not isinstance(data, self.enforced_type):
            raise TypeError('{slf} does not accept {data} type, data elements have '
                            'to be a {restr}.'.format(slf=type(self).__name__,
                                                      data=type(data).__name__,
                                                      restr=self.enforced_type.__name__))


    def _add_item(self, dim_vals, data, sort=True):
        """
        Records data indexing it in the specified feature dimensions.
        """
        if self.enforced_type is not None: self._element_check(data)
        if not isinstance(dim_vals, tuple):
            dim_vals = (dim_vals,)
        if len(dim_vals) == self.ndim:
            self._update_item(dim_vals, data)
        else:
            KeyError('Key has to match number of dimensions.')
        if sort and self.sorted:
            self._data = map_type(sorted(self._data.items()))


    def _update_item(self, dim_vals, data):
        """
        Subclasses default method to allow updating of nested data structures
        rather than simply overriding them.
        """
        if dim_vals in self._data and hasattr(self._data[dim_vals], 'update'):
            self._data[dim_vals].update(data)
        else:
            self._data[dim_vals] = data


    def update(self, other):
        """
        Updates the NdMapping with another NdMapping or map_type
        instance, checking that they are indexed along the same number
        of dimensions.
        """
        for key, data in other.items():
            self._add_item(key, data, sort=False)
        if self.sorted:
            self._data = map_type(sorted(self._data.items()))


    def reindex(self, dimension_labels):
        """
        Create a new object with a re-ordered or reduced set of dimension
        labels. Accepts either a single dimension label or a list of chosen
        dimension labels.

        Reducing the number of dimension labels will discard information held in
        the dropped dimensions. All data values are accessible in the newly
        created object as the new labels must be sufficient to address each
        value uniquely.
        """
        if dimension_labels in self.dimension_labels:
            indices = [self.dim_index(dimension_labels)]
            dimension_labels = [dimension_labels]
        else:
            indices = [self.dim_index(el) for el in dimension_labels]

        keys = [tuple(k[i] for i in indices) for k in self._data.keys()]
        initial_items = map_type((k,v) for (k,v) in zip(keys, self._data.values()))

        if len(set(keys)) != len(keys):
            raise Exception("Given dimension labels not sufficient to address all values uniquely")

        return self.__class__(initial_items=initial_items,
                              dimension_labels=dimension_labels,
                              sorted=self.sorted,
                              **self.metadata)


    def dframe(self, value_label='data'):
        try:
            import pandas
        except ImportError:
            raise Exception("Cannot build a DataFrame without the pandas library.")
        labels = self.dimension_labels + [value_label]
        return pandas.DataFrame([dict(zip(labels, k+(v,))) for (k,v) in self._data.items()])


    def _split_index(self, key):
        """
        Splits key into map and data indices. If only map indices are supplied
        the data is passed an index of None.
        """
        if not isinstance(key, tuple):
            key = (key,)
        map_slice = key[:self.ndim]
        data_slice = key[self.ndim:] if len(key[self.ndim:]) > 0 else ()
        return map_slice, data_slice


    def __getitem__(self, key):
        """
        Allows indexing in the indexed dimensions, passing any additional
        indices to the data elements.
        """
        map_slice, data_slice = self._split_index(key)
        return self._dataslice(self._data[map_slice], data_slice)


    def _dataslice(self, data, indices):
        """
        Returns slice of data element if the item is deep indexable. Warns if
        attempting to slice an object that has not been declared deep indexable.
        """
        if hasattr(data, '_deep_indexable'):
            return data[indices]
        elif len(indices) > 0:
            self.warning('Cannot index into data element, extra data indices ignored.')
        return data


    def __setitem__(self, key, value):
        self._add_item(key, value)


    def __str__(self):
        return repr(self)


    @property
    def top(self):
        """"
        Returns the item highest data item along the map dimensions.
        """
        return self._data.values()[-1] if len(self.keys()) > 0 else None


    def dim_index(self, dimension_label):
        """
        Returns the tuple index of the requested dimension.
        """
        return self.dimension_labels.index(dimension_label)


    def keys(self):
        """
        Returns indices for all data elements.
        """
        if self.ndim == 1:
            return [k[0] for k in self._data.keys()]
        else:
            return self._data.keys()


    def values(self):
        return self._data.values()


    def items(self):
        return zip(self.keys(), self.values())


    def get(self, key, default=None):
        try:
            if key is None:
                return None
            return self[key]
        except:
            return default


    def __iter__(self):
        return self


    def next(self):
        """
        Implements the iterable interface, returning values unlike a standard
        dictionary.
        """
        if self._next_ind < len(self):
            val = self.values()[self._next_ind]
            self._next_ind += 1
            return val
        else:
            self._next_ind = 0
            raise StopIteration


    def __contains__(self, key):
        return key in self.keys()


    def __len__(self):
        return len(self._data)



class NdMapping(NdIndexableMapping):
    """
    NdMapping supports the same indexing semantics as NdIndexableMapping but
    also supports filtering of items using slicing ranges.
    """

    def __getitem__(self, indexslice):
        """
        Allows slicing operations along the map and data dimensions. If no data
        slice is supplied it will return all data elements, otherwise it will
        return the requested slice of the data.
        """
        if indexslice in [Ellipsis, ()]:
            return self

        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)
        conditions = self._generate_conditions(map_slice)

        if all(not isinstance(el, slice) for el in map_slice):
            return self._dataslice(self._data[map_slice], data_slice)
        else:
            items = [(k, self._dataslice(v, data_slice)) for k, v
                     in self._data.items() if self._conjunction(k, conditions)]
            if self.ndim == 1:
                items = [(k[0], v) for (k, v) in items]
            return self.__class__(initial_items=items, metadata=self.metadata,
                                  dimension_labels=self.dimension_labels, sorted=self.sorted)


    def _transform_indices(self, indices):
        """
        Identity function here but subclasses can implement transforms of the
        dimension indices from one coordinate system to another.
        """
        return indices


    def _generate_conditions(self, map_slice):
        """
        Generates filter conditions used for slicing the data structure.
        """
        conditions = []
        for dim in map_slice:
            if isinstance(dim, slice):
                if dim == slice(None):
                    conditions.append(self._all_condition(dim))
                elif dim.start is None:
                    conditions.append(self._upto_condition(dim))
                elif dim.stop is None:
                    conditions.append(self._from_condition(dim))
                else:
                    conditions.append(self._range_condition(dim))
            elif dim is Ellipsis:
                conditions.append(self._all_condition(dim))
            else:
                conditions.append(self._value_condition(dim))
        return conditions


    def _value_condition(self, value):
        return lambda x: x == value


    def _range_condition(self, slice):
        if slice.step is not None: raise Exception('Ignoring step value.')
        return lambda x: slice.start <= x < slice.stop


    def _upto_condition(self, slice):
        if slice.step is not None: raise Exception('Ignoring step value.')
        return lambda x: x < slice.stop


    def _from_condition(self, slice):
        if slice.step is not None: raise Exception('Ignoring step value.')
        return lambda x: x > slice.start


    def _all_condition(self, ellipsis):
        return lambda x: True


    def _conjunction(self, key, conditions):
        conds = []
        for i, cond in enumerate(conditions):
            conds.append(cond(key[i]))
        return all(conds)


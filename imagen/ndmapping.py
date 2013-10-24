import param

try:
    from collections import OrderedDict
except:
    try:
        from ordereddict import OrderedDict
    except:
        raise ImportError("OrderedDict could not be imported. For Python "
                          "versions <2.7, run the command 'pip install "
                          "ordereddict' or install the package through your"
                          "repository manager.")

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
    An NdIndexableMapping extends mapping indexing semantics by supporting named
    indices (labelled dimensions) and by allowing indexing directly into the
    contained items (data dimensions). Indexing is multi-dimensional,
    accepting as many key values as there are labelled and data dimensions.

    Direct indexing is achieved by passing any indices, in excess of the map
    dimensions down to the data elements.

    Add example
    """

    dimension_labels = param.List(default=[None], constant=True, doc="""
        The dimension_labels parameter accepts a list of the features
        along which the data will indexed.""")

    metadata = param.Dict(default=AttrDict(), doc="""
        Additional labels to be associated with the Dataview.""")

    sorted = param.Boolean(default=True, doc="""
        Determines whether to keep internal data structure sorted. May be
        important for data, where the order of keys matters.""")

    __abstract = True


    def __call__(self, *args, **kwargs):
        """
        Supports slicing with keyword arguments corresponding to the
        dimension labels.
        """
        pass


    def __init__(self, initial_items=None, **kwargs):
        self._data = map_type()

        kwargs, metadata = self.write_metadata(kwargs)

        super(NdIndexableMapping, self).__init__(metadata=metadata, **kwargs)

        self.ndim = len(self.dimension_labels)

        if isinstance(initial_items, map_type):
            self.update(initial_items)
        if isinstance(initial_items, tuple):
            self[initial_items[0]] = initial_items[1]


    def write_metadata(self, kwargs):
        """
        Acts as a param override, placing any kwargs, which are not parameters
        into the metadata dictionary.
        """
        metadata = AttrDict(self.metadata,
                            **dict([(kw, val) for kw, val in kwargs.items()
                                    if kw not in self.params()]))
        for key in metadata:
            kwargs.pop(key)
        return kwargs, metadata


    @property
    def timestamp(self):
        if 'Time' in self.dimension_labels:
            return max([k[self.dimension_labels.index('Time')] if type(k) is tuple
                        else k for k in self.keys()])
        elif 'timestamp' in self.metadata:
            return self.metadata.timestamp
        else:
            return None


    def _add_item(self, feature_values, data, sort=True):
        """
        Records data indexing it in the specified feature dimensions.
        """
        if not isinstance(feature_values, tuple):
            feature_values = (feature_values,)
        if len(feature_values) == self.ndim:
            self._update_item(feature_values, data)
        else:
            KeyError('Key has to match number of dimensions.')
        if sort and self.sorted:
            self._data = map_type(sorted(self._data.items()))


    def _update_item(self, coords, data):
        """
        Subclasses default method to allow updating of nested data structures
        rather than simply overriding them.
        """
        if coords in self._data and hasattr(self._data[coords], 'update'):
            self._data[coords].update(data)
        else:
            self._data[coords] = data


    def update(self, other):
        """
        Updates the IndexMap with another IndexMap, checking that they
        are indexed along the same map dimensions.
        """
        for key, data in other[...].items():
            self._add_item(key, data, sort=False)
        if self.sorted:
            self._data = map_type(sorted(self._data.items()))


    def _split_slice(self, key):
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
        Allows indexing in the map dimensions and slicing along the data
        dimension.
        """
        map_slice, data_slice = self._split_slice(key)
        return self._data[map_slice][data_slice]


    def __setitem__(self, key, value):
        self._add_item(key, value)


    @property
    def top(self):
        """"
        Returns the item highest data item along the map dimensions.
        """
        return self._data.values()[-1] if len(self.keys()) > 0 else None


    def keys(self):
        """
        Returns indices for all data elements.
        """
        if self.ndim == 1:
            return [k[0] for k in self._data.keys()]
        else:
            return self._data.keys()


    def get(self, key, default=None):
        try:
            if key is None:
                return None
            return self[key]
        except:
            return default


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

        if indexslice is Ellipsis:
            return self._data
        elif indexslice is ():
            return self

        map_slice, data_slice = self._split_slice(indexslice)
        map_slice = self._transform_indices(map_slice)
        conditions = self._generate_conditions(map_slice)

        if all(not isinstance(el, slice) for el in map_slice):
            return self._data[map_slice][data_slice]
        else:
            return map_type((k, v[data_slice]) for k, v in self._data.items()
                            if self._conjunction(k, conditions))


    def _transform_indices(self, indices):
        """
        Hook to map indices from a continuous space to a discrete set of keys.
        Identity function here but subclasses can implement mapping from one
        coordinate system to another.
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


from itertools import groupby
import string
import numpy as np

import param
from views import SheetView, SheetLayer, SheetOverlay, SheetLines, SheetStack, SheetPoints, GridLayout, ProjectionGrid

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
import matplotlib.gridspec as gridspec


class Plot(param.Parameterized):
    """
    A Plot object returns either a matplotlib figure object (when
    called or indexed) or a matplotlib animation object as
    appropriate. Plots take view objects such as SheetViews,
    SheetLines or SheetPoints as inputs and plots them in the
    appropriate format. As views may vary over time, all plots support
    animation via the anim() method.
    """

    size = param.NumericTuple(default=(5,5), doc="""
      The matplotlib figure size in inches.""")

    show_axes = param.Boolean(default=True, doc="""
      Whether to show labelled axes for the plot.""")

    show_grid = param.Boolean(default=False, doc="""
      Whether to show a Cartesian grid on the plot""")


    def __init__(self, **kwargs):
        super(Plot, self).__init__(**kwargs)
        # List of handles to matplotlib objects for animation update
        self.handles = {'fig':None}


    def _title_fields(self, stack):
        """
        Returns the formatting fields in the title string supplied by
        the view object.
        """
        if stack.title is None:  return []
        parse = list(string.Formatter().parse(stack.title))
        return [f for f in zip(*parse)[1] if f is not None]


    def _format_title(self, stack, index):
        """
        Format a title string based on the keys/values of the view
        stack.
        """
        if stack.values()[index].title is not None:
            return stack.values()[index].title
        labels = stack.dimension_labels
        vals = stack.keys()[index]
        if not isinstance(vals, tuple): vals = (vals,)
        fields = self._title_fields(stack)
        if fields == []:
            return stack.title if stack.title else ''
        label_map = dict(('label%d' % i, l) for (i,l) in enumerate(labels))
        val_map =   dict(('value%d' % i, l) for (i,l) in enumerate(vals))
        format_items = dict(label_map,**val_map)
        if not set(fields).issubset(format_items):
            raise Exception("Cannot format")
        return stack.title.format(**format_items)


    def _stack_type(self, view, element_type=SheetLayer):
        """
        Helper method that ensures a given view is always returned as
        an imagen.SheetStack object.
        """
        if not isinstance(view, SheetStack):
            stack = SheetStack(initial_items=(0,view), title=view.title)
            if self._title_fields(stack) != []:
                raise Exception('Can only format title string for animation and stacks.')
        else:
            stack = view

        if not stack.type == element_type:
            raise TypeError("Requires View, Animation or Stack of type %s" % element_type)
        return stack


    def _axis(self, axis, title, xlabel=None, ylabel=None, lbrt=None):
        "Return an axis which may need to be initialized from a new figure."
        if axis is None:
            fig = plt.figure()
            self.handles['fig'] = fig
            fig.set_size_inches(list(self.size))
            axis = fig.add_subplot(111)
            axis.set_aspect('equal')
        if not self.show_axes:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        elif self.show_grid:
            axis.get_xaxis().grid(True)
            axis.get_yaxis().grid(True)

        if lbrt is not None:
            (l,b,r,t) = lbrt
            axis.set_xlim((l,r))
            axis.set_ylim((b,t))

        self.handles['title'] = axis.set_title(title)
        if xlabel: plt.xlabel('x')
        if ylabel: plt.ylabel('y')
        return axis


    def __getitem__(self, frame):
        """
        Get the matplotlib figure at the given frame number.
        """
        if frame > len(self):
            self.warn("Showing last frame available: %d" % len(self))
        fig = self()
        self.update_frame(frame)
        return fig


    def anim(self, start=0, stop=None, fps=30):
        """
        Method to return an Matplotlib animation. The start and stop
        frames may be specified as well as the fps.
        """
        figure = self()
        frames = range(len(self))[slice(start, stop, 1)]
        anim = animation.FuncAnimation(figure, self.update_frame,
                                       frames=frames,
                                       interval = 1000.0/fps)
        # Close the figure handle
        plt.close(figure)
        return anim


    def update_frame(self, n):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        n = n  if n < len(self) else len(self) - 1
        raise NotImplementedError


    def __len__(self):
        """
        Returns the total number of available frames.
        """
        raise NotImplementedError


    def __call__(self, ax=False, zorder=0):
        """
        Return a matplotlib figure.
        """
        raise NotImplementedError



class SheetLinesPlot(Plot):

    def __init__(self, contours, **kwargs):
        self._stack = self._stack_type(contours, SheetLines)
        super(SheetLinesPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())
        lines = self._stack.top
        line_segments = LineCollection([], zorder=zorder, **lines.style)
        line_segments.set_paths(lines.data)
        self.handles['line_segments'] = line_segments
        ax.add_collection(line_segments)
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        contours = self._stack.values()[n]
        self.handles['line_segments'].set_paths(contours.data)
        self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetPointsPlot(Plot):

    def __init__(self, contours, **kwargs):
        self._stack = self._stack_type(contours, SheetPoints)
        super(SheetPointsPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())
        points = self._stack.top
        scatterplot = plt.scatter(points.data[:,0], points.data[:,1],
                                  zorder=zorder, **points.style)
        self.handles['scatter'] = scatterplot
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        points = self._stack.values()[n]
        self.handles['scatter'].set_offsets(points.data)
        self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetViewPlot(Plot):

    colorbar = param.ObjectSelector(default='horizontal',
                                    objects=['horizontal','vertical', None],
        doc="""The style of the colorbar if applicable. """)


    def __init__(self, sheetview, **kwargs):
        self._stack = self._stack_type(sheetview, SheetView)
        super(SheetViewPlot, self).__init__(**kwargs)


    def toggle_colorbar(self, bar, cmax):
        visible = not (cmax == 0.0)
        bar.set_clim(vmin=0.0, vmax=cmax if visible else 1.0)
        elements = (bar.ax.get_xticklines()
                    + bar.ax.get_ygridlines()
                    + bar.ax.get_children())
        for el in elements:
            el.set_visible(visible)
        bar.draw_all()


    def __call__(self, axis=None, zorder=0):
        sheetview = self._stack.top
        title = self._format_title(self._stack, -1)
        (l,b,r,t) = self._stack.bounds.lbrt()
        ax = self._axis(axis, title, 'x','y', (l,b,r,t))

        cmap = 'hsv' if (sheetview.cyclic_range is not None) else 'gray'
        cmap = sheetview.style.get('cmap', cmap)
        im = ax.imshow(sheetview.data, extent=[l,r,b,t],
                       cmap=cmap, zorder=zorder,
                       interpolation='nearest')
        self.handles['im'] = im

        normalization = sheetview.data.max()
        cyclic_range = sheetview.cyclic_range
        im.set_clim([0.0, cyclic_range if cyclic_range else normalization])

        if self.colorbar is not None:
            np.seterr(divide='ignore')
            bar = plt.colorbar(im, ax=ax,
                               orientation=self.colorbar)
            np.seterr(divide='raise')
            self.toggle_colorbar(bar, normalization)
            self.handles['bar'] = bar
        else:
            plt.tight_layout()

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        im = self.handles.get('im',None)
        bar = self.handles.get('bar',None)

        sheetview = self._stack.values()[n]
        cmap = 'hsv' if (sheetview.cyclic_range is not None) else 'gray'
        im.set_cmap(sheetview.style.get('cmap', cmap))
        im.set_data(sheetview.data)
        normalization = sheetview.data.max()
        cmax = max([normalization, sheetview.cyclic_range])
        im.set_clim([0.0, cmax])
        if self.colorbar: self.toggle_colorbar(bar, cmax)

        self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetPlot(Plot):
    """
    A generic plot that visualizes SheetOverlays which themselves may
    contain SheetLayers of type SheetView, SheetPoints or SheetLine
    objects.
    """
    def __init__(self, overlays, **kwargs):
        self._stack = self._stack_type(overlays, SheetOverlay)
        self.plots = []
        super(SheetPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())

        for zorder, stack in enumerate(self._stack.split()):
            plotype = viewmap[stack.type]
            plot = plotype(stack, size=self.size, show_axes=self.show_axes)
            plot(ax, zorder=zorder)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


    def __len__(self):
        return len(self._stack)


class GridLayoutPlot(Plot):
    """
    Plot a group of views in a grid layout based on a GridLayout view
    object.
    """

    roi = param.Boolean(default=False, doc="""
      Whether to apply the ROI to each element of the grid.""")

    show_axes= param.Boolean(default=True, constant=True, doc="""
      Whether to show labelled axes for individual subplots.""")

    def __init__(self, grid, **kwargs):

        if not isinstance(grid, GridLayout):
            raise Exception("GridLayoutPlot only accepts GridLayouts.")

        self.grid = grid
        self.subplots = []
        self.rows, self.cols = grid.shape
        self._gridspec = gridspec.GridSpec(self.rows, self.cols)
        super(GridLayoutPlot, self).__init__(**kwargs)


    def __call__(self, axis=None):
        ax = self._axis(axis, '', '','', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        coords = [(r,c) for c in range(self.cols) for r in range(self.rows)]

        self.subplots = []
        for (r,c) in coords:
            view = self.grid.get((r,c),None)
            if view is not None:
                subax = plt.subplot(self._gridspec[r,c])
                subview = view.roi if self.roi else view
                vtype = subview.type if isinstance(subview,SheetStack) else subview.__class__
                subplot = viewmap[vtype](subview, show_axes=self.show_axes)
            self.subplots.append(subplot)
            subplot(subax)

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        for subplot in self.subplots:
            subplot.update_frame(n)


    def __len__(self):
        return len(self.grid)



class ProjectionGridPlot(Plot):
    """
    ProjectionGridPlot evenly spaces out plots of individual projections on
    a grid, even when they differ in size. The projections can be situated
    or an ROI can be applied to each element. Since this class uses a single
    axis to generate all the individual plots it is much faster than the
    equivalent using subplots.
    """

    border = param.Number(default=10, doc="""
        Aggregate border as a fraction of total plot size.""")

    situate = param.Boolean(default=False, doc="""
        Determines whether to situate the projection in the full bounds or
        apply the ROI.""")

    def __init__(self, grid, **kwargs):
        if not isinstance(grid, ProjectionGrid):
            raise Exception("ProjectionGridPlot only accepts ProjectionGrids.")
        self.grid = grid
        self.rows, self.cols = grid.shape
        super(ProjectionGridPlot, self).__init__(**kwargs)


    def __call__(self, axis=None):
        ax = self._axis(axis, '', '','', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        grid_shape = [[v for (k,v) in col[1]] for col in groupby(self.grid.items(),
                                                                 lambda (k,v): k[0])]
        width, height, b_w, b_h = self._compute_borders(grid_shape)

        plt.xlim(0, width)
        plt.ylim(0, height)

        cmap = self.grid.metadata.get('cmap', 'gray')
        self.handles['projs'] = []
        x, y = b_w, b_h
        for row in grid_shape:
            for view in row:
                w, h = self._get_dims(view)
                data = view.top.data if self.situate else view.top.roi.data
                self.handles['projs'].append(plt.imshow(data, extent=(x,x+w, y, y+h),
                                                        interpolation='nearest',
                                                        cmap=cmap))
                y += h + b_h
            y = b_h
            x += w + b_w

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for i, plot in enumerate(self.handles['projs']):
            view = self.grid.values()[i].values()[n]
            data = view.data if self.situate else view.roi.data
            plot.set_data(data)


    def _get_dims(self, view):
        l,b,r,t = view.bounds.lbrt() if self.situate else view.roi.bounds.lbrt()
        return (r-l, t-b)


    def _compute_borders(self, grid_shape):
        height = 0
        for view in grid_shape[0]:
            height += self._get_dims(view)[1]

        width = 0
        for view in [row[0] for row in grid_shape]:
            width += self._get_dims(view)[0]

        border_width = (width/10)/(self.cols+1)
        border_height = (height/10)/(self.rows+1)
        width += width/10
        height += height/10

        return width, height, border_width, border_height


    def __len__(self):
        return len(self.grid)


viewmap = {SheetView:SheetViewPlot,
           SheetPoints:SheetPointsPlot,
           SheetLines:SheetLinesPlot,
           SheetOverlay:SheetPlot}

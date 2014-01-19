import matplotlib.pyplot as plt
try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile

from patterngenerator import PatternGenerator
from plots import Plot, GridLayoutPlot, viewmap, ProjectionGridPlot
from views import SheetStack, SheetLayer, GridLayout, ProjectionGrid


GIF_TAG = "<img src='data:image/gif;base64,{0}'/>"

x264_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def x264_available():
    try:
        with NamedTemporaryFile(suffix='.mp4') as f:
            a = animation.FuncAnimation(plt.figure(), lambda x: x, frames=[0,1])
            a.save(f.name, extra_args=['-vcodec', 'libx264'])
        return True
    except:
        return False

VIDEO_FORMAT='x264' if x264_available() else 'gif'
GIF_FPS = 10

def opts(obj, additional_opts=[]):
    default_options = ['size']
    options = default_options + additional_opts
    return dict((k, obj.metadata.get(k)) for k in options if (k in obj.metadata))


def anim_opts(obj, additional_opts=[]):
    default_options = ['fps']
    options = default_options + additional_opts
    return dict((k, obj.metadata.get(k)) for k in options if (k in obj.metadata))


def animation_gif(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.gif') as f:
            anim.save(f.name, writer='imagemagick', fps=GIF_FPS)
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return GIF_TAG.format(anim._encoded_video)


def animation_x264(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, extra_args=['-vcodec', 'libx264']) # fps=20
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return x264_TAG.format(anim._encoded_video)


def HTML_animation(plot, view):
    anim_kwargs =  dict((k, view.metadata[k]) for k in ['fps']
                        if (k in view.metadata))
    fmt = view.metadata.get('video_format', VIDEO_FORMAT)
    return animation_to_HTML(plot.anim(**anim_kwargs), fmt)


def animation_to_HTML(anim, video_format=None):
    video_format = VIDEO_FORMAT if (video_format is None) else video_format
    assert video_format in ['x264', 'gif']
    writers = animation.writers.avail
    if video_format=='x264' and ('ffmpeg' in writers):
        try:     return animation_x264(anim)
        except:  pass
    if video_format=='gif' and 'imagemagick' in writers:
        try:     return animation_gif(anim)
        except:  pass
    return "<b>Could not generate %s animation</b>" % video_format


def figure_display(fig, size=None, format='svg', message=None):
    if size is not None:
        inches = size / float(fig.dpi)
        fig.set_size_inches(inches, inches)
    prefix = 'data:image/png;base64,'
    b64 = prefix + print_figure(fig, 'png').encode("base64")
    if size is not None:
        html = "<img height='%d' width='%d' src='%s' />" % (size, size, b64)
    else:
        html = "<img src='%s' />" % b64
    plt.close(fig)
    return html if (message is None) else '<b>%s</b></br>%s' % (message, html)


def sheetstack_display(stack, size=256, format='svg'):
    if not isinstance(stack, SheetStack): return None
    stackplot = viewmap[stack.type](stack, **opts(stack))
    if len(stack)==1:
        fig =  stackplot()
        return figure_display(fig)

    try:
        return HTML_animation(stackplot, stack)
    except:
        message = ('Cannot import matplotlib.animation' if animation is None
                   else 'Failed to generate matplotlib animation')
        fig =  stackplot()
        return figure_display(fig, message=message)


def layout_display(grid, size=256, format='svg'):
    if not isinstance(grid, GridLayout): return None
    grid_size = grid.shape[1]*Plot.size[1], grid.shape[0]*Plot.size[0]
    gridplot = GridLayoutPlot(grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)

    try:
        return HTML_animation(gridplot, grid)
    except:
        message = ('Cannot import matplotlib.animation' if animation is None
                   else 'Failed to generate matplotlib animation')
        fig =  gridplot()
        return figure_display(fig, message=message)


def projection_display(grid, size=256, format='svg'):
    if not isinstance(grid, ProjectionGrid): return None
    size_factor = 0.25
    grid_size = size_factor*grid.shape[1]*Plot.size[1], size_factor*grid.shape[0]*Plot.size[0]
    gridplot = ProjectionGridPlot(grid, **dict(opts(grid), size=grid_size))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)
    try:
        return HTML_animation(gridplot, grid)
    except:
        message = ('Cannot import matplotlib.animation' if animation is None
                   else 'Failed to generate matplotlib animation')
        fig = gridplot()
        return figure_display(fig, message=message)


def sheetlayer_display(view, size=256, format='svg'):
    if not isinstance(view, SheetLayer): return None
    fig = viewmap[view.__class__](view, **opts(view))()
    return figure_display(fig)

def update_matplotlib_rc():
    """
    Default changes to the matplotlib rc used by IPython Notebook.
    """
    import matplotlib
    rc= {'figure.figsize': (6.0,4.0),
         'figure.facecolor': 'white',
         'figure.edgecolor': 'white',
         'font.size': 10,
         'savefig.dpi': 72,
         'figure.subplot.bottom' : .125
         }
    matplotlib.rcParams.update(rc)


message = """Welcome to the Imagen IPython extension! (http://ioam.github.io/imagen/)"""

_loaded = False

def load_ipython_extension(ip, verbose=True):

    if verbose:
        print message
        if VIDEO_FORMAT=='gif':
            gif_fps = "imagen.ipython.GIF_FPS=%s" % GIF_FPS
            print "[Animations rendered in GIF format: %s]" % gif_fps

    global _loaded
    if not _loaded:
        _loaded = True
        PatternGenerator.xdensity = 256
        PatternGenerator.ydensity = 256
        html_formatter = ip.display_formatter.formatters['text/html']
        html_formatter.for_type_by_name('matplotlib.animation', 'FuncAnimation', animation_to_HTML)
        html_formatter.for_type(SheetLayer, sheetlayer_display)
        html_formatter.for_type(SheetStack, sheetstack_display)
        html_formatter.for_type(GridLayout, layout_display)
        html_formatter.for_type(ProjectionGrid, projection_display)

        update_matplotlib_rc()

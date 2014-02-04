import matplotlib.pyplot as plt

try:    from matplotlib import animation
except: animation = None

from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile

from patterngenerator import PatternGenerator
from plots import Plot, GridLayoutPlot, viewmap
from views import SheetStack, SheetLayer, GridLayout

video_format='x264'  # Either 'x264' or 'gif'
GIF_FPS = 3

GIF_TAG = "<img src='data:image/gif;base64,{0}'/>"

x264_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

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


def animation_to_HTML(anim):
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
        return animation_to_HTML(stackplot.anim(**anim_opts(stack)))
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
        return animation_to_HTML(gridplot.anim(**anim_opts(grid)))
    except:
        message = ('Cannot import matplotlib.animation' if animation is None
                   else 'Failed to generate matplotlib animation')
        fig =  stackplot()
        return figure_display(fig, message=message)


def sheetlayer_display(view, size=256, format='svg'):
    if not isinstance(view, SheetLayer): return None
    fig = viewmap[view.__class__](view, **opts(view))()
    return figure_display(fig)


_loaded = False

def load_ipython_extension(ip):
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

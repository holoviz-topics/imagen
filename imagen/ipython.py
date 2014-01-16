import matplotlib.pyplot as plt
from IPython.core.pylabtools import print_figure

from tempfile import NamedTemporaryFile

from patterngenerator import PatternGenerator
from plots import GridLayoutPlot, viewmap
from views import SheetStack, SheetLayer, GridLayout

VIDEO_TAG = """<video controls>
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


def animation_to_HTML(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, extra_args=['-vcodec', 'libx264']) # fps=20
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)


def figure_display(fig, size=256, format='svg'):
    inches = size / float(fig.dpi)
    fig.set_size_inches(inches, inches)
    prefix = 'data:image/png;base64,'
    b64 = prefix + print_figure(fig, 'png').encode("base64")
    html = "<img height='%d' width='%d' src='%s' />" % (size, size, b64)
    plt.close(fig)
    return html


def sheetstack_display(stack,size=256, format='svg'):
    if not isinstance(stack, SheetStack): return None
    stackplot = viewmap[stack.type](stack, **opts(stack))
    if len(stack)==1:
        fig =  stackplot()
        return figure_display(fig)
    else:
        return animation_to_HTML(stackplot.anim(**anim_opts(stack)))


def sheetlayer_display(view, size=256, format='svg'):
    if not isinstance(view, SheetLayer): return None
    fig = viewmap[view.__class__](view, **opts(view))()
    return figure_display(fig)


def layout_display(grid, size=256, format='svg'):
    if not isinstance(grid, GridLayout): return None

    gridplot = GridLayoutPlot(grid, **opts(grid))
    if len(grid)==1:
        fig =  gridplot()
        return figure_display(fig)
    else:
        return animation_to_HTML(gridplot.anim(**anim_opts(grid)))



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

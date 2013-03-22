from PIL import Image
from io import BytesIO
_loaded = False

def display_pil_image(im):
    """Generate PNG data for IPython display"""
    b = BytesIO()
    im.save(b, format='png')
    data = b.getvalue()
    ip_img = display.Image(data=data, format='png', embed=True)
    return ip_img._repr_png_()

def load_ipython_extension(ip):
    from IPython.core import display
    global _loaded

    if not _loaded:
        _loaded = True
        ip = get_ipython()

        # register display func with PNG formatter:
        pil_formatter = ip.display_formatter.formatters['image/png']
        pil_formatter.for_type(Image.Image, display_pil_image)


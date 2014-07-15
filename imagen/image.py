"""
PatternGenerators based on bitmap images stored in files.

Requires the Python Imaging Library (PIL). In general, the pillow fork
of PIL is recommended as it is being actively maintained and works
with Python 3.
"""

# StringIO.StringIO is *not* the same as io.StringIO:
# https://mail.python.org/pipermail/python-list/2013-May/648080.html

# In short, the former accepts bytes whereas the latter only accepts
# unicode. In Python 3, BytesIO may be used with pillow safely.

try:
    from StringIO import StringIO as BytesIO
except:
    from io import BytesIO

from PIL import Image
from PIL import ImageOps

import numpy as np

import param
from param.parameterized import overridable_property

from dataviews.sheetviews import BoundingBox, SheetCoordinateSystem
from .patterngenerator import PatternGenerator, Constant
from .transferfn import DivisiveNormalizeLinf, TransferFn

from os.path import splitext

import numbergen

from .colorspaces import color_conversion


class ImageSampler(param.Parameterized):
    """
    A class of objects that, when called, sample an image.
    """
    __abstract=True

    def _get_image(self):
        # CB: In general, might need to consider caching to avoid
        # loading of image/creation of scs and application of wpofs
        # every time/whatever the sampler does to set up the image
        # before sampling
        return self._image

    def _set_image(self,image):
        self._image = image

    def _del_image(self):
        del self._image

    # As noted by JP in FastImageSampler, this isn't easy to figure out.
    def __call__(self,image,x,y,sheet_xdensity,sheet_ydensity,width=1.0,height=1.0):
        raise NotImplementedError

    image = overridable_property(_get_image,_set_image,_del_image)



# CEBALERT: ArraySampler?
class PatternSampler(ImageSampler):
    """
    When called, resamples - according to the size_normalization
    parameter - an image at the supplied (x,y) sheet coordinates.

    (x,y) coordinates outside the image are returned as the background
    value.
    """
    whole_pattern_output_fns = param.HookList(class_=TransferFn,default=[],doc="""
        Functions to apply to the whole image before any sampling is done.""")

    background_value_fn = param.Callable(default=None,doc="""
        Function to compute an appropriate background value. Must accept
        an array and return a scalar.""")

    size_normalization = param.ObjectSelector(default='original',
        objects=['original','stretch_to_fit','fit_shortest','fit_longest'],
        doc="""
        Determines how the pattern is scaled initially, relative to the
        default retinal dimension of 1.0 in sheet coordinates:

        'stretch_to_fit': scale both dimensions of the pattern so they
        would fill a Sheet with bounds=BoundingBox(radius=0.5) (disregards
        the original's aspect ratio).

        'fit_shortest': scale the pattern so that its shortest dimension
        is made to fill the corresponding dimension on a Sheet with
        bounds=BoundingBox(radius=0.5) (maintains the original's aspect
        ratio, filling the entire bounding box).

        'fit_longest': scale the pattern so that its longest dimension is
        made to fill the corresponding dimension on a Sheet with
        bounds=BoundingBox(radius=0.5) (maintains the original's
        aspect ratio, fitting the image into the bounding box but not
        necessarily filling it).

        'original': no scaling is applied; each pixel of the pattern
        corresponds to one matrix unit of the Sheet on which the
        pattern being displayed.""")

    def _get_image(self):
        return self.scs.activity

    def _set_image(self,image):
        # Stores a SheetCoordinateSystem with an activity matrix
        # representing the image
        if not isinstance(image,np.ndarray):
            image = np.array(image,np.float)

        rows,cols = image.shape
        self.scs = SheetCoordinateSystem(xdensity=1.0,ydensity=1.0,
                                         bounds=BoundingBox(points=((-cols/2.0,-rows/2.0),
                                                                    ( cols/2.0, rows/2.0))))
        self.scs.activity=image

    def _del_image(self):
        self.scs = None


    def __call__(self, image, x, y, sheet_xdensity, sheet_ydensity, width=1.0, height=1.0):
        """
        Return pixels from the supplied image at the given Sheet (x,y)
        coordinates.

        The image is assumed to be a NumPy array or other object that
        exports the NumPy buffer interface (i.e. can be converted to a
        NumPy array by passing it to numpy.array(), e.g. Image.Image).
        The whole_pattern_output_fns are applied to the image before
        any sampling is done.

        To calculate the sample, the image is scaled according to the
        size_normalization parameter, and any supplied width and
        height. sheet_xdensity and sheet_ydensity are the xdensity and
        ydensity of the sheet on which the pattern is to be drawn.
        """
        # CEB: could allow image=None in args and have 'if image is
        # not None: self.image=image' here to avoid re-initializing the
        # image.
        self.image=image

        for wpof in self.whole_pattern_output_fns:
            wpof(self.image)
        if not self.background_value_fn:
            self.background_value = 0.0
        else:
            self.background_value = self.background_value_fn(self.image)

        pattern_rows,pattern_cols = self.image.shape

        if width==0 or height==0 or pattern_cols==0 or pattern_rows==0:
            return np.ones(x.shape)*self.background_value

        # scale the supplied coordinates to match the pattern being at density=1
        x=x*sheet_xdensity # deliberately don't operate in place (so as not to change supplied x & y)
        y=y*sheet_ydensity

        # scale according to initial pattern size_normalization selected (size_normalization)
        self.__apply_size_normalization(x,y,sheet_xdensity,sheet_ydensity,self.size_normalization)

        # scale according to user-specified width and height
        x/=width
        y/=height

        # now sample pattern at the (r,c) corresponding to the supplied (x,y)
        r,c = self.scs.sheet2matrixidx(x,y)
        # (where(cond,x,y) evaluates x whether cond is True or False)
        r.clip(0,pattern_rows-1,out=r)
        c.clip(0,pattern_cols-1,out=c)
        left,bottom,right,top = self.scs.bounds.lbrt()
        return np.where((x>=left) & (x<right) & (y>bottom) & (y<=top),
                           self.image[r,c],
                           self.background_value)


    def __apply_size_normalization(self,x,y,sheet_xdensity,sheet_ydensity,size_normalization):
        pattern_rows,pattern_cols = self.image.shape

        # Instead of an if-test, could have a class of this type of
        # function (c.f. OutputFunctions, etc)...
        if size_normalization=='original':
            return

        elif size_normalization=='stretch_to_fit':
            x_sf,y_sf = pattern_cols/sheet_xdensity, pattern_rows/sheet_ydensity
            x*=x_sf; y*=y_sf

        elif size_normalization=='fit_shortest':
            if pattern_rows<pattern_cols:
                sf = pattern_rows/sheet_ydensity
            else:
                sf = pattern_cols/sheet_xdensity
            x*=sf;y*=sf

        elif size_normalization=='fit_longest':
            if pattern_rows<pattern_cols:
                sf = pattern_cols/sheet_xdensity
            else:
                sf = pattern_rows/sheet_ydensity
            x*=sf;y*=sf




def edge_average(a):
    "Return the mean value around the edge of an array."

    if len(np.ravel(a)) < 2:
        return float(a[0])
    else:
        top_edge = a[0]
        bottom_edge = a[-1]
        left_edge = a[1:-1,0]
        right_edge = a[1:-1,-1]

        edge_sum = np.sum(top_edge) + np.sum(bottom_edge) + np.sum(left_edge) + np.sum(right_edge)
        num_values = len(top_edge)+len(bottom_edge)+len(left_edge)+len(right_edge)

        return float(edge_sum)/num_values



class FastImageSampler(ImageSampler):
    """
    A fast-n-dirty image sampler using Python Imaging Library
    routines.  Currently this sampler doesn't support user-specified
    size_normalization or cropping but rather simply scales and crops
    the image to fit the given matrix size without distorting the
    aspect ratio of the original picture.
    """

    sampling_method = param.Integer(default=Image.NEAREST,doc="""
       Python Imaging Library sampling method for resampling an image.
       Defaults to Image.NEAREST.""")

    def _set_image(self,image):
        if not isinstance(image,Image.Image):
            self._image = Image.new('L',image.shape)
            self._image.putdata(image.ravel())
        else:
            self._image = image

    def __call__(self, image, x, y, sheet_xdensity, sheet_ydensity, width=1.0, height=1.0):
        self.image=image

        # JPALERT: Right now this ignores all options and just fits the image into given array.
        # It needs to be fleshed out to properly size and crop the
        # image given the options. (maybe this class needs to be
        # redesigned?  The interface to this function is pretty inscrutable.)
        im = ImageOps.fit(self.image,x.shape,self.sampling_method)
        return np.array(im,dtype=np.float)



class ChannelTransform(param.ParameterizedFunction):
    """
    A ChannelTransform is a parameterized function that takes channels
    as input (a list of arrays) and transforms their contents in some way.
    """

    __abstract = True

    def __call__(self, p, channels):
        raise NotImplementedError



class GenericImage(PatternGenerator):
    """
    Generic 2D image generator with support for multiple channels.

    Subclasses should override the _get_image method to produce the
    image object.

    The background value is calculated as an edge average: see
    edge_average().  Black-bordered images therefore have a black
    background, and white-bordered images have a white
    background. Images with no border have a background that is less
    of a contrast than a white or black one.

    At present, rotation, size_normalization, etc. just resample; it
    would be nice to support some interpolation options as well.
    """

    __abstract = True

    aspect_ratio  = param.Number(default=1.0,bounds=(0.0,None),
        softbounds=(0.0,2.0),precedence=0.31,doc="""
        Ratio of width to height; size*aspect_ratio gives the width.""")

    size  = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.30,doc="""
        Height of the image.""")

    pattern_sampler = param.ClassSelector(class_=ImageSampler,
        default=PatternSampler(background_value_fn=edge_average,
                               size_normalization='fit_shortest',
                               whole_pattern_output_fns=[DivisiveNormalizeLinf()]),doc="""
        The PatternSampler to use to resample/resize the image.""")

    cache_image = param.Boolean(default=True,doc="""
        If False, discards the image and pattern_sampler after drawing the pattern each time,
        to make it possible to use very large databases of images without
        running out of memory.""")

    channel_transform = param.Callable( doc="""
       An optional ChannelTransform to apply any post processing to the image channels (eg, channel-specific
       scaling).""")

    def __init__(self, **params):
        self._channel_data = []
        self._image = None
        super(GenericImage, self).__init__(**params)


    @property
    def channels(self):
        if self._image is None:
            self()
        return self._channel_data


    def _get_image(self,p):
        """
        If necessary as indicated by the parameters, get a new image,
        assign it to self._image and return True.  If no new image is
        needed, return False.
        """
        raise NotImplementedError


    def _reduced_call(self, **params_to_override):
        """
        Simplified version of PatternGenerator's __call__ method.
        """
        p=param.ParamOverrides(self,params_to_override)

        fn_result = self.function(p)
        self._apply_mask(p,fn_result)
        result = p.scale*fn_result+p.offset
        return result


    def _process_channels(self,p,**params_to_override):
        """
        Method to add the channel information to the channel_data
        attribute.
        """
        orig_image = self._image

        for i in range(len(self._channel_data)):
            self._image = self._channel_data[i]
            self._channel_data[i] = self._reduced_call(**params_to_override)
        self._image = orig_image
        return self._channel_data


    def function(self,p):
        height   = p.size
        width    = p.aspect_ratio*height

        result = p.pattern_sampler(self._get_image(p),p.pattern_x,p.pattern_y,
                                   float(p.xdensity),float(p.ydensity),
                                   float(width),float(height))
        if p.cache_image is False:
            self._image = None
            del self.pattern_sampler.image

        return result


    def __getstate__(self):
        """
        Return the object's state (as in the superclass), but replace
        the '_image' attribute's Image with a string representation.
        """
        state = super(GenericImage,self).__getstate__()

        if '_image' in state and state['_image'] is not None:
            f = BytesIO()
            image = state['_image']
            # format could be None (we should probably just not save in that case)
            image.save(f,format=image.format or 'TIFF')
            state['_image'] = f.getvalue()
            f.close()

        return state

    def __setstate__(self,state):
        """
        Load the object's state (as in the superclass), but replace
        the '_image' string with an actual Image object.
        """
        # state['_image'] is apparently sometimes None (see SF #2276819).
        if '_image' in state and state['_image'] is not None:
            state['_image'] = Image.open(BytesIO(state['_image']))
        super(GenericImage,self).__setstate__(state)



class FileImage(GenericImage):
    """
    2D Image generator that reads the image from a file.

    The image at the supplied filename is converted to grayscale if it
    is not already a grayscale image. See Image's Image class for
    details of supported image file formats.
    """

    filename = param.Filename(default='images/ellen_arthur.pgm', precedence=0.9,doc="""
        File path (can be relative to Param's base path) to a bitmap
        image.  The image can be in any format accepted by PIL,
        e.g. PNG, JPG, TIFF, or PGM as well or numpy save files (.npy
        or .npz) containing 2D or 3D arrays (where the third dimension
        is used for each channel).
        """)


    def __init__(self, **params):
        super(FileImage,self).__init__(**params)
        self.last_filename = None  # Cached to avoid unnecessary reloading for each channel


    def __call__(self,**params_to_override):
        p = param.ParamOverrides(self,params_to_override)
        # Cache image to prevent channel_data to be deleted before channel specific processing has been finished.
        params_to_override['cache_image']=True
        gray = super(FileImage,self).__call__(**params_to_override)

        self._channel_data = self._process_channels(p,**params_to_override)

        if self.channel_transform:
            self._channel_data = self.channel_transform(p, self._channel_data)

        if p.cache_image is False:
            self._image = None

        return gray


    def _get_image(self,p):
        file_, ext = splitext(p.filename)
        npy = (ext.lower() == ".npy")
        reload_image = (p.filename!=self.last_filename or self._image is None)

        self.last_filename = p.filename

        if reload_image:
            if npy:
                self._load_npy(p.filename)
            else:
                self._load_pil_image(p.filename)


        return self._image


    def _load_pil_image(self, filename):
        """
        Load image using PIL.
        """
        self._channel_data = []
        im = Image.open(filename)
        self._image = ImageOps.grayscale(im)
        im.load()

        file_data = np.asarray(im, float)
        file_data = file_data / file_data.max()
        num_channels = file_data.shape[2]
        for i in range(num_channels):
            self._channel_data.append( file_data[:, :, i])


    def _load_npy(self, filename):
        """
        Load image using Numpy.
        """
        self._channel_data = []
        file_channel_data = np.load(filename)
        file_channel_data = file_channel_data / file_channel_data.max()

        for i in range(file_channel_data.shape[2]):
            self._channel_data.append(file_channel_data[:, :, i])

        self._image = file_channel_data.sum(2) / file_channel_data.shape[2]




class RGBChannelTransform(ChannelTransform):
    """
    PostProcessor for the specific case of 3-channel (RED/GREEN/BLUE)
    color images.  Color-specific processing is applied, in particular
    to rotate the hue of each image at random, thus achieving a
    balanced color input across many pattern presentations.
    """

    saturation = param.Number(default=1.0)

    apply_hue_jitter = param.Boolean(default=True, doc="""
        Whether to apply a random uniform jitter to the image,
        eg, to perform random hue rotation.""")

    random_hue_jitter = param.ClassSelector(numbergen.RandomDistribution,
        default=numbergen.UniformRandom(name='hue_jitter',lbound=0,ubound=1,seed=1048921), doc="""
        Numbergen random generator to be used to create a distribution in hue jitter,
        ie, to perform hue rotation on the images.""")

    _hack_recording = param.Parameter(default=None)


    def __call__(self,p, channel_data):
        if(self.apply_hue_jitter):
            im2pg = color_conversion.image2receptors
            pg2analysis = color_conversion.receptors2analysis
            analysis2pg = color_conversion.analysis2receptors
            jitterfn = color_conversion.jitter_hue
            satfn = color_conversion.multiply_sat

            channs_in  = np.dstack(channel_data)
            channs_out = im2pg(channs_in)
            analysis_space = pg2analysis(channs_out)

            if self._hack_recording is not None:
                self._hack_recording(self,channs=channs_in,extra=analysis_space)

            jitterfn(analysis_space,self.random_hue_jitter())
            satfn(analysis_space,self.saturation)

            channs_out = analysis2pg(analysis_space)

            channel_data = np.dsplit(channs_out, 3) # must be RGB!
            for a in channel_data:
                a.shape = a.shape[0:2]


        return channel_data



class RGBImage(FileImage):
    """
    For backwards compatibility.
    """
    channel_transform = param.Callable(default=RGBChannelTransform)


class NumpyFile(FileImage):
    """
    For backwards compatibility.
    """

    pattern_sampler = param.ClassSelector(class_=ImageSampler,
        default=PatternSampler(background_value_fn=edge_average,
                               size_normalization='original',
                               whole_pattern_output_fns=[]),doc="""
        The PatternSampler to use to resample/resize the image.""")




class CompositeImage(GenericImage):
    """
    Wrapper for any PatternGenerator to support multiple
    channels.

    If the specified generator itself has a 'generator' attribute,
    ExtendToNChannel will attempt to get the channels' data from
    generator.generator (e.g. ColorImage inside a Selector);
    otherwise, ExtendToNChannel will attempt to get channel_data
    from generator. If no channel_data is found in this
    ways, ExtendToNChannel will synthesize the channels from the traditional
    single channel generator.

    After finding or synthesizing the channels, they are
    scaled according to relative_channel_strengths.
    """

    generator = param.ClassSelector(class_=PatternGenerator,default=Constant(),doc="""
        PatternGenerator to be converted to N-Channels.""")

    channel_factors = param.Dynamic(default=[1.,1.,1],doc="""
        Channel scaling factors. The length of this list sets the number of channels to be
        created, unless the input_generator is alreay NChannel (in which case the number of
        its channels is used).""")

    correlate_channels = param.Parameter(default=None, doc="""
        List which contains 3 values:
          correlate_channels = ( dest_channel_to_correlate, channel_to_correlate_to, correlation_strength )""")


    def __init__(self,**params):
        super(CompositeImage,self).__init__(**params)
        self._channel_data = []

        for i in range(len(self.channel_factors)):
            self._channel_data.append( None )


    def post_process(self):  # old hack_hook1
        pass

    def pre_process_generator(self,generator):  # old hack_hook0
        pass

    def set_channel_values(self,p,params,gray,generator):
        """
        Given an input generator, ExtendToNChannel's channels are synthesized to match the Class scope:
        -if monochrome generators are used, each channel is a copy of the generated input, scaled by channel_factors;
        -if NChannel generators are used, their channels are copied and scaled by channel_factors.
        """
        # if the generator has the channels, take those values -
        # otherwise use gray*channel factors

        if(hasattr(generator, '_channel_data')):
            for i in range(len(self.channel_factors)):
                self._channel_data[i] = generator._channel_data[i]*self.channel_factors[i]
        else:
            for i in range(len(self.channel_factors)):
                self._channel_data[i] = gray*self.channel_factors[i]

    def __call__(self,**params):
        """
        Generate NChannel patterns, eventually synthesizing them.
        """
        p = param.ParamOverrides(self,params)

        params['xdensity']=p.xdensity
        params['ydensity']=p.ydensity
        params['bounds']=p.bounds

        # (not **p)
        gray = p.generator(**params)

        #### GET GENERATOR ####
        # Got to get the generator that's actually making the pattern
        if hasattr(p.generator,'get_current_generator'):
            # access the generator without causing any index to be advanced
            generator = p.generator.get_current_generator()
        elif hasattr(p.generator,'generator'):
            generator = p.generator.generator
        else:
            generator = p.generator
        #######################


        self.pre_process_generator(generator)

        self.set_channel_values(p,params,gray,generator)


        if self.correlate_channels is not None:
            corr_to,corr_from,corr_amt = self.correlate_channels
            self._channel_data[corr_to] = corr_amt*self._channel_data[corr_from]+(1-corr_amt)*self._channel_data[corr_to]


        self.post_process()
        
        return gray




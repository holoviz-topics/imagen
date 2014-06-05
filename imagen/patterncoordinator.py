"""
The ImaGen patterncoordinator module provides the class PatternCoordinator, which contains a set of coordinated pattern generators.
The values of the coordinated pattern generators are modified with subclasses of ValueCoordinator.
"""

import math
import json

import param
from param.parameterized import ParamOverrides

from patterngenerator import PatternGenerator
from image import FileImage
from imagen import Gaussian, Composite, Selector

import numbergen

class ValueCoordinator(param.ParameterizedFunction):
    """ 
    A ValueCoordinator modifies a PatternGenerator depending on its label. It supports having multiple pattern generators,
    by indexing the generators with i. To get new pattern generators, one can change the input seed.
    Subclasses of this class can use parameters given in params_to_override.
    This superclass ensures a common interface across all ValueCoordinator subclasses, which is necessary because they are usually
    stored in a list, and each item is called the same way."""
    
    def __call__(self, pattern, label, i, input_seed, **params_to_override):
        """
        'pattern' is the PatternGenerator to be modified
        'label' is the name of the PatternGenerator, and this might be used to handle PatternGenerators differently depending on their name
        'i' is an integer referring the i-th PatternGenerator in case the final pattern is composed of several generators
        'input_seed' is a seed which can be changed to get completely new pattern generator properties
        'params_to_override' is a parameter list of which a subset can be used in subclasses of this class
        """
        raise NotImplementedError

        
        
class VC_x(ValueCoordinator):
    """
    Coordinates the x coordinate of patterns.
    """
    
    position_bound_x = param.Number(default=0.8,doc="""
        Left/rightmost position of the pattern center on the x axis.""")

    def __call__(self, pattern, label, i, input_seed, **params_to_override):
        p = ParamOverrides(self,params_to_override,allow_extra_keywords=True)
                
        pattern.x += numbergen.UniformRandom(lbound=-p.position_bound_x,ubound=p.position_bound_x,seed=input_seed+12+i)
        
        
        
class VC_y(ValueCoordinator):
    """
    Coordinates the y coordinate of patterns.
    """
    
    position_bound_y = param.Number(default=0.8,doc="""
        Upper/lowermost position of the pattern center on the y axis.""")

    def __call__(self, pattern, label, i, input_seed, **params_to_override):
        p = ParamOverrides(self,params_to_override,allow_extra_keywords=True)
                
        pattern.y += numbergen.UniformRandom(lbound=-p.position_bound_y,ubound=p.position_bound_y,seed=input_seed+35+i)
       
        
        
class VC_orientation(ValueCoordinator):
    """
    Coordinates the orientation of patterns.
    """
    
    orientation_bound = param.Number(default=math.pi,doc="""
        Rotate pattern around the origin by at most orientation_bound radians (in both directions).""")
    
    def __call__(self, pattern, label, i, input_seed, **params_to_override):
        p = ParamOverrides(self,params_to_override,allow_extra_keywords=True)

        pattern.orientation = numbergen.UniformRandom(lbound=-p.orientation_bound,ubound=p.orientation_bound,seed=input_seed+21+i)
        
        
        
class PatternCoordinator(param.Parameterized):
    """
    A set of coordinated pattern generators.
    The pattern generators are named according to the labels parameter.
    The dimensions to be returned are specified with the dims parameter.
    PatternGenerators are first instantiated with the parameters specified in input_parameters, 
    and then subclasses of ValueCoordinator are applied to modify the properties of these generators.
    """
    
    input_type = param.ClassSelector(PatternGenerator,default=Gaussian,is_instance=False,doc="""
        PatternGenerator to be used. Usually is one of those defined in imagen/__init__
        Parameters given to the PatternGenerator must be adapted respectively in input_parameters.""")
    
    input_parameters = param.Dict(default={'size': 0.088388, 'aspect_ratio': 4.66667},doc="""
        These parameters are passed to the PatternGenerator specified in input_type.""")
    
    total_num_inputs = param.Integer(default=2,doc="""
        Number of patterns used to generate one input.""")
    
    dims = param.List(default=['xy','or'],class_=str,doc="""
        Stimulus dimensions to include, such as:
          :'xy': Position in x and y coordinates
          :'or': Orientation
          
        Subclasses and callers may extend this list to include any other dimensions
        for which a metafeature_mapping has been defined.""")
    
    labels = param.List(default=['Input1','Input2'],class_=str,bounds=(1,None),doc="""     
        For each string in this list, a PatternGenerator of the requested type will be returned,
        with parameters whose values may depend on the string supplied. For instance, if the 
        list ["Input1","Input2"] is supplied, a metafeature function might inspect those
        labels and set parameters differently for Input1 and Input2, returning two different
        PatternGenerators with those labels.""")
                    
    input_seed = param.Integer(default=0,doc="""
        Base seed for the input patterns. It can be changed to get completely new random input pattern parameters.""")
    
    composite_type = param.ClassSelector(PatternGenerator,default=Composite,is_instance=False,doc="""
        Class which combines the total_num_inputs individual patterns and creates a single pattern that it returns.
        Use Selector if there is only one individual pattern.""")
    
    composite_parameters = param.Dict(default={},doc="""
        These parameters are passed to the composite specified in composite_type.""")
    
    metafeature_mapping = param.Dict(default={
        'xy': [VC_x, VC_y],
        'or': VC_orientation},doc="""
        Mapping from the dimension name (key) to the method(s) which are applied to the pattern generator.
        The value can either be a single method or a list of methods. The order in the mapping corresponds
        to the order the functions are called.""")
    
    def create_inputs(self, properties=None):
        return [self.input_type(**self.input_parameters) for i in xrange(self.total_num_inputs)]    
    
    
    def __init__(self,inherent_dims={},**params):
        """
        inherent_dims is a dictionary, whereas the key is the name of the inherent dimension and the value specifies
        how to access this inherent dimension.
        
        params can have extra keywords which are storen in metafeature_params and passed down to the metafeature_fns
        """
        p=ParamOverrides(self,params,allow_extra_keywords=True)
     
        super(PatternCoordinator, self).__init__(**p.param_keywords())
        
        self.metafeature_params = p.extra_keywords()
        
        self.inherent_dims = inherent_dims
        
        # This checks whether there are keys in inherent_dims which are not in dims
        # And also, this key must be in metafeature_mapping because inherent_dims
        # can have additional dimensions such as i to support multiple images
        if(len((set(self.inherent_dims.keys()) - set(self.dims)) & set(self.metafeature_mapping.keys()))):
            self.warning('Inherent dimension present which is not requested in dims!')
        
        self.metafeature_fns = []
        for dimension, metafeature_function in self.metafeature_mapping.iteritems():
            if dimension in self.dims and dimension not in self.inherent_dims:
                # if it is a list, append each list item individually
                if isinstance(metafeature_function,list):
                    for metafeature_individual_function in metafeature_function:
                        self.metafeature_fns.append(metafeature_individual_function)
                else:
                    self.metafeature_fns.append(metafeature_function)
                    
    def __call__(self):
        coordinated_pattern_generators={}
        for label in self.labels:
            inputs=self.create_inputs({'label': label})
            
            # Apply metafeature_fns
            for i in xrange(len(inputs)):
                for fn in self.metafeature_fns:
                    fn(inputs[i],label,i,self.input_seed,**self.metafeature_params)
                    
            combined_inputs=self.composite_type(generators=inputs,**self.composite_parameters)
            coordinated_pattern_generators.update({label:combined_inputs})
        return coordinated_pattern_generators
    
    
    
class PatternCoordinatorImages(PatternCoordinator):
    input_type = param.ClassSelector(PatternGenerator,default=FileImage,is_instance=False) 
    
    input_parameters = param.Dict(default={'size': 10})
        
    composite_type = param.ClassSelector(PatternGenerator,default=Selector,is_instance=False)
    
    def __init__(self,dataset_name,**params):
        """
        dataset_name is the path to a json file containing a description for a dataset
        The json file must contain the following entries:
            :'name': Name of the dataset (string)
            :'length': Number of images in the dataset (integer)
            :'description': Description of the dataset (string)
            :'source': Citation of paper for which the dataset was created / website where the dataset can be found (string)
            :'filename_template': Path to the images with placeholders ({placeholder_name}) 
            for inherent dimensions and the image number, e.g. "image_filename_template": "images/shouval/combined#i#.png" 
            :'inherent_dims': Dictionary specifying how to access inherent dimensions, value is used in eval().
            Currently the label of the pattern generator ('label') as well as the image number ('current_image') are given
            as parameters, which can be used as follows:
            "inherent_dims": "{'i': lambda params: '%02d' % (params['current_image']+1)}"
            or
            "inherent_dims": "{'i': lambda params: '%02d' % (params['current_image']+1),
                                'dy':lambda params: '01' if params['label']=='LeftRetina' else '02'}"
            In the future, 'cone' as well as 'lag' will be available as well.
        """
        
        dataset=json.loads(open(param.resolve_path(dataset_name)).read())
        self.dataset_name=dataset['name']
        self.total_num_inputs=dataset['length']
        self.description=dataset['description']
        self.filename_template=dataset['filename_template']
        self.source=dataset['source']
        inherent_dims=eval(dataset['inherent_dims'])
        
        super(PatternCoordinatorImages, self).__init__(inherent_dims,**params)
    
    
    def generate_filenames(self, params):
        filenames = [self.filename_template]*self.total_num_inputs
        for dimension in self.inherent_dims:
            filenames = [filename.replace('{'+dimension+'}', self.inherent_dims[dimension](params))
                                for filename,params['current_image'] in zip(filenames,range(self.total_num_inputs))]
        return filenames
    
    
    def create_inputs(self, properties):
        return [self.input_type(
                    filename=f, 
                    cache_image=False,
                    **self.input_parameters)
                for f,i in zip(self.generate_filenames(properties),range(self.total_num_inputs))]

"""
The ImaGen patterncoordinator module provides the class PatternCoordinator, which contains a set of coordinated pattern generators.
The values of the coordinated pattern generators are modified with subclasses of FeatureCoordinator.
"""

from os import listdir
from os.path import isfile, join
import math
import json

import param
from param.parameterized import ParamOverrides

from imagen.patterngenerator import PatternGenerator
from imagen.image import FileImage
from imagen import Gaussian, Composite, Selector, CompositeBase

import numbergen

class FeatureCoordinator(param.ParameterizedFunction):
    """ 
    A FeatureCoordinator modifies a PatternGenerator. The modification can depend on the pattern_label. 
    It supports having multiple pattern generators, by indexing the generators with pattern_number. 
    To get new pattern generators, one can change the master_seed.
    Subclasses of this class can use parameters given in params.
    This superclass ensures a common interface across all FeatureCoordinator subclasses, which is necessary 
    because they are usually stored in a list, and each item is called the same way."""
    
    def __call__(self, pattern, pattern_label, pattern_number, master_seed, **params):
        """
        'pattern' is the PatternGenerator to be modified
        'pattern_label' is the name of the PatternGenerator, and this might be used to handle PatternGenerators differently depending on their name
        'pattern_number' is an integer referring the PatternGenerator with index pattern_number in case the final pattern is composed of several generators
        'master_seed' is a seed which can be changed to get completely new pattern generator properties
        'params' is a parameter list of which a subset can be used in subclasses of this class
        """
        raise NotImplementedError

        
        
class XCoordinator(FeatureCoordinator):
    """
    Chooses a random value for the x coordinate, subject to the provided position_bound_x.
    """
    
    position_bound_x = param.Number(default=0.8,doc="""
        Left/rightmost position of the pattern center on the x axis.""")

    def __call__(self, pattern, pattern_label, pattern_number, master_seed, **params):
        p = ParamOverrides(self,params,allow_extra_keywords=True)
                
        pattern.x += numbergen.UniformRandom(lbound=-p.position_bound_x,ubound=p.position_bound_x,seed=master_seed+12+pattern_number)
        
        
        
class YCoordinator(FeatureCoordinator):
    """
    Chooses a random value for the y coordinate, subject to the provided position_bound_y.
    """
    
    position_bound_y = param.Number(default=0.8,doc="""
        Upper/lowermost position of the pattern center on the y axis.""")

    def __call__(self, pattern, pattern_label, pattern_number, master_seed, **params):
        p = ParamOverrides(self,params,allow_extra_keywords=True)
                
        pattern.y += numbergen.UniformRandom(lbound=-p.position_bound_y,ubound=p.position_bound_y,seed=master_seed+35+pattern_number)
       
        
        
class OrientationCoordinator(FeatureCoordinator):
    """
    Chooses a random orientation within the specified orientation_bound in each direction.
    """
    
    orientation_bound = param.Number(default=math.pi,doc="""
        Rotate pattern around the origin by at most orientation_bound radians (in both directions).""")
    
    def __call__(self, pattern, pattern_label, pattern_number, master_seed, **params):
        p = ParamOverrides(self,params,allow_extra_keywords=True)

        pattern.orientation = numbergen.UniformRandom(lbound=-p.orientation_bound,ubound=p.orientation_bound,seed=master_seed+21+pattern_number)
        
        
        
class PatternCoordinator(param.Parameterized):
    """
    Returns a set of coordinated pattern generators. The pattern generators are named according to pattern_labels.
    
    The features to be returned are specified with the features_to_vary parameter. A feature is something coordinated between 
    the PatternGenerators, either 
        (a) one of the parameters of the PatternGenerators (such as size), 
        (b) something that affects the parameters that is calculated by a metafeature_function (such as disparity), or 
        (c) something that is inherent to the dataset.
    
    PatternGenerators are first instantiated with the parameters specified in pattern_parameters, 
    and then subclasses of FeatureCoordinator are applied to modify the properties of these generators.
    """
    
    pattern_type = param.ClassSelector(PatternGenerator,default=Gaussian,is_instance=False,doc="""
        PatternGenerator to be used. Usually is one of those defined in imagen/__init__
        Parameters passed to the pattern generator can be specified in pattern_parameters.""")
    
    pattern_parameters = param.Dict(default={'size': 0.088388, 'aspect_ratio': 4.66667},doc="""
        These parameters are passed to the PatternGenerator specified in pattern_type.""")
    
    patterns_per_label = param.Integer(default=2,doc="""
        Number of patterns used to generate one pattern.""")
    
    features_to_vary = param.List(default=['xy','or'],class_=str,doc="""
        Stimulus features to vary, such as:
          :'xy': Position in x and y coordinates
          :'or': Orientation
          
        Subclasses and callers may extend this list to include any other features
        for which a coordinator has been defined in feature_coordinators.""")
    
    pattern_labels = param.List(default=['Pattern1','Pattern2'],class_=str,bounds=(1,None),doc="""     
        For each string in this list, a PatternGenerator of the requested type will be returned,
        with parameters whose values may depend on the string supplied. For instance, if the 
        list ["Pattern1","Pattern2"] is supplied, a metafeature function might inspect those
        pattern_labels and set parameters differently for Pattern1 and Pattern2, returning two different
        PatternGenerators with those pattern_labels.""")
                    
    master_seed = param.Integer(default=0,doc="""
        Base seed for the patterns. Each numbered pattern on each of the various pattern_labels will have a different 
        random seed, but all of these seeds include this master seed value, and so changing it will alter the streams of 
        all the random pattern parameters.""")
    
    composite_type = param.ClassSelector(CompositeBase,default=Composite,is_instance=False,doc="""
        Class that combines the patterns_per_label individual patterns and creates a single pattern that it returns. 
        For instance, Composite can merge the individual patterns into a single pattern using a variety of operators 
        like add or maximum, while Selector can choose one out of a given set of patterns.""")
    
    composite_parameters = param.Dict(default={},doc="""
        These parameters are passed to the composite specified in composite_type.""")
    
    feature_coordinators = param.Dict(default={
        'xy': [XCoordinator,YCoordinator],
        'or': OrientationCoordinator},doc="""
        Mapping from the feature name (key) to the method(s) which are applied to the pattern generators.
        The value can either be a single method or a list of methods.""")
    
    
    def _create_patterns(self, properties=None):
        """
        _create_patterns returns a list of PatternGenerator instances. There must be patterns_per_label instances
        in this list. To create the instances, pattern_type should be used (passing pattern_parameters).
        
        properties is a dictionary {'pattern_label': pattern_label} which can be used to create PatternGenerators
        depending on the requested pattern_label
        """
        return [self.pattern_type(**self.pattern_parameters) for i in xrange(self.patterns_per_label)]    
    
    
    def __init__(self,_inherent_features={},**params):
        """
        params can have extra keywords. They are passed down to the methods specified in
        feature_coordinators, in case the corresponding feature is requested in features_to_vary.
        """
        #_inherent_features can be used to declare that for the dataset in use, the specified features are inherently 
        #forced to vary and need not be synthesized using the feature_coordinators.
        p=ParamOverrides(self,params,allow_extra_keywords=True)
     
        super(PatternCoordinator, self).__init__(**p.param_keywords())
        
        self._feature_params = p.extra_keywords()
        
        self._inherent_features = _inherent_features
        
        # This checks whether there are keys in _inherent_features which are not in features
        # And also, this key must be in feature_coordinators because _inherent_features
        # can have additional features such as i to support multiple images
        
        # TFALERT: Once spatial frequency (sf) is added, this will cause warnings, because all image 
        # datasets will have a spatial frequency inherent feature, but mostly we just ignore that by having 
        # only a single size of DoG, which discards all but a narrow range of sf. 
        # So the dataset will have sf inherently, but that won't be an error or even worthy of a warning.
        if(len((set(self._inherent_features.keys()) - set(self.features_to_vary)) & set(self.feature_coordinators.keys()))):
            self.warning('Inherent feature present which is not requested in features!')
        
        self._feature_coordinators_to_apply = []
        for feature, feature_coordinator in self.feature_coordinators.iteritems():
            if feature in self.features_to_vary and feature not in self._inherent_features:
                # if it is a list, append each list item individually
                if isinstance(feature_coordinator,list):
                    for individual_feature_coordinator in feature_coordinator:
                        self._feature_coordinators_to_apply.append(individual_feature_coordinator)
                else:
                    self._feature_coordinators_to_apply.append(feature_coordinator)
                    
    def __call__(self):
        coordinated_pattern_generators={}
        for pattern_label in self.pattern_labels:
            patterns=self._create_patterns({'pattern_label': pattern_label})
            
            # Apply _feature_coordinators_to_apply
            for i in xrange(len(patterns)):
                for fn in self._feature_coordinators_to_apply:
                    fn(patterns[i],pattern_label,i,self.master_seed,**self._feature_params)
                    
            combined_patterns=self.composite_type(generators=patterns,**self.composite_parameters)
            coordinated_pattern_generators.update({pattern_label:combined_patterns})
        return coordinated_pattern_generators
    
    
    
class PatternCoordinatorImages(PatternCoordinator):
    pattern_type = param.ClassSelector(PatternGenerator,default=FileImage,is_instance=False)
    
    pattern_parameters = param.Dict(default={'size': 10})
        
    composite_type = param.ClassSelector(CompositeBase,default=Selector,is_instance=False)
        
    def __init__(self,dataset_name,**params):
        """
        dataset_name is the path to a JSON file (https://docs.python.org/2/library/json.html) containing a description for a dataset
        The JSON file should contain the following entries:
            :'name': Name of the dataset (string, default=basename(dataset_name))
            :'length': Number of images in the dataset (integer, default=number of files in directory of dataset_name minus 1)
            :'description': Description of the dataset (string, default="")
            :'source': Citation of paper for which the dataset was created (string, default=name)
            :'filename_template': Path to the images with placeholders ({placeholder_name})
            for inherent features and the image number, e.g. "filename_template": "images/image{i}.png"
            default={current_image}.jpg
            :'inherent_features': Dictionary specifying how to access inherent features, value is used in eval().
            Currently the label of the pattern generator ('pattern_label') as well as the image number ('current_image') are given
            as parameters, whereas current_image varies from 0 to length-1 and pattern_label is one of the items of
            pattern_labels. Default={'i': lambda params: '%02d' % (params['current_image']+1)}
            Example 1: Imagine having images without any inherent features named as follows: "images/image01.png",
            "images/image02.png" and so on. Then, filename_template: "images/image{i}.png" and
            "inherent_features": "{'i': lambda params: '%02d' % (params['current_image']+1)}"
            This replaces {i} in the template with the current image number + 1
            Example 2: Imagine having image pairs from a stereo webcam named as follows: "images/image01_left.png", 
            "images/image01_right.png" and so on. If pattern_labels=['Left','Right'], then
            filename_template: "images/image{i}_{dy}" and
            "inherent_features": "{'i': lambda params: '%02d' % (params['current_image']+1),
                                   'dy':lambda params: 'left' if params['pattern_label']=='Left' else 'right'}"
            Here, additionally {dy} gets replaced by either 'left' if the pattern_label is 'Left' or 'right' otherwise
        """
        
        filename=param.resolve_path(dataset_name)
        filepath=os.path.dirname(filename)
        dataset=json.loads(open(filename).read())
        self.dataset_name=dataset.get('name', os.path.basename(dataset_name))
        self.patterns_per_label=dataset.get('length', len([ f for f in listdir(filepath) if isfile(join(filepath,f)) ]) - 1)
        self.description=dataset.get('description', "")
        self.filename_template=dataset.get('filename_template', filepath+"/{i}.jpg")
        self.source=dataset.get('source', self.dataset_name)
        inherent_features=eval(dataset['inherent_features']) if 'inherent_features' in dataset else {'i': lambda params: '%02d' % (params['current_image']+1)}
        
        super(PatternCoordinatorImages, self).__init__(inherent_features,**params)
    
    
    def _generate_filenames(self, params):
        filenames = [self.filename_template]*self.patterns_per_label
        for feature in self._inherent_features:
            filenames = [filename.replace('{'+feature+'}', self._inherent_features[feature](params))
                                for filename,params['current_image'] in zip(filenames,range(self.patterns_per_label))]
        return filenames
    
    
    def _create_patterns(self, properties):
        return [self.pattern_type(
                    filename=f, 
                    cache_image=False,
                    **self.pattern_parameters)
                for f,i in zip(self._generate_filenames(properties),range(self.patterns_per_label))]

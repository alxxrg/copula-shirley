"""
* MIT License
* Copyright <2018> <dataresponsibly.com> 
* https://github.com/DataResponsibly/DataSynthesizer
"""

import warnings
import sys, os
sys.path.append(os.getcwd() + '/PrivBayes/')

from PrivBayes.DataDescriber import DataDescriber
from PrivBayes.DataGenerator import DataGenerator
from PrivBayes.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
    
def PrivBayes(dataset, num_to_generate, dp_eps, degree_max, verbose=0, seed=0):    
    # An attribute is categorical if its domain size is less than this threshold.
    # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
    threshold_value = 20

    # specify categorical attributes
    # can be left empty
    categorical_attributes = {}

    # specify which attributes are candidate keys of input dataset.
    candidate_keys = {}

    # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
    # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
    epsilon = dp_eps

    # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
    # 0 indicates that the parameter will be selected automatically
    degree_of_bayesian_network = int(degree_max)
    
    num_tuples_to_generate = int(num_to_generate)

    WD = os.getcwd()
    
    # input dataset
    input_data = f'{WD}/temp/temp_{dataset}.csv'
    
    # location of two output files
    mode = 'correlated_attribute_mode'
        
    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys,
                                                        verbose=verbose,
                                                        seed=seed)
    description_dic = describer.data_description
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_dic, verbose=verbose, seed=seed)
    return generator.synthetic_dataset
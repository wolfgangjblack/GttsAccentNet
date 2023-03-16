##This script is used to generate a .json, which acts as a parameter guide to inference.p

import json
from utils.utils import (write_json)

config = {}
##Specify where the data for inference - if this were to ever be put into production this could point to a kafka stream or some other backend table
# Users should note, this data has already been transformed and saved off. In the case of live streaming/batch inference the data WOULD need to be put through a pipleine. 
# The function generate_tf_dataset from model_utils.py will produced the desired (albiet shuffled) dataset
config['test_data_dir'] = './model/artifacts/test_data'
##Specify where to save model inference artifacts
config['artifacts_dir'] = './artifacts/'
##Specify where to model is stored, so it can be loaded for inference
config['model_dir'] = './model/artifacts/models/'
##Specify type of model - this can either be shallow_cnn or inception. The specified model will be used for inference
config['model_type'] = 'shallow_cnn'
#config['model_type'] = 'inception'
##Class labels as strings should be produced and verified in EDA. This is just a mapping of the int labels back to their human readable strings.
config['class_labels_as_strs'] = ['us', 'au', 'uk', 'ca', 'in']

write_json('config', 'inf_config.json', config)
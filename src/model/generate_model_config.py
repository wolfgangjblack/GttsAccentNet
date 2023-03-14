##This script is used to generate a .json, which acts as a guide to model.py

import json

config = {}
##Specify where the data is for model training
config['data_dir'] = '../../data'
##Specify where to save model training artifacts
config['artifacts_dir'] = './artifacts'
##Specify where to save model after training
config['model_dir'] = './artifacts/models/'
##Specify where to save test data (a ts.data.Dataset object) - note: this is during model training and used in src/main.py to simulate inference
config['test_data_dir'] = './artifacts/test_data/'
##Specify number of frames to limit data - this should be found from eda. In this example, its 19520
config['frames'] = 19520
##Specify type of model - this can either be shallow_cnn or inception. if shallow_cnn, we'll generate a shallow cnn. 
config['model_type'] = 'shallow_cnn'
# If inception, we'll call down a pre-trained inceptionV3 model with imagenet weights and repurpose the input and output for our model
#config['model_type'] = 'inception'
##Specify spectrogram type. Can be either spec, or mels. WARNING ANY STRING OTHER THAN SPEC WILL RESULT IN MELS
config['spec_type'] = 'spec'
##Specify a batch size for training, 16 is a decent size. 
config['batch_size'] = 16
##specify base learning rate for adam
config['base_learning_rate'] = 0.001
##
config['epochs'] = 25

json_data = json.dumps(config)
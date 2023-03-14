import os
import json

import tensorflow as tf

from utils.model_utils import (check_artifacts_dir,
                             generate_tf_dataset,
                             get_input_shape,
                             build_and_compile_model,
                             save_model_performance
                             )

##load from config
data_dir = config['data_dir'] 
artifacts_dir = config['artifacts_dir'] 
model_dir = config['model_dir'] 
test_data_dir = config['test_data_dir'] 
frames = config['frames']
model_type = config['model_type']
batch_size = config['batch_size']
spec_type = config['spec_type']
base_learning_rate = config['base_learning_rate']
epochs = config['epochs']

##Check if artifact/model dir exist, and if they don't - generate them
check_artifacts_dir(artifacts_dir)
check_artifacts_dir(model_dir)
check_artifacts_dir(test_data_dir)


class_labels = [i for i in os.listdir(data_dir) if '_' not in i]

data = generate_tf_dataset(data_dir, class_labels, frames, batch_size, spec_type)

##This splits the dataset into test/train
split = int(len(data)*0.7)+1

train = data.take(split)
test_and_val = data.skip(split).take(len(data)-split)

input_shape = get_input_shape(train)

#Seperate for validation data for training
val_ds = test_and_val.shard(num_shards=2, index=1)

##save off the test data to simulate inference
test = test_and_val.shard(num_shards=2, index=0)
test.save(test_data_dir)

model = build_and_compile_model(input_shape, class_labels, model_type, base_learning_rate)
model.summary()

history = model.fit(train,
                 epochs = epochs,
                 validation_data = val_ds,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=4, restore_best_weights = True))

save_model_performance(history, model_type)


model.save(model_dir + model_type + '_model.h5')

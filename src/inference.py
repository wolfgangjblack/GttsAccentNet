import json
import tensorflow as tf
import numpy as np

from utils.utils import(check_artifacts_dir,
                        get_model,
                        get_preds_array,
                        save_confusion_matrix,
                        save_classification_report)

##load config, if this DNE - User must first run generate_inference_config.py
#this was done intentionally so user can meaningfully set config parameters
f = open('config/inf_config.json')
config = json.load(f)

test_data_dir = config['test_data_dir'] 
artifacts_dir = config['artifacts_dir']
model_dir = config['model_dir']
model_type = config['model_type']
class_labels_as_strs = config['class_labels_as_strs']
#Get labeled test data that was saved off during training - again, if replaced with OTHER data, users should process through pipeline
test_data = tf.data.Dataset.load(test_data_dir)

y_true = np.concatenate([y for x,y in test_data], axis =0)

model = get_model(model_dir, model_type)

raw_preds = model.predict(test_data)

preds = get_preds_array(raw_preds)

save_confusion_matrix(y_true, preds, class_labels_as_strs, model_type, artifacts_dir)
save_classification_report(y_true, preds, model_type, artifacts_dir)
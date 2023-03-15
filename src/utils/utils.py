import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)


def check_artifacts_dir(save_path = './artifacts/'):
    '''
    This function checks if there is an artifacts dir in our root dir, 
    if not it'll create the aritifacts dir to prevent an error
    '''
    try:
         os.listdir(save_path)
    except FileNotFoundError:
        os.mkdir(save_path)
    return 

def get_model(model_dir:str, model_type:str):
    """
    This function searches the model_dir for either a shallow or tranfer learned inception model to be used for inference.
    It looks within the model dir, and selects the model type as specified
    """
    if model_type == 'shallow_cnn':
        model = tf.keras.models.load_model(model_dir+model_type+'_model.h5')
    elif model_type == 'inception':
        model = tf.keras.models.load_model(model_dir+'tranfer_inception_model.h5')
    
    return model

def get_preds_array(raw_preds:np.ndarray):
    preds = [np.argmax(i) for i in raw_preds]
    return preds

def save_confusion_matrix(y_true: np.ndarray, y_preds: np.ndarray, class_labels_as_strs: list, model_type: str, artifacts_dir: str):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_preds)
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtx, 
                xticklabels = class_labels_as_strs,
                yticklabels = class_labels_as_strs,
                annot = True, fmt = 'g')
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    fig.savefig(artifacts_dir+model_type+'_confusion_matrix.jpg')
    return

def save_classification_report(y_true: np.ndarray, y_preds: np.ndarray, model_type: str, artifacts_dir: str):
    """Saves the sklearn.metrics.classification report as a .txt for future reference"""
    textfile = open(artifacts_dir+model_type+'_classification_report.txt', 'w')
    textfile.write(classification_report(y_true, y_preds) )
    textfile.close()
    return
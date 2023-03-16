# Synthetic Data Generation and Classification with Google Text-to-Speech

Date Modified: 3/6/23 <br>
Author: Wolfgang Black <br>

This repository contains code for generating synthetic audio data using Google Text-to-Speech (gTTS) and classifying the data using transfer learning and a shallow custom convolutional neural network (CNN). The data consists of 3000 English words, each spoken with one of five different accents.

## Dependencies
  - Python 3.9
  - Tensorflow 2.4
  - Tensorflow_io
  - NumPy
  - Matplotlib
  - gTTS
  - Seaborn
  - Sklearn
  - Pandas
  
## Usage
To generate the synthetic data, run datagen.py found in ./datagen. This script takes a list of English words as input and outputs a folder containing .wav files of the words spoken with the five different accents. Each accent is saved in its own subdirectory in the data folder. 

An Exploratory Data script is provided in /src/eda/ as eda.py. This script generates wav form, audio file length, audio sampling rates, and spectrogram plots. These should be used to explore who the user wants to process the data. For the example here, we decide to use a standard spectrogram with **19520** frames - though users should feel free to explore these choices in the model. The figure below is an example of the types of spectrogram subplots generated in eda.py:

![alt text](https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/eda/artifacts/heritage_spec_subplot.png)

To generate the model, use /src/model/generate_model_config.py to change the parameters used in /src/model/model.py. This model.py script generates either a shallow CNN or pulls down the inceptionV3 trained on imagenet. The transfer learning model only retrains the output layer. This script uses the spectrogram/mel spectrogram transformation on the audio .wav files to generate images for the CNN. 

Finally inference is done in /src/inference.py relying on /src/generate_inference_config.py to set the parameters such as where to save inference metrics, which model to use, and what dataset to do inference on. The native dataset for inference is generated during model development (in /src/model/model.py) and has already been transformed. A new dataset should be transformed by the **generate_tf_dataset** function found in /src/model/utils/model_utils.py script.  

## Results
During training, both models achieved an accuracy of around 80% on the validation data. However, when evaluated on a hold out dataset, the transfer learned CNN outperformed the shallow custom CNN as can be seen on the training metrics show below.

![alt text](https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/model/artifacts/shallow_cnn_training_metrics.png)

Above we can see that the shallow CNN ends its training with an accuracy and validation accuracy of around 0.8, however we can see in the inference confusion matrix that the shallow CNN has trouble predicting on the canadian english accent. 

![alt text](https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/artifacts/shallow_cnn_confusion_matrix.jpg)

The transfer learned model performed similary during training in that the training accuracy reached 0.8, however its validation accuracy was much worse. 
![alt text](https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/model/artifacts/transfer_inception_training_metrics.png)

Despite this, the transfer learning model actually generalizes MUCH better and can score decently on each class
![alt text](https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/artifacts/inception_confusion_matrix.jpg)

Even though the model does perform better more generally, both models struggle to predict on the sythetic canadian accent. **To address this, I recommend seeding in real accents instead of synthetic accents**. 

## Conclusion
In this project, we generated synthetic audio data using gTTS and used it to train and evaluate two CNNs for word classification. Our results suggest that transfer learning can improve the performance of the CNN for this task. This code can be used as a starting point for other projects that require synthetic data generation and classification - as well as the start of a transcripting software which works to recognize individual accents to better tackle fair and accurate transcription. 

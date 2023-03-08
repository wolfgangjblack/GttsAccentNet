
import os
from utils.eda_utils import (check_artifacts_dir, get_frames_from_quantile, plot_ex_wavs, plot_spectrogram_subplots, 
get_lengths_subplots, save_native_sample_rates)

data_dir = '../../data/'

check_artifacts_dir()

class_labels = [i for i in os.listdir(data_dir) if '_' not in i]

##Get the frames we'll use for data preprocessing - to do this, we'll get the 
##max 80% quartile from the lengths of all the audio files across the class
print('\n\ngetting frames - this may take a moment\n\n')
frames = get_frames_from_quantile(class_labels, 0.8, 16000, data_dir)

##Plot a few words waveforms and spectrograms
print('\n\nsaving word plots\n\n')
words = ['bird', 'hello', 'hi', 'heritage']
for word in words:
    plot_ex_wavs(word, class_labels, data_dir)
    plot_spectrogram_subplots(word, class_labels, data_dir, frames)
    plot_spectrogram_subplots(word, class_labels, data_dir, frames, 'mels_spec')

##check lengths in subplots for each class
print('\n\n getting length subplots \n\n')
get_lengths_subplots(class_labels, data_dir)

##here we've already assumed the native output rate is 16000, if the model
# doesn't train well, we can always inspect this output rate by inspecting the 
# audio files native output rate. We'll create this artifact for later cases
print('\n\n recording native sample rates \n\n')
save_native_sample_rates(class_labels, data_dir)
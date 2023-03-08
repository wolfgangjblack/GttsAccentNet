import os

import tensorflow_io as tfio
import tensorflow as tf

from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def load_wav_output_mono_channel_file(filename, sample_rate_out = 16000):
    """This function takes a filename, which is the full
     path of a specific .wav file, then decodes that file 
     to find the tensor associated with the sound - this is
     later used to get the spectrograms
    """
    #load encoded wav file
    file_contents = tf.io.read_file(filename)
    
    #Decode wav (tensors by channel)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)

    #Remove trailing axis
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)

    #Goes from 44100Hz to 16000Hz - amplitude of the audio signal 
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = sample_rate_out)

    return wav

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

def plot_ex_wavs(word:str, labels: list, data_dir:str, save_path = './artifacts/'):
    """
    This is meant to be a program which saves a comparison 
    graph of wavs for the same word across various accents (classes) plotted with a legend as a .png
    """

    check_artifacts_dir(save_path)

    wav_dict ={}
    fig = plt.figure(figsize = (10,10))

    for label in labels:
        wav_dict[label] = load_wav_output_mono_channel_file(data_dir+label+'/'+word+'.wav')
        plt.plot(wav_dict[label], alpha = 0.25)

    plt.legend(labels)
    plt.title('wav form for '+word)
    
    fig.savefig(save_path+word+'.png' )
    plt.close()
    return

def get_descriptive_class_metrics(data_dir, label):
    """A poorly named function meant to product the lengths of all wavs within a labeled directory, 
    also outputs the mean, min, and max length associated with the class"""
    lengths = []
    for file in os.listdir(data_dir+label+'/'):
        tensor_wave = load_wav_output_mono_channel_file(data_dir+label+'/'+file)
        lengths.append(len(tensor_wave))
    
    mean = tf.math.reduce_mean(lengths).numpy()
    mini = tf.math.reduce_min(lengths).numpy()
    maxi = tf.math.reduce_max(lengths).numpy()

    return lengths, mean, mini, maxi

def get_lengths_subplots(labels, data_dir, savepath = './artifacts/'):
    """
    This produces len(labels) +1 suplots, in a n x 2 array which displays the lengths of the wavs within 
    each label, and a final plot for the box plots of each label. This will help use visually understand the
    length of the wavs, and whether our sampled output rate and designated frames are appropriate. 
    """
    
    nrows = round(len(labels) / 2)+1
    ncols = 2

    lengths = {}
    means = []
    mins = []
    maxs = []
    fig = plt.figure(figsize = (10,10))
    plt.tight_layout()
    for i in range(1, nrows*ncols+1):
        if i == nrows*ncols:
            ax = plt.subplot(nrows, ncols, i)
            ax.set_title('All classes Box Plot')
            sns.boxplot(pd.DataFrame(lengths))
        else:
            label = labels[i-1]
            length, mean, mini, maxi = get_descriptive_class_metrics(data_dir, label)
            ax = plt.subplot(nrows, ncols, i)
            ax.set_title(label +' wav lengths')
            plt.plot(length, 'k--')
            plt.plot([i for i in range(len(length))], np.ones(len(length)) * mean, 'r-')
            plt.plot([i for i in range(len(length))], np.ones(len(length)) * maxi, 'b')
            plt.plot([i for i in range(len(length))], np.ones(len(length)) * mini, 'g')

            lengths[label] = length
            means.append(mean)
            mins.append(mini)
            maxs.append(maxi)
    fig.savefig(savepath + 'lengths_subplot.png')
    plt.close()
    return 

def get_frames_from_quantile(labels:list, quantile:int, sample_out: int, data_dir:str):
    """
    As we process data for model training/testing/inference we need to determine the max frames to standardize the input date.
    This should be roughly informed by get_lengths_subplots so the users can see the lengths of the class wavs plotted together, 
    as well as the boxplot"""
    lengths = {}
    for label in labels:
        lengths[label], _, _, _ = get_descriptive_class_metrics(data_dir, label)
    return round(np.max(pd.DataFrame(lengths).quantile(.8))/sample_out, 2)*sample_out

def save_native_sample_rates(labels:list, data_dir:str, savepath = './artifacts/', filename = 'wav_native_sample_rates'):
    sample_dict = {}
    for label in labels:    
        sample_dict[label] =[]
        for f in os.listdir(data_dir+label+'/'):
            #load encoded wav file
            file = data_dir+label+'/'+f
            file_contents = tf.io.read_file(file)
            
            #Decode wav (tensors by channel)
            _, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
            sample_dict[label].append(sample_rate.numpy())
    pd.DataFrame(sample_dict).describe().to_csv(savepath+filename+'.cvs')
    return

def wav_to_spectrogram(file_path:str, label:int, frames: int):
    wav = load_wav_output_mono_channel_file(file_path)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)

    #Adds a second dimension 
    spectrogram = tf.expand_dims(spectrogram, axis = 2)

    return spectrogram, label

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    # Scale magnitude relative to maximum value in S. Zeros in the output 
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def wav_to_mels_spectrogram(file_path:str, label:int, frames:int):
    wav = load_wav_output_mono_channel_file(file_path)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav,
     frame_length = 320, ##This is fft_size
     frame_step = 32 ## this is hop_size
        ) #

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)
    
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=100,
        num_spectrogram_bins = 257,
        sample_rate=frames,
        lower_edge_hertz=frames/100,
        upper_edge_hertz=frames/2,
        dtype=tf.dtypes.float32)

    mel_power_spectrogram = tf.matmul(tf.square(spectrogram), mel_filter)

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrogram)

    log_magnitude_mel_spectrograms = tf.expand_dims(log_magnitude_mel_spectrograms, axis = 2)

    return log_magnitude_mel_spectrograms, label

def plot_spectrogram_subplots(word:str, labels:int, data_dir:str, frames:str, spec_type = 'spec', savepath = './artifacts/'):
    wav_dict ={}
    fig = plt.figure(figsize = (30, 25))
    plt.tight_layout()
    for i in range(len(labels)):
        temp_dir = data_dir+labels[i]+'/'
        temp_file_list = tf.data.Dataset.list_files(temp_dir+word+' .wav')
        temp = tf.data.Dataset.zip((temp_file_list, tf.data.Dataset.from_tensor_slices(tf.ones(len(temp_file_list))*i)))
        temp_wav, temp_label = temp.as_numpy_iterator().next()
        if spec_type == 'spec':
            wav_dict[temp_label], _ = wav_to_spectrogram(temp_wav, temp_label, frames)            
        else:
            wav_dict[temp_label], _ = wav_to_mels_spectrogram(temp_wav, temp_label, frames)
        ax = plt.subplot(len(labels), 1, i+1)
        ax.set_title(labels[i]+word+' spectrogram')
        plt.imshow(tf.transpose(wav_dict[temp_label])[0])
    
    fig.savefig(savepath+word+'_'+'_'+spec_type+'_subplot.png')
    plt.close()

def generate_tf_dataset(data_dir:str, class_labels:list, frames:int, spec_type = 'spec'):
    for i in range(len(class_labels)):
        temp_dir = data_dir+class_labels[i]+'/'
        temp_file_list = tf.data.Dataset.list_files(temp_dir+'*.wav')
        temp = tf.data.Dataset.zip((temp_file_list, tf.data.Dataset.from_tensor_slices(tf.ones(len(temp_file_list))*i)))
        if i == 0:
            data = temp
        else:
            data = data.concatenate(temp)
    frames = tf.constant(frames, dtype=tf.int64)
    if spec_type == 'spec':
        data = data.map(lambda filepath, label: wav_to_spectrogram(filepath, label, frames))
    else:
        data = data.map(lambda filepath, label: wav_to_mels_spectrogram(filepath, label, frames))
    data = data.shuffle(3)
    data = data.cache()
    data = data.batch(16)
    data = data.prefetch(10)
    return data
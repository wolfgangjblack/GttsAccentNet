import os
from gtts import gTTS #google Text to Speech
from pydub import AudioSegment

import time

def create_lexicon_from_file(filepath:str):
    try:
        os.path.isfile(filepath)
    except FileNotFoundError:
        print("Can't locate .txt file")

    with open(filepath) as f:
        lines = [line for line in f]

    lexicon = []
    for line in lines:
        lexicon.append(line.split('\n')[0])
    
    return lexicon

def generate_data_dir(data_dir: str):
    try:
        os.path.isdir(data_dir)
    except: 
        print("data directory not initialized, will create now")
        os.mkdir(data_dir)
        print("directory for data made at", data_dir)
    return

def check_data_and_lexicon(data_dir:str, lexicon:list):
    '''
  path is the path where the data lives, this is the base data foundaccent is the accent for the lexicon lexicon is the list of n words for google to generate voice files for. here, we've got 3k
    '''

    #check if directory we're saving to exists - if it does exist, we need to know how many of our
    # words already exists within the data. We're assuming this data comes from the orignal source
    # and that we're not mixing lexicons
    if os.path.exists(data_dir):
        print('path exists, changing to path \n')
        os.chdir(data_dir)
        current = os.listdir('.')
        current.sort()
        last_item = current[-1].split('.')[0]
        ind = lexicon.index(last_item)
        print('returning reduced lexicon')
        return lexicon[ind+1:]
    # If data dir doesn't exist, this means no data for this class has been generated - thus we'll
    # create the class directory and return the full lexicon for data generation
    else:
      print('creating class directory within data directory \n')
      os.mkdir(data_dir)
      print('changing to class directory \n')
      os.chdir(data_dir)
      print('returning lexicon')
      return lexicon 

def call_gtts_API_gen_data(savepath: str, language:str, accent:str, lexicon: list, wait_time: int):
    """This function calls the gtts API and uses it to generate an .mp3 file at the savepath data directory.
    inputs:
        savepath: class labeled data pathway
        language: argument used in gTTS, the root language to be recorded and saved
        accent: argument used in gTTS, the accent type to be recorded
        lexicon: a list of words for audio files
        wait_time: an int, representing seconds, to wait between API call
    """
    for i in lexicon:
        gtts_object = gTTS(text = i,
        lang = language,
        tld = accent,
        slow = False)

        gtts_object.save(i+'.mp3')
        time.sleep(wait_time)
    return

def change_mp3_to_wav():
    """this function assumes one is in the data_dir just after call_gtts_API_gen_data has been called
    it steps through all .mp3 files generated and uses AudioSegment to transform them into .wav files
    """
    files = [f for f in os.listdir() if '.mp3' in f]
    for file in files:
        sound = AudioSegment.from_mp3(file)
        sound.export(file.split('.')[0]+'.wav', format = 'wav')
        os.remove(file)
    return
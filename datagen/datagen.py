
##Expect this to take many hours - especially with 3k words and 5 accents

import os
import numpy as np
import time
from gtts import gTTS #google Text to Speech

from utils.datagen_utils import (create_lexicon_from_file,
                                 generate_data_dir,
                                 check_data_and_lexicon,
                                 call_gtts_API_gen_data,
                                 change_mp3_to_wav)

lexicon_path = 'common_words.txt'
language = 'english'
tld_list = ['com.au', 'co.uk', 'us', 'ca', 'co.in']
data_dir = '../data/'
class_labels = ['au', 'uk', 'us', 'ca', 'in']
wait_time = 10

cwd = os.getcwd()

##Create a lexicon of words from the common words .txt
lexicon = create_lexicon_from_file(lexicon_path)

##verify data dir exists, otherwise, create it
generate_data_dir(data_dir)

for i in range(len(tld_list)):
    accent = tld_list[i]
    savepath = '../data/'+class_labels[i]+'/'

    # Verify lexicon - in case we've been interrupted previously, this will allow us to not 
    # waste cycles on already generated data
    lexicon = check_data_and_lexicon(savepath, lexicon)

    # take the verified lexicon and save data as .mp3s using gTTS to generate data. wait_time
    #  should be increased if api timeout or api requests per second error occurs
    call_gtts_API_gen_data(savepath, language, accent, lexicon, wait_time)
    
    # this takes all .mp3 files in a directory and changes them to .wav files. 
    change_mp3_to_wav()

    #Return to 'base' directory
    os.chdir(cwd)



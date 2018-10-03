
# coding: utf-8

# Conversion of nb5a notebook from https://github.com/fatchord/WaveRNN

# ## Alternative Model (Preprocessing)
# You need to run this before you run notebook 5b.
# 
# The wavs in your dataset will be converted to 9bit linear and 80-band mels.

# In[1]:

import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
from .utils import *
from .dsp import *

bits = 9
model_name = 'tacorn_wavernn'

DATA_PATH = f'data/{model_name}/'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

GTA = True

if not GTA:
    # Point SEG_PATH to a folder containing your training wavs 
    # Doesn't matter if it's LJspeech, CMU Arctic etc. it should work fine
    SEG_PATH = 'TODO' 

    def get_files(path, extension='.wav') :
        filenames = []
        for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
            filenames += [filename]
        return sorted(filenames)

    wav_files = get_files(SEG_PATH)
    print(len(wav_files))

    def convert_file(path) :
        wav = load_wav(path, encode=False)
        mel = melspectrogram(wav)
        quant = (wav + 1.) * (2**bits - 1) / 2
        return mel.astype(np.float32), quant.astype(np.int)


    m, x = convert_file(wav_files[0])
    wav_files[0]

    #plot_spec(m)
    #plot(x)

    x = 2 * x / (2**bits - 1) - 1

    librosa.output.write_wav(DATA_PATH + 'test_quant.wav', x, sr=sample_rate)


    QUANT_PATH = DATA_PATH + 'quant/'
    MEL_PATH = DATA_PATH + 'mel/'
    if not os.path.exists(QUANT_PATH):
        os.makedirs(QUANT_PATH)
    if not os.path.exists(MEL_PATH):
        os.makedirs(MEL_PATH)

    print(wav_files[0].split('/')[-1][:-4])

    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(wav_files) :
        id = path.split('/')[-1][:-4]
        dataset_ids += [id]
        m, x = convert_file(path)
        np.save(f'{MEL_PATH}{id}.npy', m)
        np.save(f'{QUANT_PATH}{id}.npy', x)
        print('%i/%i' % (i + 1, len(wav_files)))

    with open(DATA_PATH + 'dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)

# GTA processing
else:
    AUDIO_PATH = "tacotron2/training_data/audio/"
    GTA_MEL_PATH = "tacotron2/tacotron_output/gta/"

    def get_files(path, extension='.wav') :
        filenames = []
        for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
            filenames += [filename]
        return sorted(filenames)

    audio_files = get_files(AUDIO_PATH, extension=".npy")
    mel_files = get_files(GTA_MEL_PATH, extension=".npy")
    #print ((audio_files[0].split('/')[-1][:-4], mel_files[0].split('/')[-1][:-4]))

    def convert_gta_mel(mel_path):
        mel = np.load(mel_path).T
        return mel.astype(np.float32)

    def convert_gta_audio(audio_path):
        audio = np.load(audio_path)
        quant = (audio + 1.) * (2**bits - 1) / 2
        return quant.astype(np.int)

    #m, x = convert_gta_file(audio_files[0], mel_files[0])
    #plot_spec(m)
    #plot(x)

    QUANT_PATH = DATA_PATH + 'quant/'
    MEL_PATH = DATA_PATH + 'mel/'
    if not os.path.exists(QUANT_PATH):
        os.makedirs(QUANT_PATH)
    if not os.path.exists(MEL_PATH):
        os.makedirs(MEL_PATH)

    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(zip(audio_files, mel_files)):
        audio_id = path[0].split('/')[-1][6:-4]
        mel_id = path[1].split('/')[-1][4:-4]
        assert(mel_id == audio_id)
        dataset_ids += [audio_id]
        x = convert_gta_audio(path[0])
        m = convert_gta_mel(path[1])
        np.save(f'{QUANT_PATH}{audio_id}.npy', x)
        np.save(f'{MEL_PATH}{mel_id}.npy', m)
        print('%i/%i : audio: %s mel: %s' % (i + 1, len(audio_files), audio_id, mel_id))
    dataset_ids_unique = list(set(dataset_ids))

    with open(DATA_PATH + 'dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids_unique, f)

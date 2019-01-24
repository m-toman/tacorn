""" Wraps functionality of the alternative WaveRNN model. """
import os
import sys
import zipfile
import logging
import pickle
from collections import namedtuple
from typing import Mapping
# import tensorflow as tf
import numpy as np

import tacorn.fileutils as fu
# import tacorn.constants as constants
from tacorn.experiment import Experiment

#sys.path.insert(0, 'wavernn')
import wavernn.preprocess as wpreproc
import wavernn.train as wtrain
import wavernn.synthesize as wsynth


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
LOGGER = logging.getLogger(__name__)
NUM_TEST_FILES = 4


def _get_pretrained_folder(experiment: Experiment) -> str:
    return os.path.join(experiment.paths["wavegen_model"], "pretrained")


def _get_hparams_path(experiment: Experiment) -> str:
    return os.path.join(experiment.paths["config"], "wavernn_hparams.py")


def _check_pretrained_model(experiment: Experiment) -> None:
    pretrained_dir = _get_pretrained_folder(experiment)
    # checkpoint_file = os.path.join(pretrained_dir, "checkpoint")
    # if not os.path.exists(pretrained_dir):
    #    raise FileNotFoundError(pretrained_dir)
    # if not os.path.exists(checkpoint_file):
    #    raise FileNotFoundError(checkpoint_file)


def download_pretrained(experiment: Experiment, url: str) -> None:
    """ Downloads a pretrained model. """
    pretrained_dir = _get_pretrained_folder(experiment)
    pretrained_zip = os.path.join(pretrained_dir, "wavernnalt_pretrained.zip")
    fu.ensure_dir(pretrained_dir)
    fu.download_file(url, pretrained_zip)
    with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
        zip_ref.extractall(pretrained_dir)
    os.remove(pretrained_zip)
    # TODO copy params associated with this model to config folder
    # fu.copy_file(os.path.join(pretrained_dir, "hparams.py"),
    #             _get_hparams_path(experiment))


def create(experiment: Experiment, args) -> None:
    """ Creates specific folders and configuration. """
    # copy default hparams if not exists yet
    hparams_path = _get_hparams_path(experiment)
    if not os.path.exists(hparams_path):
        fu.copy_file(os.path.join(
            "wavernn", "hyperparams.py"), hparams_path)


def convert_training_data(experiment: Experiment, args: Mapping) -> None:
    """ Converts output of acoustic model for training, returns None or raises an Exception. """
    # TODO: load voice specific hparams
    # wpreproc.hp.override_from_dict(otherhparams.values())
    dataset_ids = []
    map_file = os.path.join(
        experiment.paths["acoustic2wavegen_training_features"], "map.txt")
    output_dir = experiment.paths["wavegen_features"]
    output_wav_dir = os.path.join(output_dir, "wav")
    output_mel_dir = os.path.join(output_dir, "mel")
    output_test_dir = os.path.join(output_dir, "test")
    fu.ensure_dir(output_wav_dir)
    fu.ensure_dir(output_mel_dir)
    fu.ensure_dir(output_test_dir)

    if not os.path.exists(map_file):
        # in future we might find the files by some other means
        raise FileNotFoundError(map_file)

    # read map file to get paths to audio, original mel and GTA mel
    gta_map = np.genfromtxt(map_file, dtype=None, delimiter='|')
    for i, line in enumerate(gta_map):
        (audio_file, mel_file, gta_file, _, _) = line
        fileid = "%05d" % (i)
        LOGGER.info("Converting %s and %s" % (audio_file, gta_file))
        gta_mel = np.load(gta_file.decode("utf-8")).T
        #audio, mel = wpreproc.get_wav_mel(audio_file.decode("utf-8"))
        audio = wpreproc.get_wav(audio_file.decode("utf-8"))

        if i < NUM_TEST_FILES:
            np.save(os.path.join(output_test_dir, "test_{}_mel.npy".format(
                i)), gta_mel.astype(np.float32))
            np.save(os.path.join(output_test_dir,
                                 "test_{}_wav.npy".format(i)), audio)
        else:
            np.save(os.path.join(output_mel_dir, fileid),
                    gta_mel.astype(np.float32))
            np.save(os.path.join(output_wav_dir, fileid), audio)
            dataset_ids.append(fileid)

    with open(os.path.join(output_dir, 'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)


def train(experiment: Experiment, args) -> None:
    """ Trains a Tacotron-2 model. """
    # TODO: only do conversion if necessary?
    # TODO: train from GTA or from raw?
    #convert_training_data(experiment, args)
    features_dir = experiment.paths["wavegen_features"]
    pretrained_dir = _get_pretrained_folder(experiment)
    checkpoint_file = os.path.join(pretrained_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = None
    wtrain.main(features_dir, pretrained_dir, checkpoint_file)


def generate(experiment: Experiment, sentences=None) -> None:
    """
    Generates waveforms from existing acoustic features.

    :param Experiment experiment: The experiment to generate from.
    :param sentences: List of feature files to synthesize. If None synthesizes all files in the feature folder. 
    """
    mels_dir = experiment.paths["acoustic2wavegen_features"]
    output_dir = os.path.join(
        experiment.paths["synthesized_wavs"], "wavernn")
    fu.ensure_dir(output_dir)
    pretrained_dir = _get_pretrained_folder(experiment)
    checkpoint_file = os.path.join(pretrained_dir, "checkpoint.pth")
    map_file = os.path.join(mels_dir, "eval", "map.txt")
    print(map_file)
    map_content = np.genfromtxt(map_file, dtype=None, delimiter='|')
    for i, line in enumerate(map_content):
        (text, mel_file, _) = line
        LOGGER.info("Generating waveform from %s with text: %s" % (line, text))
        mel_file = mel_file.decode("utf-8")
        mel = np.load(mel_file).T
        output_file = os.path.join(output_dir, os.path.basename(mel_file))
        wsynth.main(mel, checkpoint_file, output_file)

""" Wraps functionality of the alternative WaveRNN model. """
import os
import sys
import multiprocessing
import zipfile
from collections import namedtuple
from typing import Mapping
import tensorflow as tf

import tacorn.fileutils as fu
import tacorn.constants as constants
from tacorn.experiment import Experiment

sys.path.append('wavernn_alt')
import wavernn_alt.preprocess as wpreproc


def _get_pretrained_folder(experiment: Experiment) -> str:
    return os.path.join(experiment.paths["wavegen_model"], "pretrained")


def _get_hparams_path(experiment: Experiment) -> str:
    return os.path.join(experiment.paths["config"], "wavernn_alt_hparams.py")


def _check_pretrained_model(experiment: Experiment) -> None:
    pretrained_dir = _get_pretrained_folder(experiment)
    # checkpoint_file = os.path.join(pretrained_dir, "checkpoint")
    # if not os.path.exists(pretrained_dir):
    #    raise FileNotFoundError(pretrained_dir)
    # if not os.path.exists(checkpoint_file):
    #    raise FileNotFoundError(checkpoint_file)


def download_pretrained(experiment: Experiment, url: str) -> None:
    """ Downloads a pretrained model. """
    pretrained_dir = os.path.join(
        experiment.paths["wavegen_model"], "pretrained")
    pretrained_zip = os.path.join(pretrained_dir, "wavernnalt_pretrained.zip")
    fu.ensure_dir(pretrained_dir)
    fu.download_file(url, pretrained_zip)
    with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
        zip_ref.extractall(pretrained_dir)
    os.remove(pretrained_zip)
    # copy params associated with this model to config folder
    fu.copy_file(os.path.join(pretrained_dir, "hparams.py"),
                 _get_hparams_path(experiment))


def create(experiment: Experiment, args) -> None:
    """ Creates specific folders and configuration. """
    # copy default hparams if not exists yet
    hparams_path = _get_hparams_path(experiment)
    if not os.path.exists(hparams_path):
        fu.copy_file(os.path.join("wavernn_alt", "hparams.py"), hparams_path)


def convert_training_data(experiment: Experiment, args: Mapping) -> None:
    """ Converts output of acoustic model for training, returns None or raises an Exception. """
    # wavegen_features for output of acoustic model
    # raw_wavs for raw wavs
    # wavegen_features
    # wpreproc.hp.override_from_dict(otherhparams.values())


def train(experiment: Experiment, args) -> None:
    """ Trains a Tacotron-2 model. """
    tacoargs = namedtuple(
        "tacoargs", "mels_dir output_dir mode base_dir hparams tacotron_input name model input_dir GTA restore summary_interval embedding_interval checkpoint_interval eval_interval tacotron_train_steps tf_log_level slack_url".split())  # name?
    tacoargs.base_dir = experiment.paths["acoustic_model"]
    tacoargs.hparams = ''
    # tacoargs.name
    tacoargs.tacotron_input = os.path.join(
        experiment.paths["acoustic_features"], "train.txt")
    tacoargs.input_dir = experiment.paths["acoustic_features"]
    tacoargs.model = 'Tacotron'
    tacoargs.name = None
    tacoargs.tf_log_level = 1
    tacoargs.GTA = True
    tacoargs.restore = True
    tacoargs.summary_interval = 250
    tacoargs.embedding_interval = 10000
    tacoargs.checkpoint_interval = 1000
    tacoargs.eval_interval = 500
    tacoargs.tacotron_train_steps = int(args["acoustic_max_steps"])
    tacoargs.slack_url = None

    tacoargs.mels_dir = experiment.paths["wavegen_features"]
    tacoargs.output_dir = experiment.paths["wavegen_features"]
    tacoargs.mode = 'synthesis'

    log_dir, hparams = tacotron2.train.prepare_run(tacoargs)
    checkpoint = tacotron2.tacotron.train.tacotron_train(
        tacoargs, log_dir, hparams)
    tf.reset_default_graph()
    input_path = tacotron2.synthesize.tacotron_synthesize(
        tacoargs, hparams, checkpoint)
    print("input path: " + input_path)


def generate_wavegen_features(experiment: Experiment, args) -> None:
    """ Generate features for the wavegen model. """
    # TODO
    return


def generate(experiment: Experiment, sentences, generate_features: bool=True, generate_waveforms: bool=True) -> None:
    """
    Generates from the model.

    :param Experiment experiment: The experiment to generate from.
    :param bool generate_features: Store acoustic features
    :param bool generate_waveforms: Generate a waveform from acoustic features using Griffin-Lim
    """
    # python synthesize.py --model Tacotron --tacotron_name Tacotron-2 --mode eval --text_list text_list.txt &> /dev/null
    tacoargs = namedtuple(
        "tacoargs", "mode model checkpoint output_dir mels_dir hparams name tacotron_name GTA".split())
    tacoargs.checkpoint = _get_pretrained_folder(experiment)
    tacoargs.hparams = ''
    tacoargs.name = "Tacotron"
    tacoargs.tacotron_name = "Tacotron"
    tacoargs.model = "Tacotron"
    # tacoargs.input_dir = 'training_data/'
    tacoargs.mels_dir = experiment.paths["wavegen_features"]
    tacoargs.output_dir = experiment.paths["wavegen_features"]
    tacoargs.mode = "eval"
    tacoargs.GTA = False
    # tacoargs.base_dir = ''
    # tacoargs.log_dir = None
    # taco_checkpoint, _, hparams = tacotron2.synthesize.prepare_run(tacoargs)
    modified_hp = tacotron2.hparams.hparams.parse(tacoargs.hparams)
    # taco_checkpoint = os.path.join("tacotron2", taco_checkpoint)
    tacotron2.tacotron.synthesize.tacotron_synthesize(
        tacoargs, modified_hp, tacoargs.checkpoint, sentences)
    fu.copy_files(os.path.join(experiment.paths["wavegen_features"], "logs-eval", "wavs"),
                  experiment.paths["synthesized_wavs"])

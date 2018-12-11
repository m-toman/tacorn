""" Wraps Tacotron-2 functionality. """
import os
import sys
import multiprocessing
import zipfile
from collections import namedtuple
from typing import Mapping

import tacorn.fileutils as fu
import tacorn.constants as constants
from tacorn.experiment import Experiment

sys.path.append('tacotron2')
import tacotron2.train
import tacotron2.preprocess
import tacotron2.hparams
import tacotron2.tacotron.train


def _check_pretrained_model(experiment: Experiment) -> None:
    pretrained_dir = os.path.join(
        experiment.paths["acoustic_model"], "logs-Tacotron", "taco_pretrained")
    checkpoint_file = os.path.join(pretrained_dir, "checkpoint")
    if not os.path.exists(pretrained_dir):
        raise FileNotFoundError(pretrained_dir)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(checkpoint_file)


def download_pretrained(experiment: Experiment, url: str) -> None:
    """ Downloads a pretrained model. """
    pretrained_dir = os.path.join(
        experiment.paths["acoustic_model"], "logs-Tacotron")
    pretrained_zip = os.path.join(pretrained_dir, "taco_pretrained.zip")
    fu.ensure_dir(pretrained_dir)
    fu.download_file(url, pretrained_zip)
    with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
        zip_ref.extractall(pretrained_dir)
    os.remove(pretrained_zip)


def create(experiment: Experiment, args) -> None:
    """ Creates Tacotron-2 specific folders and configuration. """
    return


def preprocess(experiment: Experiment, args: Mapping) -> None:
    """ Preprocesses wavs and text, returns None or raises an Exception. """
    # bring data in format usable by Tacotron-2
    # for now just copy them over
    raw_wav_dir = os.path.join(
        experiment.paths["acoustic_model"], "LJSpeech-1.1", "wavs")
    fu.ensure_dir(raw_wav_dir)
    fu.copy_files(args["wav_dir"], raw_wav_dir)
    raw_metadata_file = os.path.join(
        experiment.paths["acoustic_model"], "LJSpeech-1.1", "metadata.csv")
    fu.copy_file(args["text_file"], raw_metadata_file)

    # run Tacotron-2 preprocessing
    tacoargs = namedtuple(
        "tacoargs", "base_dir hparams dataset language voice reader merge_books book output n_jobs".split())
    tacoargs.base_dir = experiment.paths["acoustic_model"]
    tacoargs.language = experiment.config["language"]
    tacoargs.output = experiment.paths["acoustic_features"]
    tacoargs.hparams = ""
    tacoargs.n_jobs = multiprocessing.cpu_count()
    # for now we always exploit the LJ settings
    tacoargs.dataset = "LJSpeech-1.1"
    tacoargs.voice = "female"
    tacoargs.reader = "LJ"
    tacoargs.merge_books = "True"
    tacoargs.book = "northandsouth"

    modified_hp = tacotron2.hparams.hparams.parse(tacoargs.hparams)
    tacotron2.preprocess.run_preprocess(tacoargs, modified_hp)


def train(experiment: Experiment, args) -> None:
    """ Trains a Tacotron-2 model. """
    args = namedtuple(
        "tacoargs", "base_dir hparams tacotron_input name model input_dir GTA restore summary_interval embedding_interval checkpoint_interval eval_interval tacotron_train_steps tf_log_level slack_url".split())  # name?
    args.base_dir = experiment.paths["acoustic_model"]
    args.hparams = ''
    # args.name
    args.tacotron_input = os.path.join(
        experiment.paths["acoustic_features"], "train.txt")
    args.input_dir = experiment.paths["acoustic_features"]
    args.model = 'Tacotron'
    args.name = None
    args.tf_log_level = 1
    args.GTA = True
    args.restore = True
    args.summary_interval = 250
    args.embedding_interval = 10000
    args.checkpoint_interval = 5000
    args.eval_interval = 10000
    args.tacotron_train_steps = 150000
    args.slack_url = None

    log_dir, hparams = tacotron2.train.prepare_run(args)
    tacotron2.tacotron.train.tacotron_train(args, log_dir, hparams)


def generate(experiment: Experiment, generate_features: bool = True, generate_waveforms: bool = True) -> None:
    """
    Generates from the model.

    :param Experiment experiment: The experiment to generate from.
    :param bool generate_features: Store acoustic features
    :param bool generate_waveforms: Generate a waveform from acoustic features using Griffin-Lim
    """
    pass

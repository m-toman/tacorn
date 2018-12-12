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
import tacotron2.tacotron.synthesize
import tacotron2.synthesize


def _get_pretrained_folder(experiment: Experiment) -> str:
    return os.path.join(experiment.paths["acoustic_model"], "logs-Tacotron", "taco_pretrained")


def _check_pretrained_model(experiment: Experiment) -> None:
    pretrained_dir = _get_pretrained_folder(experiment)
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
    tacoargs = namedtuple(
        "tacoargs", "base_dir hparams tacotron_input name model input_dir GTA restore summary_interval embedding_interval checkpoint_interval eval_interval tacotron_train_steps tf_log_level slack_url".split())  # name?
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
    tacoargs.eval_interval = 1000
    tacoargs.tacotron_train_steps = args["acoustic_max_steps"]
    tacoargs.slack_url = None

    log_dir, hparams = tacotron2.train.prepare_run(tacoargs)
    tacotron2.tacotron.train.tacotron_train(tacoargs, log_dir, hparams)


def generate(experiment: Experiment, sentences, generate_features: bool = True, generate_waveforms: bool = True) -> None:
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
    #tacoargs.input_dir = 'training_data/'
    tacoargs.mels_dir = experiment.paths["wavegen_features"]
    tacoargs.output_dir = experiment.paths["wavegen_features"]
    tacoargs.mode = "eval"
    tacoargs.GTA = False
    #tacoargs.base_dir = ''
    #tacoargs.log_dir = None
    print("tacoargs: " + str(vars(tacoargs)))
    #taco_checkpoint, _, hparams = tacotron2.synthesize.prepare_run(tacoargs)
    modified_hp = tacotron2.hparams.hparams.parse(tacoargs.hparams)
    #taco_checkpoint = os.path.join("tacotron2", taco_checkpoint)
    tacotron2.tacotron.synthesize.tacotron_synthesize(
        tacoargs, modified_hp, tacoargs.checkpoint, sentences)

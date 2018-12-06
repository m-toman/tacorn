""" Wraps Tacotron-2 functionality. """
import sys
import multiprocessing
from collections import namedtuple

import tacorn.fileutils as fu
from tacorn.experiment import Experiment

sys.path.append('tacotron2')
import tacotron2.preprocess
import tacotron2.hparams


def create(experiment: Experiment, args) -> None:
    """ Creates Tacotron-2 specific folders and configuration. """
    return


def preprocess(experiment: Experiment, args) -> None:
    """ Preprocesses wavs and text, returns None or raises an Exception. """
    # bring data in format usable by Tacotron-2
    # for now just copy them over
    raw_wav_dir = os.path.join(
        experiment.paths["feature_model"], "LJSpeech-1.1", "wavs")
    fu.ensure_dir(raw_wav_dir)
    fu.copy_files(args["wav_dir"], raw_wav_dir)
    raw_metadata_file = os.path.join(
        experiment.paths["feature_model"], "LJSpeech-1.1", "metadata.csv")
    fu.copy_file(args["text_file"], raw_metadata_file)

    # run Tacotron-2 preprocessing
    args = namedtuple(
        "tacoargs", "base_dir hparams dataset language voice reader merge_books book output n_jobs".split())
    args.base_dir = experiment.paths["feature_model"]
    args.language = experiment.config["language"]
    args.output = experiment.paths["features"]
    args.hparams = ""
    args.n_jobs = multiprocessing.cpu_count()
    # for now we always exploit the LJ settings
    args.dataset = "LJSpeech-1.1"
    args.voice = "female"
    args.reader = "LJ"
    args.merge_books = "True"
    args.book = "northandsouth"

    modified_hp = tacotron2.hparams.parse(args.hparams)
    tacotron2.preprocess.run_preprocess(args, modified_hp)


def train(experiment: Experiment, args) -> None:
    """ Trains a Tacotron-2 model. """
    args = namedtuple(
        "tacoargs", "base_dir hparams tacotron_input name model input_dir".split())  # name?
    args.base_dir = experiment.paths["feature_model"]
    args.hparams = ''
    # args.name
    args.tacotron_input = os.path.join(
        experiment.paths["feature_model"], "training_data", "train.txt")
    args.input_dir = 'training_data'
    args.model = 'Tacotron'

    log_dir, hparams = tacotron2.train.prepare_run(args)
    tacotron2.tacotron.train.tacotron_train(args, log_dir, hparams)
""" Wraps Tacotron-2 functionality. """

from collections import namedtuple
import multiprocessing
from tacorn.experiment import Experiment

sys.path.append('tacotron2')
import tacotron2.preprocess
import tacotron2.hparams


def preprocess(experiment: Experiment, wav_dir: str, text_file: str) -> None:
    """ Preprocesses wavs and text, returns None or raises an Exception. """
    args = namedtuple(
        "tacoargs", "base_dir hparams dataset language voice reader merge_books book output n_jobs".split())
    args.base_dir = experiment.paths["root"]
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

    return

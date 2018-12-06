""" Wraps Tacotron-2 functionality. """

from collections import namedtuple
import multiprocessing
import tacorn.fileutils as fu
from tacorn.experiment import Experiment

sys.path.append('tacotron2')
import tacotron2.preprocess
import tacotron2.hparams


def create(experiment: Experiment, args) -> None:
    """ Creates Tacotron-2 specific folders and configuration. """
    experiment.paths["tacotron2_root"] = os.path.join(experiment.paths["root"], )


def preprocess(experiment: Experiment, wav_dir: str, text_file: str) -> None:
    """ Preprocesses wavs and text, returns None or raises an Exception. """

    # bring data in format usable by Tacotron-2
    # for now just copy them over
    raw_wav_dir = os.path.join(experiment.paths["base_dir"], "wavs")
    fu.ensure_dir(raw_wav_dir)
    fu.copy_files(wav_dir, raw_wav_dir)
    raw_metadata_file = os.path.join(experiment.paths["raw"], "metadata.csv")
    fu.copy_file(text_file, raw_metadata_file)

    # run Tacotron-2 preprocessing
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

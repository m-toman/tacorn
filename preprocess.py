""" preprocess.py
    Preprocesses raw wavs and texts in a given experiment folder.
"""

import sys
import argparse
import importlib

import logging
import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def preprocess(exp: experiment.Experiment):
    """ Preprocesses data given in args using the experiment
        stored in exp. """
    logger.info("Loading feature model wrapper %s" %
                (experiment.config["feature_model"]))
    wrapper_module = importlib.import_module(
        "tacorn." + experiment.config["feature_model"] + "_wrapper")
    logger.info("Preprocessing")
    wrapper_module.preprocess(
        exp, wav_dir=args["wav_dir"], text_file=args["text_file"])
    logger.info("Preprocessing done")
    


def main():
    """ main function for preprocessing data. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Experiment directory.')
    parser.add_argument('--wav_dir', default=None,
                        help='Folder containing wavefiles, if no given the corpus is assumed to already be in experiment/raw')
    parser.add_argument('--text_file', default=None,
                        help='Text file containing transcriptions, if not given the corpus is assumed to already be in experiment/raw')
    args = parser.parse_args()

    try:
        exp = experiment.load(args.experiment_dir)
    except Exception:
        print("Invalid experiment folder given: %s" % (args.experiment_dir))
        sys.exit(1)

    preprocess(exp, args)


if __name__ == '__main__':
    main()

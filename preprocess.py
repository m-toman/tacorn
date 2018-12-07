""" preprocess.py
    Preprocesses raw wavs and texts in a given experiment folder.
"""

import os
import sys
import argparse
import importlib

import logging
import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment
import tacorn.wrappers as wrappers

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _get_raw(exp: experiment.Experiment, args):
    """ Retrieves raw data, performing potential pre-preprocessing before
        feeding into the preprocessing of the feature prediction model. """
    # for now just check text file format
    return


def preprocess(exp: experiment.Experiment, args):
    """ Preprocesses data given in args using the experiment
        stored in exp. """
    #_get_raw(exp, args)
    logger.info("Loading feature model wrapper %s for preprocessing" %
                (exp.config["feature_model"]))
    wrappers.load(exp.config["feature_model"]).preprocess(exp, vars(args))
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

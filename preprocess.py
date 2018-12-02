""" preprocess.py
    Preprocesses raw wavs and texts in a given experiment folder.
"""

import sys
import argparse

import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment


def preprocess(exp, args):
    """ Preprocesses data given in args using the experiment
        stored in exp. """
    paths = exp.paths
    print(paths)
    print(exp.config)


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

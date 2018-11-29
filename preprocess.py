""" preprocess.py
    Preprocesses raw wavs and texts in a given experiment folder.
"""

import argparse

import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment


# def preprocess(paths, args):


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
        paths = experiment.check_file_structure(args.experiment_dir)
    except FileNotFoundError:
        print("Invalid experiment folder given: %s" % (args.experiment_dir))

    preprocess(paths, args)


if __name__ == '__main__':
    main()

""" create.py
    Creates and initializes a new experiment folder.
"""

import os
import argparse
import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment
import tacorn.wrappers as wrappers


def get_pretrained_wavernn(model_id, targetdir):
    """ Downloads a pretrained waveRNN model. """
    if model_id not in consts.PRETRAINED_WAVERNN_MODELS:
        raise FileNotFoundError(model_id)

    (model_url, model_filename) = consts.PRETRAINED_WAVERNN_MODELS[model_id]
    model_dir = os.path.join(targetdir, model_id)
    model_path = os.path.join(model_dir, model_filename)
    if os.path.exists(model_path):
        return model_path

    fu.ensure_dir(model_dir)
    fu.download_file(model_url, model_path)
    return model_path


def main():
    """ main function for creating a new experiment directory. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Directory for the experiment to create.')
    parser.add_argument('--feature_model', default="tacotron2",
                        help='Model to use for feature prediction.')
    parser.add_argument('--wavegen_model', default="wavernn",
                        help='Model to use for waveform generation.')
    args = parser.parse_args()

    exp = experiment.create(args.experiment_path, args)
    wrappers.load(exp.config["feature_model"]).create(exp, args)


if __name__ == '__main__':
    main()

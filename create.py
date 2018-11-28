""" create.py
    Creates and initializes a new experiment folder.
"""

import os
import argparse
import tacorn.fileutils as fu
import tacorn.constants as consts


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


def create_experiment_subfolder(experiment_path, folder):
    """ Creates a subfolder inside the experiment directory. """
    absfolder = os.path.join(experiment_path, folder)
    fu.ensure_dir(absfolder)
    return absfolder


def create_experiment_structure(experiment_path):
    """ Creates the folder structure for a new experiment. """
    fu.ensure_dir(experiment_path)
    paths = {}
    paths["raw"] = create_experiment_subfolder(experiment_path, "raw")
    paths["features"] = create_experiment_subfolder(
        experiment_path, "features")
    paths["workdir_taco2"] = create_experiment_subfolder(
        experiment_path, "workdir_taco2")
    paths["workdir_wavernn"] = create_experiment_subfolder(
        experiment_path, "workdir_wavernn")
    paths["synthesized_wavs"] = create_experiment_subfolder(
        experiment_path, "synthesized_wavs")
    # paths["wavernn_pretrained"] = create_experiment_subfolder(
    #    experiment_path, "wavernn_pretrained")
    return paths


def main():
    """ main function for creating a new experiment directory. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', default=None,
                        help='Directory for the experiment to create.')
    parser.add_argument('--feature_model', default="tacotron2",
                        help='Model to use for feature prediction.')
    parser.add_argument('--wavegen_model', default="wavernn",
                        help='Model to use for waveform generation.')
    args = parser.parse_args()

    paths = create_experiment_structure(args.experiment_path)


if __name__ == '__main__':
    main()

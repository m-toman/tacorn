""" Handles experiment folders. """
import os
from typing import MutableMapping
import tacorn.fileutils as fu


class Experiment(object):
    """ Represents an experiment. """

    def __init__(self):
        self.config = dict()
        self.paths: MutableMapping[str, str] = dict()

    def __repr__(self):
        return repr(self.config) + "\n" + repr(self.paths)


def _create_subfolder(experiment_path, folder):
    """ Creates a subfolder inside the experiment directory. """
    absfolder = os.path.join(experiment_path, folder)
    fu.ensure_dir(absfolder)
    return absfolder


def _check_subfolder(experiment_path, folder):
    """ Checks if a subfolder exists inside the experiment directory.
        Otherwise raises a FileNotFoundError. """
    absfolder = os.path.join(experiment_path, folder)
    if os.path.exists(absfolder):
        return absfolder
    raise FileNotFoundError(absfolder)


def _apply_file_structure(experiment_path, function):
    """ Applies function to all folder elements and returns a path map. """
    paths = {}
    paths["raw"] = function(experiment_path, "raw")
    paths["features"] = function(experiment_path, "features")
    paths["workdir_taco2"] = function(experiment_path, "workdir_taco2")
    paths["workdir_wavernn"] = function(experiment_path, "workdir_wavernn")
    paths["synthesized_wavs"] = function(experiment_path, "synthesized_wavs")
    # paths["wavernn_pretrained"] = function(experiment_path, "wavernn_pretrained")
    return paths


def create_file_structure(experiment_path):
    """ Creates the folder structure for a new experiment. """
    fu.ensure_dir(experiment_path)
    return _apply_file_structure(experiment_path, _create_subfolder)


def check_file_structure(experiment_path):
    """ Checks if all folders exist in the experiment folder
        and returns a path map, otherwise raises FileNotFoundError. """
    return _apply_file_structure(experiment_path, _check_subfolder)


def check_config(config):
    """ Checks a configuration for validity.
        Returns the config if valid, otherwise raises an ValueError """
    # TODO
    return config


def create(experiment_path, config) -> Experiment:
    """ Creates a new experiment at path. """
    exp = Experiment()
    exp.paths = create_file_structure(experiment_path)
    exp.config = check_config(config)
    # TODO: store config
    return exp


def load(experiment_path) -> Experiment:
    """ Loads an experiment or raises an Error. """
    exp = Experiment()
    exp.paths = check_file_structure(experiment_path)
    # TODO: load config
    # exp.config =
    return exp

""" Handles experiment folders. """
import os
import json
from typing import MutableMapping
import tacorn.fileutils as fu


class Experiment(object):
    """ Represents an experiment. """

    def __init__(self):
        self.config = {"language": "en_US",
                       "wavegen_model": "wavernn", "feature_model": "tacotron2"}
        self.paths: MutableMapping[str, str] = dict()

    def __repr__(self):
        return repr(self.config) + "\n" + repr(self.paths)

    def __eq__(self, other):
        return (self.config == other.config and
                self.paths == other.paths)


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
    paths["root"] = function(experiment_path, "")
    paths["raw"] = function(experiment_path, "raw")
    paths["features"] = function(experiment_path, "features")
    paths["config"] = function(experiment_path, "config")
    paths["feature_model"] = function(experiment_path, "feature_model")
    paths["wavegen_model"] = function(experiment_path, "wavegen_model")
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
    cfg_file = os.path.join(exp.paths["config"], "experiment_config.json")
    with open(cfg_file, "wt") as cfg_fp:
        json.dump(exp.config, cfg_fp)
    return exp


def load(experiment_path) -> Experiment:
    """ Loads an experiment or raises an Error. """
    exp = Experiment()
    exp.paths = check_file_structure(experiment_path)
    cfg_file = os.path.join(exp.paths["config"], "experiment_config.json")
    with open(cfg_file, "rt") as cfg_fp:
        exp.config = json.load(cfg_fp)
    return exp

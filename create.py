""" create.py
    Creates and initializes a new experiment folder.
"""
import os
import sys
import shutil
import argparse
import logging
import tacorn.fileutils as fu
import tacorn.constants as consts
import tacorn.experiment as experiment
import tacorn.wrappers as wrappers

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


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


def _get_download_url(download_map, args):
    return download_map[args.download_acoustic_model][args.acoustic_model]


def main():
    """ main function for creating a new experiment directory. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Directory for the experiment to create.')
    parser.add_argument('--acoustic_model', default="tacotron2",
                        help='Model to use for acoustic feature prediction (tacotron2).')
    parser.add_argument('--download_acoustic_model', default=None,
                        choices=consts.PRETRAINED_FEATURE_MODELS.keys(),
                        help=('Name of a pretrained feature model to download (%s)'
                              % (" ".join(consts.PRETRAINED_FEATURE_MODELS.keys()))))
    parser.add_argument('--wavegen_model', default="none",
                        help='Model to use for waveform generation (wavernn or none). Default: none')
    parser.add_argument('--force', action='store_const', const=True,
                        help='Forces creation of this experiment, deleting an existing experiment if necessary')
    args = parser.parse_args()

    if os.path.exists(args.experiment_dir):
        if args.force:
            logger.info("Deleting existing experiment at %s" %
                        (args.experiment_dir))
            shutil.rmtree(args.experiment_dir)
        else:
            print("Experiment already exists at %s, stopping" %
                  (args.experiment_dir))
            return -1

    logger.info("Creating experiment at %s" % (args.experiment_dir))
    exp = experiment.create(args.experiment_dir, args)
    try:
        module_wrapper = wrappers.load(exp.config["acoustic_model"])
        module_wrapper.create(exp, vars(args))
        if args.download_acoustic_model:
            logger.info("Downloading feature model %s" %
                        (args.download_acoustic_model))
            module_wrapper.download_pretrained(
                exp, _get_download_url(consts.PRETRAINED_FEATURE_MODELS, args))
        experiment.save(exp)
    except ModuleNotFoundError as mnfe:
        print(mnfe)
        print("Module for %s not found, did you run install.sh?" %
              (args.acoustic_model))
    return 0


if __name__ == '__main__':
    sys.exit(main())

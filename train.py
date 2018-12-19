import argparse
import logging

import tacorn.wrappers as wrappers
import tacorn.experiment as experiment

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def train(exp: experiment.Experiment, args) -> None:
    """ Trains acoustic feature prediction and waveform generation models. """
    # TODO: pause/resume training
    if args.model in ("acoustic", "both"):
        logger.info("Loading acoustic feature model wrapper %s for training" %
                    (exp.config["acoustic_model"]))
        module_wrapper = wrappers.load(exp.config["acoustic_model"])
        module_wrapper.train(exp, vars(args))
        logger.info("Training acoustic feature model done")
        # TODO: generate intermediate features if successful

    if args.model in ("wavegen", "both") and exp.config["wavegen_model"]:
        # TODO: check if intermediate features exist
        logger.info("Loading waveform generation model wrapper %s for training" %
                    (exp.config["wavegen_model"]))
        module_wrapper = wrappers.load(exp.config["wavegen_model"])
        module_wrapper.train(exp, vars(args))


def main():
    """ main function for training. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Experiment directory.')
    parser.add_argument('--model', default='both',
                        help='Which model to train: acoustic, wavegen, both. Default: both')
    parser.add_argument('--acoustic_max_steps', default=150000,
                        help='Maximum number of steps to train the acoustic model (including pretrained steps)')
    args = parser.parse_args()

    try:
        exp = experiment.load(args.experiment_dir)
    except Exception:
        print("Invalid experiment folder given: %s" % (args.experiment_dir))
        sys.exit(1)

    train(exp, args)


if __name__ == '__main__':
    main()

import argparse
import logging

import tacorn.wrappers as wrappers
import tacorn.experiment as experiment

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def train(exp: experiment.Experiment, args) -> None:
    """ Trains feature prediction and waveform generation models. """
    # TODO: pause/resume training
    if args["model"] in ("feature", "both"):
        logger.info("Loading feature model wrapper %s for training" %
                    (experiment.config["feature_model"]))
        wrappers.load(exp.config["feature_model"]).train(exp, args)
        logger.info("Training feature model done")
        # TODO: generate intermediate features if successful

    if args["model"] in ("wavegen", "both"):
        # TODO: check if intermediate features exist
        # TODO: train waveform gen model
        pass


def main():
    """ main function for training. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Experiment directory.')
    parser.add_argument('--model', default='both',
                        help='Which model to train: feature, wavegen, both. Default: both')
    args = parser.parse_args()

    try:
        exp = experiment.load(args.experiment_dir)
    except Exception:
        print("Invalid experiment folder given: %s" % (args.experiment_dir))
        sys.exit(1)

    train(exp, args)


if __name__ == '__main__':
    main()
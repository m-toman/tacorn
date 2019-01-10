import sys
import argparse
import logging

import tacorn.wrappers as wrappers
import tacorn.experiment as experiment

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _get_sentences(args):
    with open(args.sentences_file, 'rb') as f:
        sentences = [l.decode("utf-8").rstrip() for l in f.readlines()]
    return sentences


def synthesize(exp: experiment.Experiment, args) -> None:
    """ Synthesizes inside an experiment folder. """
    sentences = _get_sentences(args)
    logger.info("Loading acoustic feature model wrapper %s for synthesis" %
                (exp.config["acoustic_model"]))
    acoustic_module_wrapper = wrappers.load(exp.config["acoustic_model"])
    acoustic_module_wrapper.generate(exp, sentences, generate_features=args.use_wavegen,
                                     generate_waveforms=(not args.use_wavegen))
    logger.info("Synthesis from acoustic feature model done")

    if args.use_wavegen:
        wavegen_module_wrapper = wrappers.load(exp.config["wavegen_model"])
        wavegen_module_wrapper.generate(exp)


def main():
    """ main function for synthesis. """
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Experiment directory.')
    parser.add_argument('sentences_file',
                        help='File containing sentences to synthesize')
    parser.add_argument('--use_wavegen', default='True',
                        help='Use the wavegen model for waveform generation, if False resort to e.g. Griffin-Lim')
    args = parser.parse_args()

    try:
        exp = experiment.load(args.experiment_dir)
    except Exception:
        print("Invalid experiment folder given: %s" % (args.experiment_dir))
        sys.exit(1)

    synthesize(exp, args)


if __name__ == '__main__':
    main()

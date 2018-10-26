import argparse
import os
import torch
import librosa
import sys
from collections import namedtuple

import matplotlib
matplotlib.use('Agg')

from wavernn.model import Model
from wavernn.model import bits
from wavernn.utils import *
import config
import tacorn.fileutils as fu


sys.path.append('tacotron2')
from tacotron2.synthesize import prepare_run
from tacotron.synthesize import tacotron_synthesize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def synthesize(sentences, output_dir):
    # Tacotron first
    args = namedtuple(
        "tacoargs", "mode model checkpoint output_dir mels_dir hparams name".split())
    args.mode = "eval"
    args.model = "Tacotron-2"
    args.checkpoint = "pretrained/"
    args.output_dir = "output"
    args.mels_dir = "tacotron_output/eval"
    args.base_dir = ''
    args.input_dir = 'training_data/'
    args.hparams = ''
    args.name = "Tacotron-2"
    taco_checkpoint, _, hparams = prepare_run(args)
    taco_checkpoint = os.path.join("tacotron2", taco_checkpoint)
    tacotron_synthesize(args, hparams, taco_checkpoint, sentences)

    # now WaveRNN
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model = Model(rnn_dims=512, fc_dims=512, bits=bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    print("Loading WaveRNN model from " + MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))


    # mels_paths = [f for f in sorted(
    #     os.listdir(args.mels_dir)) if f.endswith(".npy")]
    map_path = os.path.join(args.mels_dir, 'map.txt')
    f = open(map_path)
    maps = f.readlines()
    mels_paths = [x.split('|')[1] for x in maps]
    f.close()
    test_mels = [np.load(m).T for m in mels_paths]


    fu.ensure_dir(output_dir)

    for i, mel in enumerate(test_mels):
        print('\nGenerating: %i/%i' % (i+1, len(test_mels)))
        model.generate(mel, output_dir + f'/{i}_generated.wav')

def main():
    # TODO: use custom workdir directory etc. instead of
    # original tacotron and wavernn paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences_file', default=None,
                        help='Input file containing sentences to synthesize')
    parser.add_argument('--output_dir', default="synthesized_wavs",
                        help='Output folder for synthesized wavs')
    args = parser.parse_args()
    isSentenceFile = False

    if args.sentences_file is None:
        sentences = ["Hello, World!",
                     "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]

    else:
        with open(args.sentences_file, 'rt') as fp:
            sentences = [x.strip() for x in fp.readlines()]
        isSentenceFile = True

    synthesize(sentences, args.output_dir)


if __name__ == '__main__':
    main()

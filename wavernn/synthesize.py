"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hyperparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import glob
import pickle
import time

import librosa
from docopt import docopt

from .model import *
from .hyperparams import hyperparams
from .utils import num_params_count


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]
    mel_file_name = args['<mel_input.npy>']
    mel = np.load(mel_file_name)
    # Override hyper parameters
    # hyperparams.parse(args["--hyperparams"])
    main(mel, checkpoint_path, output_path, preset, no_cuda)


def main(input_features, checkpoint_path, output_path, preset_path=None, no_cuda=False):
    device = torch.device('cpu' if
                          not torch.cuda.is_available() or no_cuda else 'cuda')
    print("using device:{}".format(device))

    # Load preset if specified
    if preset_path:
        with open(preset_path) as f:
            hyperparams.parse_json(f.read())

    if os.path.isdir(checkpoint_path):
        flist = glob.glob(f'{checkpoint_dir}/checkpoint_*.pth')
        latest_checkpoint = max(flist, key=os.path.getctime)
    else:
        latest_checkpoint = checkpoint_path

    print('Loading: %s' % latest_checkpoint)

    # build model, create optimizer
    model = build_model().to(device)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    print("I: %.3f million" % (num_params_count(model.I)))
    print("Upsample: %.3f million" % (num_params_count(model.upsample)))
    print("rnn1: %.3f million" % (num_params_count(model.rnn1)))
    print("rnn2: %.3f million" % (num_params_count(model.rnn2)))
    print("fc1: %.3f million" % (num_params_count(model.fc1)))
    print("fc2: %.3f million" % (num_params_count(model.fc2)))
    print("fc3: %.3f million" % (num_params_count(model.fc3)))

    #mel = np.pad(mel,(24000,0),'constant')
    # n_mels = mel.shape[1]
    # n_mels = hyperparams.batch_size_gen * (n_mels // hyperparams.batch_size_gen)
    # mel = mel[:, 0:n_mels]

    mel = input_features
    mel_file_name = output_path
    output_path_noext = os.path.splitext(output_path)[0]
    mel0 = mel.copy()
    start = time.time()
    output0 = model.generate(mel0, batched=False, target=2000, overlap=64)
    total_time = time.time() - start
    frag_time = len(output0) / hyperparams.sample_rate
    print("Generation time: {}. Sound time: {}, ratio: {}".format(
        total_time, frag_time, frag_time/total_time))

    # librosa.output.write_wav(output_path_noext+'_orig.wav',
    #                         output0, hyperparams.sample_rate)

    #mel = mel.reshape([mel.shape[0], hyperparams.batch_size_gen, -1]).swapaxes(0,1)
    #output, out1 = model.batch_generate(mel)
    #bootstrap_len = hp.hop_size * hp.resnet_pad
    # output=output[:,bootstrap_len:].reshape(-1)
    output = output0.astype(np.float32)
    librosa.output.write_wav(output_path_noext+".wav",
                             output0, hyperparams.sample_rate)
    # with open(os.path.join(output_path, os.path.basename(mel_file_name)+'.pkl'), 'wb') as f:
    #    pickle.dump((output0,), f)
    print('done')

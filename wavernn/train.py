# coding: utf-8

import os
import matplotlib.pyplot as plt
import math
import pickle
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .utils import *
from .dsp import *
from .model import Model
from .model import bits

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}mel/{file}.npy')
        x = np.load(f'{self.path}quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def collate(batch):
    pad = 2
    mel_win = seq_len // hop_length + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win]
            for i, x in enumerate(batch)]

    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1]
              for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)

    x_input = 2 * coarse[:, :seq_len].float() / (2**bits - 1.) - 1.

    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse


def _generate(model, step, test_ids, samples=3):
    outputs = []
    k = step // 1000
    test_mels = [np.load(f'{DATA_PATH}mel/{id}.npy')
                 for id in test_ids[:samples]]
    ground_truth = [np.load(f'{DATA_PATH}quant/{id}.npy')
                    for id in test_ids[:samples]]
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.
        librosa.output.write_wav(
            f'{GEN_PATH}{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
        outputs.append(model.generate(
            mel, f'{GEN_PATH}{k}k_steps_{i}_generated.wav'))
    return outputs


def _train(model, dataset, test_ids, optimiser, epochs, batch_size, classes, seq_len, step, lr=1e-4):
    loss_threshold = 4.0

    for p in optimiser.param_groups:
        p['lr'] = lr
    criterion = nn.NLLLoss().to(device)

    for e in range(epochs):
        trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)

        running_loss = 0.
        val_loss = 0.
        start = time.time()
        running_loss = 0.

        iters = len(trn_loader)

        for i, (x, m, y) in enumerate(trn_loader):
            x, m, y = x.to(device), m.to(device), y.to(device)

            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            speed = (i + 1) / (time.time() - start)
            avg_loss = running_loss / (i + 1)

            step += 1
            k = step // 1000
            print('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- Speed: %.2f steps/sec -- Step: %ik ' %
                  (e + 1, epochs, i + 1, iters, avg_loss, speed, k))

        # generate sample
        if e % 20 == 0:
            _generate(model=model, step=step, test_ids=test_ids, samples=1)

        torch.save(model.state_dict(), MODEL_PATH)
        np.save(STEP_PATH, [step])
        print(' <saved>')


def prepare_workdir(cfg):
    if not os.path.exists(model_checkpoints):
        os.makedirs(model_checkpoints)

    if not os.path.exists(GEN_PATH):
        os.makedirs(GEN_PATH)

def train(cfg):
    prepare_workdir(cfg)

    # load dataset
    with open(f'{DATA_PATH}dataset_ids.pkl', 'rb') as f:
        dataset_ids = pickle.load(f)

    test_ids = dataset_ids[-50:]
    dataset_ids = dataset_ids[:-50]
    dataset = AudiobookDataset(dataset_ids, DATA_PATH)
    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=32,
                             num_workers=0, shuffle=True)
    logger.info("Dataset length: " + str(len(dataset)))

    x, m, y = next(iter(data_loader))
    logger.debug("x.shape: %s, m.shape: %s, y.shape: %s" % (str(x.shape), str(m.shape), str(y.shape))

    model = Model(hidden_size=896, quantisation=256)

    # Load existing weights
    if not os.path.exists(MODEL_PATH):
        logger.debug("Creating new initial model")
        torch.save(model.state_dict(), MODEL_PATH)
    logger.debug("Loading model from " + MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))

    step = 0
    if not os.path.exists(STEP_PATH):
        np.save(STEP_PATH, [step])
    step = np.load(STEP_PATH)[0]
    logger.info("Starting from step: " + str(step))

    optimiser = optim.Adam(model.parameters())
    _train(model, dataset, test_ids, optimiser, epochs=1000, batch_size=16, classes=2**bits,
           seq_len=seq_len, step=step, lr=1e-4)


if __name__ == '__main__':
    train(None)

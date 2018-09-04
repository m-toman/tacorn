# coding: utf-8

# Conversion of nb5b notebook from https://github.com/fatchord/WaveRNN

# ## Alternative Model (Training)
#
# I've found WaveRNN quite slow to train so here's an alternative that utilises the optimised rnn kernels in Pytorch. The model below is much much faster to train, it will converge in 48hrs when training on 22.5kHz samples (or 24hrs using 16kHz samples) on a single GTX1080. It also works quite well with predicted GTA features.
#
# The model is simply two residual GRUs in sequence and then three dense layers with a 512 softmax output. This is supplemented with an upsampling network.
#
# Since the Pytorch rnn kernels are 'closed', the options for conditioning sites are greatly reduced. Here's the strategy I went with given that restriction:
#
# 1 - Upsampling: Nearest neighbour upsampling followed by 2d convolutions with 'horizontal' kernels to interpolate. Split up into two or three layers depending on the stft hop length.
#
# 2 - A 1d resnet with a 5 wide conv input and 1x1 res blocks. Not sure if this is necessary, but the thinking behind it is: the upsampled features give a local view of the conditioning - why not supplement that with a much wider view of conditioning features, including a peek at the future. One thing to note is that the resnet layers are computed only once and in parallel, so it shouldn't slow down training/generation much.
#
# There's a good chance this model needs regularisation since it overfits a little, so for now train it to ~500k steps for best results.

# In[1]:


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



def _generate(step, samples=3):
    outputs = []
    k = step // 1000
    test_mels = [np.load(f'{DATA_PATH}mel/{id}.npy')
                 for id in test_ids[:samples]]
    ground_truth = [np.load(f'{DATA_PATH}quant/{id}.npy')
                    for id in test_ids[:samples]]
    print("test_ids: " + str(test_ids[:samples]))
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.
        librosa.output.write_wav(
            f'{GEN_PATH}{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
        outputs.append(model.generate(
            mel, f'{GEN_PATH}{k}k_steps_{i}_generated.wav'))
    return outputs


def _train(model, dataset, optimiser, epochs, batch_size, classes, seq_len, step, lr=1e-4):

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
            _generate(step=step, samples=1)

        torch.save(model.state_dict(), MODEL_PATH)
        np.save(STEP_PATH, [step])
        print(' <saved>')


def train(cfg):
    if not os.path.exists(model_checkpoints):
        os.makedirs(model_checkpoints)

    if not os.path.exists(GEN_PATH):
        os.makedirs(GEN_PATH)

    with open(f'{DATA_PATH}dataset_ids.pkl', 'rb') as f:
        dataset_ids = pickle.load(f)

    test_ids = dataset_ids[-50:]
    dataset_ids = dataset_ids[:-50]

    dataset = AudiobookDataset(dataset_ids, DATA_PATH)

    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=32,
                             num_workers=0, shuffle=True)

    print("len: " + str(len(dataset)))

    x, m, y = next(iter(data_loader))
    x.shape, m.shape, y.shape

    # plot(x.numpy()[0])
    # plot(y.numpy()[0])
    # plot_spec(m.numpy()[0])

    model = Model(rnn_dims=512, fc_dims=512, bits=bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    if not os.path.exists(MODEL_PATH):
        print("Storing empty model")
        torch.save(model.state_dict(), MODEL_PATH)
    print("Loading model from " + MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH))

    mels, aux = model.preview_upsampling(m.to(device))

    # plot_spec(m[0].numpy())
    # plot_spec(mels[0].cpu().detach().numpy().T)
    # plot_spec(aux[0].cpu().detach().numpy().T)

    step = 0
    if not os.path.exists(STEP_PATH):
        np.save(STEP_PATH, [step])
    step = np.load(STEP_PATH)[0]
    print("starting from step: " + str(step))

    optimiser = optim.Adam(model.parameters())

    _train(model, dataset, optimiser, epochs=1000, batch_size=16, classes=2**bits,
           seq_len=seq_len, step=step, lr=1e-4)

    # ## Generate Samples

    # generate()
    # plot(output)
    # print(step)


if __name__ == '__main__':
    train(None)

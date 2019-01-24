import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .hyperparams import hyperparams as hp
from .distributions import *
from .utils import num_params, mulaw_quantize, inv_mulaw_quantize

#TODO: clean this mess
#TODO: remove all "cuda" calls and replace with device

class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims):
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims,
                                compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size,
                             padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks, device):
        super().__init__()
        if hp.input_type == 'raw':
            self.n_classes = 2
        elif hp.input_type == 'mixture':
            # mixture requires multiple of 3, default at 10 component mixture, i.e 3 x 10 = 30
            self.n_classes = 30
        elif hp.input_type == 'mulaw':
            self.n_classes = hp.mulaw_quantize_channels
        elif hp.input_type == 'bits':
            self.n_classes = 2**bits
        else:
            raise ValueError("input_type: {hp.input_type} not supported")
            
        self.device = device
        self.rnn_dims = rnn_dims
        # TODO: perhaps drop fc_dims parameter in constructor
        self.fc_dims = self.n_classes
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims,
                                        res_blocks, res_out_dims, pad)
        self.rnn1 = nn.GRU(feat_dims + self.n_classes, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims, self.fc_dims)
        self.fc2 = nn.Linear(self.fc_dims, self.n_classes)
        num_params(self)

    def forward(self, x, mels):
        batch_size = x.shape[0]
        n_classes = self.n_classes
        batch_len = x.shape[1]

        # one hot encode x
        netinput = torch.Tensor(batch_size, batch_len, n_classes).to(self.device)
        for b in range(batch_size):
            for l in range(batch_len):
                netinput[b, l, x[b, l]] = 1.0
        
        #h1 = torch.zeros(1, batch_size, self.rnn_dims).to(self.device)
        # upsample mels to fit audio input
        mels, aux = self.upsample(mels)

        # concat audio and mels, dimensions: (batch, time, audio-one-hot + mels)
        x = torch.cat([netinput, mels], dim=2)
        x, _ = self.rnn1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        #if hp.input_type == 'raw':
        #    return x
        #elif hp.input_type == 'mixture':
        #    return x
        #elif hp.input_type == 'bits' or hp.input_type == 'mulaw':
        #    return F.log_softmax(x, dim=-1)
        #else:
        #    raise ValueError("input_type: {hp.input_type} not supported")

    def preview_upsampling(self, mels):
        mels, aux = self.upsample(mels)
        return mels, aux

    def generate(self, mels) :
        self.eval()
        output = []
        rnn1cell = self.get_gru_cell(self.rnn1)
        #rnn2 = self.get_gru_cell(self.rnn2)
        
        print("mels to gen: " + str(mels.shape))

        with torch.no_grad() :
            x = torch.zeros(1, self.n_classes).to(self.device)
            h1 = torch.zeros(1, self.rnn_dims).to(self.device)
   
            # add batch dimension
            mels = torch.FloatTensor(mels).to(self.device).unsqueeze(0)
            mels, aux = self.upsample(mels)
            print("mels upsampled: " + str(mels.shape))
   
            seq_len = mels.size(1)
   
            for i in tqdm(range(seq_len)) :
                m_t = mels[:, i, :]

                print("mels at time t: " + str(m_t.shape))
                print("x input shape pre-cat: " + str(x.shape))
                
                x = torch.cat([x, m_t], dim=1)

                print("x input shape: " + str(x.shape))

                x, h1 = rnn1cell(x, h1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                print("x: " + str(x))

                if hp.input_type == 'raw':
                    if hp.distribution == 'beta':
                        sample = sample_from_beta_dist(x.unsqueeze(0))
                    elif hp.distribution == 'gaussian':
                        sample = sample_from_gaussian(x.unsqueeze(0))
                elif hp.input_type == 'mixture':
                    sample = sample_from_discretized_mix_logistic(x.unsqueeze(-1),hp.log_scale_min)
                elif hp.input_type == 'bits':
                    posterior = F.softmax(x, dim=1).view(-1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                elif hp.input_type == 'mulaw':
                    posterior = F.softmax(x, dim=1).view(-1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = inv_mulaw_quantize(distrib.sample(), hp.mulaw_quantize_channels, True)
                output.append(sample.view(-1))
                #x = torch.FloatTensor([[sample]]).cuda()
        output = torch.stack(output).cpu().numpy()
        self.train()
        return output

    def pad_tensor(self, x, pad, side='both'):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c).cuda()
        if side == 'before' or side == 'both':
            padded[:, pad:pad+t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):
        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len))

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len))

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def generate_overlapping(self, mels, target=11000, overlap=550, batched=True):

        self.eval()
        output = []

        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            # add a fake batch dimension and padding
            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            mels = self.pad_tensor(mels.transpose(
                1, 2), pad=hp.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                #target = mels.shape[1] // hp.batch_size_gen
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()
            x = torch.zeros(b_size, 1).cuda()

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = \
                    (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)
                posterior = F.softmax(logits, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                output.append(sample)
                x = sample.unsqueeze(-1)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        self.train()
        return output

    def batch_generate(self, mels):
        """mel should be of shape [batch_size x 80 x mel_length]
        """
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        b_size = mels.shape[0]
        assert len(
            mels.shape) == 3, "mels should have shape [batch_size x 80 x mel_length]"

        with torch.no_grad():
            x = torch.zeros(b_size, 1).cuda()
            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()

            mels = torch.FloatTensor(mels).cuda()
            mels, aux = self.upsample(mels)

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

            seq_len = mels.size(1)

            for i in tqdm(range(seq_len)):

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                if hp.input_type == 'raw':
                    sample = sample_from_beta_dist(x.unsqueeze(0))
                elif hp.input_type == 'mixture':
                    sample = sample_from_discretized_mix_logistic(
                        x.unsqueeze(-1), hp.log_scale_min)
                elif hp.input_type == 'bits':
                    posterior = F.softmax(x, dim=1).view(b_size, -1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                elif hp.input_type == 'mulaw':
                    posterior = F.softmax(x, dim=1).view(b_size, -1)
                    distrib = torch.distributions.Categorical(posterior)
                    print(type(distrib.sample()))
                    sample = inv_mulaw_quantize(
                        distrib.sample(), hp.mulaw_quantize_channels, True)
                output.append(sample.view(-1))
                x = sample.view(b_size, 1)
        output = torch.stack(output).cpu().numpy()
        self.train()
        # output is a batch of wav segments of shape [batch_size x seq_len]
        # will need to merge into one wav of size [batch_size * seq_len]
        assert output.shape[1] == b_size
        output = (output.swapaxes(1, 0)).reshape(-1)
        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


def build_model():
    """build model with hyperparams settings

    """
    if hp.input_type == 'raw':
        print('building model with Beta distribution output')
    elif hp.input_type == 'mixture':
        print("building model with mixture of logistic output")
    elif hp.input_type == 'bits':
        print("building model with quantized bit audio")
    elif hp.input_type == 'mulaw':
        print("building model with quantized mulaw encoding")
    else:
        raise ValueError('input_type provided not supported')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
                  hp.pad, hp.upsample_factors, hp.num_mels,
                  hp.compute_dims, hp.res_out_dims, hp.res_blocks, device)

    return model


def no_test_build_model():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
                  hp.pad, hp.upsample_factors, hp.num_mels,
                  hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))


def test_batch_generate():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
                  hp.pad, hp.upsample_factors, hp.num_mels,
                  hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))
    batch_mel = torch.rand(3, 80, 100)
    output = model.batch_generate(batch_mel)
    print(output.shape)

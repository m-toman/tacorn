import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.filters

from .hyperparams import hyperparams

# r9r9 preprocessing
import lws


def load_wav(path):
    if path.endswith(".npy"):
        return np.load(path)
    else:
        return librosa.load(path, sr=hyperparams.sample_rate)[0]


def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hyperparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hyperparams.preemphasis)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, hyperparams.preemphasis)


def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - hyperparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) +
                   hyperparams.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hyperparams.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hyperparams.ref_level_db
    if not hyperparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hyperparams.min_level_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hyperparams.win_length, hyperparams.hop_size, fftsize=hyperparams.fft_size, mode="speech")


# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if hyperparams.fmax is not None:
        assert hyperparams.fmax <= hyperparams.sample_rate // 2
    return librosa.filters.mel(hyperparams.sample_rate, hyperparams.fft_size,
                               fmin=hyperparams.fmin, fmax=hyperparams.fmax,
                               n_mels=hyperparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hyperparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hyperparams.min_level_db) / -hyperparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hyperparams.min_level_db) + hyperparams.min_level_db


# Fatcord's preprocessing
def quantize(x):
    """quantize audio signal

    """
    quant = (x + 1.) * (2**hyperparams.bits - 1) / 2
    return quant.astype(np.int)


# testing
def test_everything():
    wav = np.random.randn(12000,)
    mel = melspectrogram(wav)
    spec = spectrogram(wav)
    quant = quantize(wav)
    print(wav.shape, mel.shape, spec.shape, quant.shape)
    print(quant.max(), quant.min(), mel.max(),
          mel.min(), spec.max(), spec.min())

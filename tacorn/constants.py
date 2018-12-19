""" Holds constants required by tacorn.
    This mostly means paths into the Tacotron 2 and WaveRNN repositories. """

PRETRAINED_ACOUSTIC_MODELS = {
    "lj": {"tacotron2": "https://www.dropbox.com/s/kx7ephpl2oj72s3/taco_pretrained.zip?dl=1"}
}

PRETRAINED_WAVEGEN_MODELS = {
    "lj": {"wavernn_alt", "https://www.dropbox.com/s/o3v4obxdizliwgq/LJ_pretrained.zip?dl=1"}
}

TACOTRON2_DIR = "tacotron2"
WAVERNNALT_DIR = "wavernn_alt"

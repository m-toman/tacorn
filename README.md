# tacorn

WARNING1: The pre-trained models are (yet again) not compatible with the latest version of Rayhane-mamah's Tacotron2 repository.
WARNING2: This is experimental, messy and will most likely not be developed further.

This repository combines the Tacotron-2 implementation of Rayhane-mamah (https://github.com/Rayhane-mamah/Tacotron-2) with the WaveRNN-inspired (but heavily diverged) method by fatchord (https://github.com/fatchord/WaveRNN).

## Samples

- German: https://www.dropbox.com/s/6b90kuj5ce3mogr/de_1_generated.wav?dl=0
- English: https://www.dropbox.com/s/szrkknthxoj3znl/en_1_generated.wav?dl=0

## Synthesis

If you just want to synthesize from the pre-trained (English, LJ) models, currently you just have to run

```
bash install.sh
```
then:
```
python synthesize.py
```

Usage:
```
usage: synthesize.py [-h] [--sentences_file SENTENCES_FILE]
                     [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --sentences_file SENTENCES_FILE
                        Input file containing sentences to synthesize
  --output_dir OUTPUT_DIR
                        Output folder for synthesized wavs
```

Please note that the install.sh script pulling a pre-trained model is just a temporary solution and will be changed with future version.

## Training

install.sh grabs pre-trained models from the LJ dataset (https://keithito.com/LJ-Speech-Dataset/), so you don't necessarily have to do this step.

To continue training on the LJ dataset, or start from scratch:
```
bash install.sh
cd tacotron
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bunzip LJSpeech-1.1.tar.bz2
tar xf LJSpeech-1.1.tar
python3 preprocess.py
python3 train.py
# you can stop after synthesis of the GTA mels, once it's in wavenet training
cd ..
bash preprocess.sh
bash train.sh
```

If you're happy with the Tacotron output before it finished by itself, you can also interrupt the training and do:
```
cat 1|0|0| > logs-Tacotron-2/state_log
python3 train.py 
```

## Pre-trained models

- LJ dataset, Tacotron, 80k steps: https://www.dropbox.com/sh/z3nnetvyrsq9cip/AABXTGSl-P3dXJDIt6JpS8Eia?dl=0
(extract in ./Tacotron-2)
- LJ dataset, WaveRNN, 720k steps: https://www.dropbox.com/sh/ruq9elymhh9cyjl/AAD8u_PefFz_qwiAckqwqGzwa?dl=0
(extract in ./model_checkpoints)

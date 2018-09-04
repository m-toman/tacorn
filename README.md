# tacorn
Tacotron + WaveRNN synthesis

WARNING: Work in progress
This is extremely messy as I gradually integrate the jupyter notebooks from https://github.com/fatchord/WaveRNN into the codebase.

Makes use of:
 - https://github.com/fatchord/WaveRNN
 - https://github.com/Rayhane-mamah/Tacotron-2

 So the requirements from those have to be fulfilled until we have a merged requirements.txt.

 You'll at least need python3, PyTorch 0.4.1, Tensorflow >= 1.9.0 and librosa.

## Synthesis

If you just want to synthesize from the pre-trained models, currently you just have to run

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

At the moment, install.sh grabs pre-trained models from the LJ dataset (https://keithito.com/LJ-Speech-Dataset/), so you don't necessarily have to do this step.
Next iterations of this repository will give the option to download a pre-trained model or start from scratch.
Also, the training process will be controlled by ./train.py instead of manually calling Tacotron and WaveRNN training scripts

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

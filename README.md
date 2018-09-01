# tacorn
Tacotron + WaveRNN synthesis

Work in progress

Makes use of:
 - https://github.com/fatchord/WaveRNN
 - https://github.com/Rayhane-mamah/Tacotron-2

## Training

Current steps to get it to train, this will be automated with the next commits:
```
bash install.sh
cd Tacotron-2
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

If you're happy with the Tacotron output earlier, you can also do
```
cat 1|0|0| > logs-Tacotron-2/state_log
python3 train.py 
```

## Pre-trained models

- LJ dataset, Tacotron, 80k steps: https://www.dropbox.com/sh/z3nnetvyrsq9cip/AABXTGSl-P3dXJDIt6JpS8Eia?dl=0
(extract in tacorn/Tacotron-2)
- LJ dataset, WaveRNN, 720k steps: https://www.dropbox.com/sh/ruq9elymhh9cyjl/AAD8u_PefFz_qwiAckqwqGzwa?dl=0
(extract in tacorn/model_checkpoints)

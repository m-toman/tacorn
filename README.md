# tacorn
Tacotron + WaveRNN synthesis

This is work in progress and there are no results yet.

Makes use of:
 - https://github.com/fatchord/WaveRNN
 - https://github.com/Rayhane-mamah/Tacotron-2


Current steps to get it to train, this will be automatated in the next steps of the project:
```
bash install.sh
cd Tacotron-2
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bunzip LJSpeech-1.1.tar.bz2
tar xf LJSpeech-1.1.tar
python3 preprocess.py
python3 train.py
# stop at some point when you're happy with the Tacotron output
cat 1|0|0| > logs-Tacotron-2/state_log
python3 train.py 
# stop after synthesis
cd ..
bash preprocess.sh
bash train.sh
```

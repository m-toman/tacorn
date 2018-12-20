# tacorn

WARNING: Currently only supports Griffin-Lim synthesis from the Tacotron-2 repository.

TTS framework bridging different 2018/2019 state-of-the-art open source methods.
Currently aims to combine the Tacotron-2 implementation by Rayhane-mamah (https://github.com/Rayhane-mamah/Tacotron-2) with a fork of the alternative WaveRNN implementation by fatchord (https://github.com/fatchord/WaveRNN).
The overall goal is to more easily allow swapping out single components.

## Introduction

Speech synthesis systems consist of multiple components which have traditionally been developed manually and are increasingly being replaced by machine learning models.

Here we define three components used in statistical parametric speech synthesis. We don't consider unit selection or hybrid unit selection systems or physical modeling based systems.

Data flows along those components, producing intermediate representations which are then input to the next component. While in training we typically deal with large datasets and intermediate representations are typically stored on hard disk, we want to avoid this at synthesis time and aim to hold everything in memory.

### Text analysis
Component to generate a linguistic specification from text input.

Traditionally this involves hand-coded language specific rules, a pronuncation dictionary, letter-to-sound (or grapheme-to-phoneme) model for out of dictionary words and potentially additional models, e.g. ToBI endtone prediction, part of speech tagging, phrasing prediction etc.
The result for a given input sentence is a sequence of linguistic specifications, for example encoded as HTK labels. This specification typically at least holds a sequence of phones (or phonemes) but typically also includes contextual information like surrounding phones, interpunctuation, counts for segments, syllables, words, phrases etc. (see for example <https://github.com/MattShannon/HTS-demo_CMU-ARCTIC-SLT-STRAIGHT-AR-decision-tree/blob/master/data/lab_format.pdf>).
Examples for actual systems to perform text analysis are Festival, Flite or Ossian (REFs).

Recent systems take a step towards end-to-end synthesis and aim to replace those often complex codebases by machine learning models. 
Here we focus on Tacotron (REF).

### Acoustic feature prediction
Component consuming a linguistic specification to predict some sort of intermediate acoustic representation.

Intermediate acoustic representations are used because of useful properties for modeling but also because they are typically using a lower time resolution than the raw waveforms. Almost all commonly used representations employ a Fourier transformation, so for example with a commonly used window shift of 5ms we end up with only 200 feature vectors per second instead of 48000 for 48kHz speech. 
Examples for commonly used features include Mel-Frequency Cepstral Coefficents (MFCCs) and Line Spectral Pairs (LSPs).
Furthermore, additional features like fundamental frequency (F0) or aperiodicity features are commonly used.

The acoustic feature prediction component traditionally often employed a separate duration model to predict the number of acoustic features to be generated for each segment (i.e. phone), then an acoustic model to predict the actual acoustic features. Here we focus on Tacotron, which employs an attention-baed sequence to sequence model to merge duration and acoustic feature prediction into a single model.

### Waveform generation
Component generating waveforms from acoustic features.

The component performing this operation is often called a Vocoder and traditionally involves signal processing to encode and decode speech. Examples for Vocoders are STRAIGHT, WORLD, hts_engine, GlottHMM, GlottDNN or Vocaine.

Recently neural vocoders were employed with good success and include WaveNet, WaveRNN, WaveGlow, FFTNet and SampleRNN (REFs).
The main disadvantage of neural vocoders is that they are yet another model that has to be trained, typically even per speaker. This now only means additional computing resources and time required but also complicates deployment and requires additional hyperparameter tuning for this model. Possibilities to work around this include multi-speaker models or speaker-independent modeling (<https://arxiv.org/abs/1811.06292>).

Here we focus on WaveRNN although the currently included Tacotron-2 implementation by Rayhane-mamah also includes WaveNet.



## Experiment folder contents

- `config`: holds configurations for the experiment and the components.
- `raw`: input corpus.
- `raw/wavs`: input waveforms.
- `raw/meta`: input meta information, typically at least a transcription.
- `features`: holds intermediate representations used in training and synthesis.
- `features/acoustic`: holds preprocessed features for acoustic model training, e.g. mel spectrum, linguistic specifications.
- `features/acoustic2wavegen`: holds output features from acoustic used as input to wavegen.
- `features/acoustic2wavegen/training`: holds output features from acoustic used as input to wavegen training (e.g. ground-truth-aligned mel spectra).
- `features/acoustic2wavegen/synthesis`: holds output features from acoustic used as input to wavegen synthesis (e.g. mel spectra).
- `features/wavegen`: holds input features for the waveform generation model training, e.g. mel spectrum and raw waveforms.
- `models`: working directories for models/components.
- `models/acoustic`: working directory for the acoustic feature prediction component.
- `models/wavegen`: working directory for the waveform generation component.
- `synthesized`: synthesized wavefiles and metainformation.


## Process

### Create

* Input: Configuration parameters
* Output: Configured experiment directory
* Invocation: create.py

Creates a new experiment directory.


### Preprocessing

* Input: corpus in `raw` or given by parameter
* Output: processed features in `features` or `acoustic_model`
* Invocation: preprocess.py

Preprocessing waveforms and orthographic transcription.


### Training

* Input: processed features 
* Output: trained models in `acoustic_model` and `wavegen_model`
* Invocation: train.py

Train feature prediction and neural vocoder models.

### Synthesis

* Input: text, trained models in `acoustic_model`and `wavegen_model`
* Output: wavefiles in `synthesized_wavs`
* Invocation: synthesis.py


### Export

TODO

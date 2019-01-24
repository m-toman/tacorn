import tensorflow as tf

# Default hyperparameters:
hyperparams = tf.contrib.training.HParams(
    name="WaveRNN",
    num_workers=8,
    # Input type:
    # 1. raw [-1, 1]
    # 2. mixture [-1, 1]
    # 3. bits [0, 512]
    # 4. mulaw[0, mulaw_quantize_channels]
    #
    input_type='bits',
    #
    # distribution type, currently supports only 'beta' and 'mixture'
    distribution='gaussian',  # or "mixture"
    log_scale_min=-32.23619130191664,  # = float(np.log(1e-7))
    # quantize channel used for compute loss for mixture of logistics
    quantize_channels=65536,
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits=10,
    # for mu-law
    mulaw_quantize_channels=512,
    # note: r9r9's deepvoice3 preprocessing is used instead of Fatcord's original.
    # --------------
    # audio processing parameters
    num_mels=80,
    fmin=125,
    fmax=7600,
    # fft_size=1024,
    fft_size=1100,
    # hop_size=256,
    hop_size=275,
    # win_length=1024,
    win_length=2048,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    rescaling=False,
    rescaling_max=0.999,
    allow_clipping_in_normalization=True,
    # ----------------
    #
    # ----------------
    # model parameters
    rnn_dims=896,
    fc_dims=512,
    pad=2,
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 4 x 16 = 256)
    #upsample_factors=(4, 4, 16),
    upsample_factors=(5, 5, 11),
    compute_dims=128,
    res_out_dims=128,
    res_blocks=10,
    # ----------------
    #
    # ----------------
    # training parameters
    batch_size=32,
    nepochs=5000,
    save_every_step=30,
    evaluate_every_step=30,
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor=5,

    grad_norm=10,
    # learning rate parameters
    initial_learning_rate=1e-3,
    lr_schedule_type='noam',  # or 'noam'

    # for step learning rate schedule
    step_gamma=0.5,
    lr_step_interval=15000,

    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    weight_decay=1e-4,
    # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    fix_learning_rate=None,
    # -----------------
    batch_size_gen=32,
)

hyperparams.seq_len = hyperparams.seq_len_factor * hyperparams.hop_size

# for noam learning rate schedule
hyperparams.noam_warm_up_steps = 2000 * (hyperparams.batch_size // 16)

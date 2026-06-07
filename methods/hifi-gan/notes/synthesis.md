# HiFi-GAN synthesis (notes-first)

PROBLEM: stage-2 vocoder: mel-spectrogram -> 22.05kHz raw waveform, fast AND high fidelity.
Prior: WaveNet (AR, slow, 1 sample/forward), Parallel WaveNet (IAF distill teacher), WaveGlow (flow, >90 layers, heavy), MelGAN (GAN, MSD, real-time CPU but quality gap), Parallel WaveGAN (multi-res STFT loss), GAN-TTS (RWD ensemble of window disc). GAN quality lagged AR/flow.

KEY INSIGHT: speech = sum of sinusoids of various PERIODS. A 1D conv over the flat waveform mixes all periods; to see periodic structure cleanly, RESHAPE 1D audio length T into 2D (T/p, p) and convolve with kernels of width 1 along the period axis so each periodic phase is processed independently. Use prime periods [2,3,5,7,11] to minimize overlap. => Multi-Period Discriminator (MPD). Reshape (not subsampling) so gradients reach all timesteps.
Plus MSD from MelGAN (raw, x2, x4 avg-pooled) for consecutive/long-term patterns; first MSD sub uses spectral norm, rest weight norm.

GENERATOR: transposed-conv upsampling to waveform rate. After each upsample, Multi-Receptive Field Fusion (MRF): sum outputs of |k_r| residual blocks of different kernel sizes/dilations, then divide by num kernels (average). ResBlock1: 2 conv layers x3, dilations (1,3,5)/(1,1,1). conv_pre Conv1d(80,h,7), conv_post Conv1d(ch,1,7)+tanh. leaky relu slope 0.1, weight norm.
V1: h=512, k_u=[16,16,4,4], k_r=[3,7,11], D_r=[[1,1],[3,1],[5,1]]x3.

LOSSES (LSGAN, not BCE — non-vanishing gradients):
  L_adv(D;G) = E[(D(x)-1)^2 + (D(G(s)))^2]
  L_adv(G;D) = E[(D(G(s))-1)^2]
  Mel loss: L_mel = E[ ||phi(x)-phi(G(s))||_1 ]  (L1 on mel)
  Feature matching: L_fm = E[ sum_i 1/N_i ||D^i(x)-D^i(G(s))||_1 ]
  L_G = sum_k [L_adv(G;D_k) + lambda_fm L_fm(G;D_k)] + lambda_mel L_mel
  L_D = sum_k L_adv(D_k;G)
  lambda_fm=2, lambda_mel=45.
Train: LJSpeech 22kHz, 80-mel, FFT/win/hop=1024/1024/256. AdamW b1=0.8 b2=0.99 wd=0.01, lr=2e-4, decay 0.999/epoch.

CODE GROUNDING: jik876/hifi-gan models.py (verbatim above).

SCAFFOLD (pre-method): generic GAN vocoder harness. mel transform, a Generator stub (upsample mel->waveform, TODO architecture), a Discriminator stub (TODO), generic adversarial training loop, mel L1 reconstruction helper. NO MPD/MSD/MRF/period names.

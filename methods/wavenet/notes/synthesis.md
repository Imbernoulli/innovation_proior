# WaveNet synthesis (design-decision → why)

## Pain point / goal
Generate *raw* wideband audio (≥16 kHz) sample-by-sample as a tractable autoregressive
likelihood p(x)=Π_t p(x_t|x_{<t}). Audio has huge temporal resolution: 1s = 16000 samples,
and perceptually relevant structure spans tens of ms (phonemes) to seconds (prosody, music).
So the model needs a *very large receptive field* but must stay trainable.

## Tools on the table + where each breaks
- **Parametric/concatenative TTS**: vocoder params (LPC/cepstra/F0) every 5ms, then a model
  over those trajectories (HMM/DNN/LSTM). Two-step, sub-optimal; vocoder caps quality
  (muffled, buzzy). Strong priors: fixed analysis window, linear filter, Gaussian excitation —
  all violated by real speech. Concatenative = no model, glues recorded units; great segmental
  quality but inflexible, large footprint, joins audible.
- **Autoregressive neural models (PixelRNN/PixelCNN, neural LMs)**: factorize joint as product
  of conditionals, masked/causal convs to forbid future leakage, categorical softmax output.
  Proven on thousands of variables (64×64 images). PixelCNN-gated showed softmax beats mixtures,
  gating beats ReLU. → adopt the frame; question is whether it scales to 16k-sample sequences.
- **RNN audio models**: long memory in principle but training is inherently *sequential*
  (BPTT step-by-step), so on million-step sequences it's painfully slow; memory is also limited
  in practice (vanishing gradients beyond a few hundred steps).
- **Dilated (à trous) convolutions** (signal processing; Yu & Koltun 2016 seg): conv with holes,
  filter applied over a region larger than its length by skipping inputs; same output resolution
  as input, multi-scale context aggregation cheaply.

## Derivations to live out
1. **AR likelihood**: p(x)=Π_t p(x_t|x_1..x_{t-1}); train = maximize Σ log p; tractable NLL ⇒
   validation-set hyperparam tuning, over/underfit detection. Output = categorical softmax.
2. **Causal conv**: prediction at t must not see x_{t},x_{t+1},… For 1-D: shift a normal conv's
   output by (filter_width−1) steps (= left-pad input, drop the tail). No recurrence ⇒ train in
   parallel over all t; generation still sequential (feed sample back).
3. **Plain deep causal RF is linear**: stack of L causal layers, filter width k ⇒ RF = L(k−1)+1.
   For RF=1024 with k=2 you'd need ~1023 layers. Too deep.
4. **Dilated causal**: dilation d skips d−1 inputs. RF of a stack = (k−1)·Σ_layers d_ℓ + 1.
   Doubling d = 1,2,4,…,512 per block: Σ = 1+2+…+512 = 1023, so one block (10 layers, k=2) gives
   RF = 1·1023+1 = 1024 — **exponential RF in depth at linear cost**. Stack M such blocks:
   RF = (k−1)·M·Σ_block + 1 (e.g. 3 blocks → 3070 for the dilated stack; a width-2 initial
   causal input convolution makes the one-hot implementation 3071). Each 1,2,…,512 block
   ≈ a nonlinear 1×1024 conv but vastly cheaper.
5. **Gated activation**: z = tanh(W_f * x) ⊙ σ(W_g * x). The σ gate is a learned per-unit,
   data-dependent multiplicative mask on the tanh "content"; lets the net decide which features
   pass. Empirically >> ReLU for audio (PixelCNN found same for images).
6. **Residual + parameterized skip**: each block: gated unit → 1×1 conv → add to block input
   (residual, He 2015) for trainable depth; separately 1×1 conv → skip, all skips summed at the
   top → ReLU → 1×1 → ReLU → 1×1 → softmax. Skips give the output direct access to every depth's
   features (multi-scale), residual eases optimization of the deep stack.
7. **μ-law + 256-softmax**: 16-bit PCM ⇒ 65536-way softmax, intractable & wasteful. Companding
   f(x)=sign(x)·ln(1+μ|x|)/ln(1+μ), μ=255, maps [−1,1]→[−1,1] nonlinearly (log-like), then
   quantize to 256 levels. μ-law allocates more levels to small amplitudes (where ear is
   sensitive) ⇒ far better perceptual reconstruction than 8-bit *linear*; "sounded ~identical to
   original." Categorical softmax (no shape assumption) beats mixture-density / Gaussian.
8. **Conditioning**: p(x|h)=Π p(x_t|x_{<t},h).
   - Global: z = tanh(W_f*x + V_f^T h) ⊙ σ(W_g*x + V_g^T h), V^T h broadcast over time (speaker id).
   - Local: upsample h_t (lower rate, e.g. linguistic features) via transposed conv to y at audio
     rate, then z = tanh(W_f*x + V_f*y) ⊙ σ(W_g*x + V_g*y), V*y a 1×1 conv. (Repeat-upsample
     worked slightly worse.)
9. **Context stacks** (optional): a separate smaller stack over a long span locally-conditions the
   main stack on a short span; can pool/run slower — cheaper long-range context.

## Canonical impl (ibab/tensorflow-wavenet)
- calculate_receptive_field: `(filter_width - 1) * sum(dilations) + 1`, then add the initial layer term (`filter_width - 1` for one-hot input, `initial_filter_width - 1` for scalar input).
- dilation layer: causal_conv filter & gate → tanh*σ → 1×1 dense (residual add) + 1×1 skip.
- postprocess: sum skips → relu → 1×1 → relu → 1×1.
- mu_law_encode: sign(x)*log1p(mu*|x|)/log1p(mu), quantize to int in [0,mu]; one-hot input.
- dilations = [2**i for i in range(N)] * M (e.g. N=10 → 1..512, repeated M times).

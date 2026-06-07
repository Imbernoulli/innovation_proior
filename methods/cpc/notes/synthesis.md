# CPC / InfoNCE — synthesis notes

## Pain point
Want generic unsupervised representations from high-dim sequential data (audio, images-as-sequences, text, RL frames). Standard recipe: predict future/missing/context (predictive coding). But predicting raw future x_{t+k} requires a conditional generative model p(x_{t+k}|c_t) with a unimodal-or-powerful reconstruction loss (MSE/cross-entropy/autoregressive decoder). Wasteful: an image has thousands of bits but the shared/predictable latent (e.g. class = 10 bits for 1024 classes) is tiny. A generative model burns capacity on low-level local detail and noise, often ignoring c. Modeling p(x|c) is not the right objective for extracting what x and c SHARE.

## Core object
What we actually want is the information shared between context c_t and future x_{t+k}: the mutual information
  I(x;c) = sum p(x,c) log [ p(x|c)/p(x) ].
The integrand is the log density ratio p(x|c)/p(x). So if we can learn a model of the DENSITY RATIO (not the density itself), we directly capture MI and skip generative modeling.

## Architecture
- z_t = g_enc(x_t): non-linear encoder (strided conv on waveform), possibly lower temporal resolution.
- c_t = g_ar(z_{<=t}): autoregressive summary of the past (GRU).
- Score f_k(x_{t+k}, c_t) ∝ p(x_{t+k}|c_t)/p(x_{t+k}). Unnormalized (need not integrate to 1). Log-bilinear: f_k = exp(z_{t+k}^T W_k c_t), a separate W_k per step k. Encode the target x_{t+k} into z_{t+k}; never reconstruct x.

## InfoNCE loss
Set X = {x_1..x_N}: 1 positive from p(x_{t+k}|c_t), N-1 negatives from proposal p(x_{t+k}).
  L_N = -E_X [ log f_k(x_{t+k},c_t) / sum_{x_j in X} f_k(x_j,c_t) ].
= categorical cross-entropy of picking the positive slot; the softmax over scores is the classifier.

## Optimal critic (proof)
p(d=i | X, c) = prob slot i is the positive. Generative story: choose positive slot uniformly, positive ~ p(.|c), others ~ p(.). Likelihood slot i positive ∝ p(x_i|c) prod_{l≠i} p(x_l). Normalize:
  p(d=i|X,c) = [p(x_i|c) prod_{l≠i} p(x_l)] / sum_j [p(x_j|c) prod_{l≠j} p(x_l)].
Divide num+denom by prod_l p(x_l):
  = [p(x_i|c)/p(x_i)] / sum_j [p(x_j|c)/p(x_j)].
Softmax cross-entropy is minimized when model posterior = true posterior, so optimal f_k ∝ p(x|c)/p(x). Independent of N. QED.

## MI lower bound (appendix proof)
Insert optimal f (ratio r=p(x|c)/p(x)) into L_N, split positive vs negatives:
  L_N^opt = -E log[ r_pos / (r_pos + sum_{neg} r_j) ]
         = E log[ 1 + (1/r_pos) sum_{neg} r_j ]
         = E log[ 1 + (p(x_{t+k})/p(x_{t+k}|c)) sum_{neg} r_j ].
Negatives ~ p(x): E_{x_j ~ p}[r_j] = sum_x p(x) p(x|c)/p(x) = sum_x p(x|c) = 1. So sum_{neg} r_j ≈ (N-1).
  ≈ E log[ 1 + (p/p|c)(N-1) ]
  ≥ E log[ (p/p|c) N ]            (since 1+(N-1)a ≥ Na for a=p/p|c with the joint-sample regime a≤1)
  = E log[p(x)/p(x|c)] + log N
  = -I(x;c) + log N.
=> I(x_{t+k}; c_t) ≥ log N - L_N^opt. Holds for worse f too (higher L). Tighter as N grows; bound itself log N - L_N also grows with N. Approx (E sum_neg r_j = N-1) sharp at large N.

Note on the inequality step: 1+(N-1)a - Na = 1-a ≥ 0 iff a≤1. a = p(x)/p(x|c); under sampling x,c from the joint, p(x|c)≥p(x) typically so a≤1. Paper's commented LaTeX line confirms this reasoning. Present in-frame as: drop the +1 and bump (N-1)→N, valid because p(x|c)≥p(x) for jointly-drawn pairs.

## MINE relation (appendix)
f=e^F. E[log f/sum f] = E_{(x,c)}[F] - E[log(e^{F(x,c)} + sum_neg e^F)] ≤ E_{(x,c)}[F] - E_c[log sum_neg e^F] = E[F] - E_c[log (1/(N-1)) sum_neg e^F + log(N-1)] = MINE estimator up to const. So InfoNCE maximizes a lower bound on MINE/DV. MINE direct was unstable when target trivially predictable (1-step, overlap).

## Why contrastive beats generative
- Density ratio sidesteps modeling p(x) or p(x|c); only need samples + a score.
- Discards low-level noise; keeps only what discriminates future from random.
- Tractable via negative sampling (NCE / IS lineage): turn intractable normalization into a classification among samples.
- Multi-step prediction (k up to 12 audio / rows in image / sentences) forces slow/global features; 1-step exploits trivial smoothness.

## Ancestors (lineage)
- Predictive coding (Elias 1955, Atal-Schroeder 1970): predict next sample for compression — low-level, signal smoothness.
- Slow feature analysis (Wiskott-Sejnowski 2002): extract slowly-varying = global features.
- word2vec (Mikolov 2013): predict neighbor words with negative sampling — contrastive, low-dim, but discrete/local.
- NCE (Gutmann-Hyvärinen 2010): estimate unnormalized models by logistic discrimination of data vs noise; turns partition function into binary classification. mnih2012fast, jozefowicz2016 used it for LM.
- Importance sampling for NPLM (Bengio-Senecal 2008): approximate softmax denominator with samples from proposal.
- Triplet / metric losses (Chopra 2005, Weinberger 2009, FaceNet Schroff 2015): max-margin separate pos/neg — no MI/probabilistic grounding.
- TCN (Sermanet 2017), Time-Contrastive Learning + nonlinear ICA (Hyvärinen-Morioka 2016): contrastive over time for features.
- Generative/autoregressive predictors (WaveNet, PixelCNN/RNN, seq2seq, skip-thought): predict raw future — the wasteful baseline reacted against.
- MINE (Belghazi 2018): neural MI estimator (Donsker-Varadhan); InfoNCE is a stabler lower bound.

## Canonical code (audio CDCK2)
- encoder: 5 Conv1d, strides [5,4,2,2,2], k [10,8,4,4,4], 512 ch, downsample 160.
- gru: GRU(512,256). Wk: ModuleList of timestep Linear(256,512).
- forward: pick random t. z=enc(x).transpose. encode_samples[k]=z[:,t+k]. c_t=gru(z[:, :t+1])[:,t]. pred[k]=Wk[k](c_t). total = encode_samples[k] @ pred[k].T  (batch×batch); diagonal=positive logit, off-diag = in-batch negatives. nce += sum diag(log_softmax(total)). loss = -nce/(batch*timestep).
- IN-BATCH NEGATIVES: the N-1 negatives are the other batch elements' true futures — drawn from marginal p(x_{t+k}). N = batch size.
- evaluate: freeze, extract c_t, train linear classifier (phone/speaker).

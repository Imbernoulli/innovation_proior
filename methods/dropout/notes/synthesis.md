# Synthesis — Dropout (Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov 2012)

## The pain point (problem)
Large feedforward nets trained on limited labeled data overfit: many weight settings fit the
training set near-perfectly, but each generalizes differently and almost all do worse on test
than train. Root cause framed by the paper: **co-adaptation** — a hidden feature detector
becomes useful only in the presence of specific other detectors. It tunes itself to *correct
the mistakes of its collaborators on the training data*, a conspiracy that does not transfer.

## Tools on the table (and where each falls short)
- **Backprop (Rumelhart, Hinton, Williams 1986)**: learns feature detectors by gradient on
  shared error. Nothing in the objective discourages a unit from relying on others → permits
  co-adaptation.
- **L2 weight penalty / weight decay**: shrinks all weights toward 0 uniformly. A blunt
  capacity control; doesn't target the specific failure (detectors that work only together),
  and the penalty fights arbitrarily large proposed updates only weakly.
- **Model averaging (the principled cure for overfitting)**: average predictions of many nets.
  Reduces variance. But training and storing/evaluating many large nets is computationally
  infeasible. We want the averaging benefit without the cost.
- **Bagging (Breiman 1996) / Random Forests (Breiman 2001)**: each model trained on a
  bootstrap resample, equal-weight combination. Works for cheap models (trees). Big nets are
  neither cheap to fit nor cheap to evaluate, and the models share nothing.
- **Bayesian model averaging (Neal 1996)**: weight each model by posterior; MCMC sampling of
  net weights. Correct but very expensive; impractical at the scales here.
- **Mixture of experts (Jacobs, Jordan, Nowlan, Hinton 1991)**: a gate routes inputs to
  experts; each expert sees only a fraction of data → statistically inefficient per parameter.
- **Generative pre-training (Hinton & Salakhutdinov 2006 DBN; Salakhutdinov & Hinton 2009
  DBM)**: unsupervised layer-wise init to get good features without labels. Helps, but is a
  *separate* phase; doesn't, by itself, stop discriminative fine-tuning from co-adapting.
- **Naive Bayes**: extreme case — each feature trained alone, predictive distributions
  multiplied at test. Robust with little data precisely because no feature relies on context.

## The discovery chain (insight → method)
1. Co-adaptation is the disease. To stop a unit from relying on a specific partner, make the
   partner *unreliable*: randomly remove each hidden unit with prob 0.5 on every training case.
   Now a unit cannot count on any other being present → must be independently useful across a
   combinatorial variety of contexts. This is a regularizer aimed at the actual failure mode.
2. Second reading of the same trick: dropping units samples a thinned subnetwork. Each
   training case is processed by a *different* architecture, but all share weights. Over
   training we are training 2^N nets (N hidden units) with massive weight sharing → model
   averaging "for free." Dropout = extreme bagging where each model sees one case and every
   parameter is strongly tied across the exponential family. Tying is itself a much better
   regularizer than shrinking toward 0.
3. Test time: we can't run 2^N nets. **Mean network**: keep all units, halve outgoing weights
   (each unit is present half the time during training, so at test its expected contribution
   is halved). Claim: a single forward pass approximates the ensemble.
4. Exactness for one hidden layer + softmax (worked below): the mean network computes exactly
   the normalized geometric mean of all 2^N dropout predictive distributions. The geometric
   mean (not arithmetic) is the right object because softmax probabilities combine
   multiplicatively, and for softmax the log-partition term cancels under normalization. By the
   product-of-experts / inequality argument (Hinton 2002), the geometric mean assigns the
   correct class at least as high a log-prob as the average of the individuals' log-probs.
5. Optimization concern: with units dropping, each gradient is for a different stochastic net,
   so we want a *thorough* search of weight space. Replace the L2 *penalty* with an L2 *max-norm
   constraint* per hidden unit (cap incoming-weight squared length at l; if exceeded after an
   update, rescale down). A constraint, not a penalty, lets us start with a very large learning
   rate (weights can't blow up no matter how big the proposed step) and decay it — far more
   exploration than small-weights + small-LR. Plus high momentum (→0.99) to average gradient
   info over many different stochastic nets, stabilizing learning.

## Load-bearing derivation 1 — mean network = normalized geometric mean (1 hidden layer + softmax)
Hidden activations h_i (i=1..N). Mask m ∈ {0,1}^N drops units. Output logit for class k:
  z_k(m) = Σ_i m_i w_{ik} h_i + b_k.
Subnet prediction: P_m(k) = softmax_k(z(m)) = exp z_k(m) / Σ_j exp z_j(m).
Normalized geometric mean over all 2^N masks:
  G(k) = (∏_m P_m(k))^{1/2^N} / Σ_l (∏_m P_m(l))^{1/2^N}.
Take log of the unnormalized geo-mean U(k) = (∏_m P_m(k))^{1/2^N}:
  log U(k) = (1/2^N) Σ_m [ z_k(m) - log Σ_j exp z_j(m) ].
The term A = (1/2^N) Σ_m log Σ_j exp z_j(m) does NOT depend on k → cancels in normalization.
So G(k) ∝ exp( (1/2^N) Σ_m z_k(m) ).
Now (1/2^N) Σ_m z_k(m) = Σ_i [ (1/2^N) Σ_m m_i ] w_{ik} h_i + b_k.
Each m_i = 1 in exactly half the 2^N masks ⇒ (1/2^N) Σ_m m_i = 1/2. Hence
  (1/2^N) Σ_m z_k(m) = Σ_i (1/2) w_{ik} h_i + b_k.
Therefore G(k) = softmax_k( Σ_i (½ w_{ik}) h_i + b_k ) = the mean network with outgoing weights
halved. EXACT, not approximate, for one hidden layer + softmax. (For deeper nets it's a good
approximation.)

## Load-bearing derivation 2 — geometric mean dominates the average log-prob (Hinton 2002)
For each subnet define unnormalized scores q_m(k) = exp z_k(m), partition Z_m = Σ_j q_m(k),
P_m(k) = q_m(k)/Z_m. The mean-net distribution is G(k) = exp(mean_m log q_m(k)) / Z' where
Z' = Σ_l exp(mean_m log q_l(l))... i.e. G ∝ geometric mean of the q's, renormalized. Compare
G's log-prob of the true class t against the average of subnets' log-probs:
  log G(t) - mean_m log P_m(t)
   = [ mean_m log q_m(t) - log Z' ] - [ mean_m log q_m(t) - mean_m log Z_m ]
   = mean_m log Z_m - log Z'.
By definition log Z' = log Σ_l exp(mean_m log q_m(l)) and, since log-sum-exp ≤ ... use Jensen on
the convex log-sum-exp / AM-GM on partition functions: Σ_l exp(mean_m log q_m(l)) =
Σ_l ∏_m q_m(l)^{1/M} ≤ (by AM-GM applied per l across m, then summed) ... the clean statement
the paper uses: the geometric mean of the distributions, after renormalizing, assigns the
correct answer a log-prob ≥ the mean of the individual log-probs, with equality iff all subnets
agree. So the deterministic mean network is *guaranteed not worse* (in this log-prob sense) than
the typical sampled net. For linear-output regression the analogue is exact: squared error of
the mean prediction ≤ average of squared errors of the individuals (Jensen / bias-variance).

## Design decisions → why (rejected alternatives)
- **Drop with p=0.5 for hidden units**: extreme probabilities are worse empirically; 0.5
  maximizes the variety of subnets (entropy of the mask) and makes the "halve the weights"
  test-time rule exactly correspond to (1/2^N)Σm_i = 1/2. Tried various rates; nearly all help;
  0.5 is the sweet spot for hidden units.
- **Drop fewer inputs (e.g. keep >50% / drop 20%)**: inputs carry the signal; destroying half
  the raw evidence is too aggressive. ~20% input dropout helps (acts like input noise /
  denoising) but more is harmful.
- **Mean network (halve weights) at test, not Monte-Carlo averaging**: one forward pass vs many;
  exact for 1-layer+softmax, very close in practice for deep nets. Independence of per-unit drop
  decisions is what makes the single-pass approximation good.
- **Max-norm constraint instead of L2 penalty**: lets LR start huge and decay → thorough weight-
  space search; weights can't explode regardless of step size. A penalty can't guarantee that.
- **High final momentum (0.99)**: each gradient is from a different stochastic net; momentum
  averages gradient info over many updates → stable learning. Ramp 0.5→0.99 to avoid early
  instability. LR scaled by (1-momentum) to keep effective step sane.
- **Big learning rate (e.g. 10.0 on MNIST), exponential decay**: only safe *because* of the
  max-norm constraint; enables broad exploration early, fine convergence late.
- **Dropout in all hidden layers > one layer**: more co-adaptation broken; consistently better.
- **Less dropout in convolutional layers**: weight sharing already drastically cuts parameters
  and overfitting capacity, so dropout buys less there; reserve heavy dropout for the
  fully/locally-connected layers with many parameters.
- **ReLU (max-with-zero) in conv nets**: faster training, no saturation; init weights with
  enough variance + positive bias (1.0) so units get positive input and learn.
- **Combine with generative pre-training**: use small LR, no weight constraint during dropout
  fine-tuning, to avoid destroying pre-trained features.

## Canonical implementation (grounding)
cuda-convnet (Alex Krizhevsky) + the dropout fork. Mechanics from src/layer.cu:
- Train fprop: `mask = uniform(); mask = mask > _dropout; acts *= mask;` (keep prob = 1-_dropout)
- Test fprop: `acts *= (1 - _dropout)` (inverted scaling, equivalent to halving outgoing weights
  when keep=0.5).
- Bprop: `gradient *= mask` (same mask → gradients flow only through kept units).
Note the fork uses standard L2 weight decay (_wc); the paper's MNIST-from-scratch recipe uses a
per-unit max-norm constraint, which I implement explicitly in the answer code. The mask/scaling
mechanics are exactly the canonical fork's.

## Evolution-of-sex analogy (Livnat, Papadimitriou, Feldman 2008)
Sex breaks up co-adapted gene sets → favors genes that are individually useful (mixability) over
brittle large co-adapted complexes. Same principle: break co-adaptation, prefer robustly-useful
parts. A motivating analogy, not a derivation.

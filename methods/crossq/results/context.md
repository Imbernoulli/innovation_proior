# Context

## Research question

In off-policy actor-critic RL for continuous control, can we get the sample efficiency that
high update-to-data (UTD) methods reach — without the heavy machinery (large critic ensembles,
many gradient steps per environment step) and without the target networks that everyone treats as
a non-negotiable fixture? Concretely: a single critic update per environment step (UTD = 1), no
target networks at all, and accuracy that matches or beats methods doing ten or twenty critic
updates per step.

The dominant off-policy algorithms — DDPG, TD3, SAC — all bootstrap their Q-targets through a
*target network*: a slowly tracked copy of the critic that supplies a stationary regression target
so the bootstrap does not chase itself. The folklore is that without it deep value learning
diverges. The cost is a deliberately stale critic and a slow learning signal. The high-UTD line of
work (REDQ, DroQ) buys back accuracy by doing many critic updates per step against large ensembles,
which is computationally expensive. The question is whether the target network is actually
necessary, or whether there is a different stabilizer that removes it *and* the ensemble cost.

## Background

**Deterministic and stochastic off-policy actor-critics.** DDPG (Lillicrap et al. 2016) learns a
deterministic actor that ascends a learned Q-critic via the deterministic policy gradient, with a
replay buffer and Polyak-averaged target networks. TD3 (Fujimoto et al. 2018) adds twin critics
with a clipped-double-Q (min) target, delayed actor updates, and target-policy smoothing to fight
the overestimation that DDPG's actor-driven max induces. SAC (Haarnoja et al. 2018) replaces the
deterministic actor with a stochastic tanh-Gaussian one under a maximum-entropy objective, keeps
twin critics with a min target, and auto-tunes the entropy temperature. All three share the same
bootstrap skeleton: target network(s) supply the next-state value used in the Bellman target.

**The target network as a stationarity device.** Target networks come from DQN (Mnih et al. 2015):
the regression target `r + γ max_a Q(s',a)` is computed from a frozen copy `Q⁻` updated slowly, so
the critic fits a stationary objective over many gradient steps instead of one that moves every
step. Bootstrapping a network off itself otherwise compounds error. The stale copy is the price.

**Batch Normalization (Ioffe & Szegedy 2015) and Batch Renormalization (Ioffe 2017).** BatchNorm
normalizes a layer's pre-activations by the *batch's* mean/variance during training and by running
statistics at inference, which accelerates and stabilizes supervised training. It has a reputation
for *failing* in deep RL: naive insertion into value networks tends to destabilize training rather
than help. Batch Renormalization corrects BatchNorm's train/inference mismatch by introducing
clipped correction terms `r` and `d` that align the batch normalization with the running statistics,
making it more robust to small or non-i.i.d. batches — exactly the regime RL lives in.

**High-UTD efficiency without target networks.** REDQ (Chen et al. 2021) and DroQ (Hiraoka et al.
2022) reach high sample efficiency by raising the update-to-data ratio (many critic updates per
environment step) over a large randomized ensemble (REDQ) or with dropout + LayerNorm critics
(DroQ). They keep target networks. They are sample-efficient but computationally heavy.

The method this trace reconstructs — CrossQ (Bhatt, Palenicek et al., ICLR 2024) — argues that the
target network is not load-bearing once the critic is normalized correctly, and that the *right* way
to normalize is to pass the current and next state-action batches *jointly* through a
BatchRenorm critic so both are normalized under one shared distribution. The result removes target
networks entirely, keeps UTD = 1, and matches high-UTD accuracy with a few lines on top of SAC.

# PlaNet — synthesis (arXiv 1811.04551, verified)

## Pain point / goal
Control from raw 64x64x3 pixels, maximize E[sum_{t=1..T} r_t], with FAR fewer real env steps than model-free (A3C/D4PG need ~order more episodes). POMDP: hidden s_t, obs o_t, action a_t, reward r_t. Planning works great with KNOWN dynamics (MPC, AlphaGo, DeepStack) — so learn the dynamics and plan in it. Key difficulties of learned dynamics for planning: model inaccuracies, accumulating multi-step errors, failure to capture multiple futures, overconfident OOD predictions.

Why model-based planning over model-free: (1) data-efficient — richer training signal, no need to propagate reward through Bellman backups; (2) more compute -> better actions (AlphaGo); (3) task-independent dynamics transfer. PlaNet uses NO policy/value network — pure online planning (MPC).

## Lineage (load-bearing ancestors, all in refs)
- PILCO (Deisenroth & Rasmussen 2011): GP dynamics, analytic long-horizon gradient, remarkable sample efficiency BUT low-dim state + needs underlying state/reward fn. Gap: doesn't scale to pixels.
- E2C (Watter et al. 2015) / RCE (Banijamali et al. 2017): embed images to latent, local-linear latent transitions, plan with LQR. Balance cartpole, 2-link arm from dense rewards. Gap: Markov assumption on latent, hard to scale, only simple tasks/dense rewards.
- PETS (Chua et al. 2018): ensembles of NN dynamics + CEM/MPC, scales to cheetah BUT from low-dim state. PlaNet borrows CEM planner from here.
- Deep Kalman Filters (Krishnan et al. 2015): nonlinear SSM trained by VI (the ELBO machinery).
- VRNN (Chung et al. 2015): combine RNN+SSM via VI, but feeds generated obs back -> forward prediction expensive. PlaNet keeps it all latent.
- DVBF (Karl et al. 2016), PR-SSM (Doerr et al. 2018), DSSM (Buesing et al. 2018): latent SSMs; Buesing's is similar to RSSM but used in a hybrid agent, not explicit planning.
- VAE (Kingma & Welling 2014; Rezende et al. 2014): reparameterization + ELBO. beta-VAE (Higgins et al. 2016): the beta weights on KL — basis for beta_d in latent overshooting.
- World Models (Ha & Schmidhuber 2018): VAE + MDN-RNN + small controller; PlaNet reuses Ha's conv/deconv encoder-decoder architecture.
- Amos et al. 2018 (awareness): train deterministic model on all multi-step predictions (observation-space overshooting) — direct inspiration for latent overshooting.
- A3C (Mnih et al. 2016), D4PG: model-free yardsticks.
- CEM (Rubinstein 1997): population-based optimizer. MPC (Richards 2005).

## Method pieces & WHY (design-decision table)

1. **POMDP, filtering posterior, not smoothing.** Individual frames don't reveal full state -> POMDP. Use FILTERING posterior q(s_t | o_{<=t}, a_{<t}) (conditions only on PAST) rather than full smoothing because at plan time the model must predict without future observations. (Could use smoothing during training — they don't.)

2. **Latent dynamics, plan in latent space.** Need to evaluate THOUSANDS of action sequences per step. Predicting in pixel space too expensive. So learn transition p(s_t|s_{t-1},a_{t-1}), observation p(o_t|s_t), reward p(r_t|s_t). Observation model gives rich training signal but is NOT used at plan time (planner only needs reward, which is a fn of latent state -> no image generation). Transition: Gaussian, mean+var from feedforward net. Observation: Gaussian mean from deconv net, identity covariance. Reward: scalar Gaussian, unit variance. NOTE: log-likelihood under unit-variance Gaussian = MSE up to constant.

3. **Variational (ELBO) training.** Nonlinear -> can't compute posteriors in closed form. Encoder q(s_{1:T}|o,a)=prod_t q(s_t|s_{t-1},a_{t-1},o_t), diagonal Gaussian, conv+ff net. Bound (Jensen/importance weighting):
   ln p(o_{1:T}|a_{1:T}) >= sum_t [ E_{q(s_t|...)}[ln p(o_t|s_t)]  (reconstruction)
                                   - E_{q(s_{t-1}|...)}[ KL( q(s_t|o_{<=t},a_{<t}) || p(s_t|s_{t-1},a_{t-1}) ) ]  (complexity) ]
   Estimate outer expectations with ONE reparameterized sample. Optimize by grad ascent.
   DERIVATION (appendix): ln p(o_{1:T}|a) = ln E_{p(s)}[prod p(o_t|s_t)] = ln E_q[prod p(o_t|s_t)p(s_t|s_{t-1},a)/q(s_t|...)] >= (Jensen) E_q[sum ln p(o_t|s_t)+ln p(s_t|..)-ln q(s_t|..)] = sum_t(recon - KL). Importance weighting then Jensen.

4. **RSSM = deterministic + stochastic path.** Purely stochastic transition can't reliably REMEMBER over many steps (info must survive a sampled bottleneck each step; in theory could set variance->0 but optimizer won't find it). Purely deterministic RNN can't represent multiple futures (no stochastic state for KL to regularize). Split:
   h_t = f(h_{t-1}, s_{t-1}, a_{t-1})   [GRU, 200 units] — deterministic memory highway
   s_t ~ p(s_t | h_t)                     — stochastic prior
   o_t ~ p(o_t | h_t, s_t),  r_t ~ p(r_t | h_t, s_t)
   posterior q(s_t | h_t, o_t). Crucial: ALL obs info must pass through the sampling step of the encoder — no deterministic shortcut from inputs to reconstruction. Experiments show BOTH paths crucial.
   Features fed to decoders = concat([sample s_t, belief h_t]).

5. **CEM planner (MPC, replan each step).** Policy = planning. Chose CEM for robustness + it solves all tasks given TRUE dynamics. Time-dependent diagonal Gaussian belief over action seqs a_{t:t+H} ~ N(mu, sigma^2 I), start mu=0, sigma=1. Repeat I times: sample J candidates, evaluate (sample ONE state trajectory from current belief through prior, sum mean predicted rewards over horizon — single trajectory per seq since population optimizer), refit to top K. Return mu_t (first action mean). RESET belief to N(0,I) after each env step to avoid local optima. Horizon H=12, I=10, J=1000, K=100. Operate purely in latent space (reward is fn of latent) -> fast batch eval.
   Refit: mu = mean of top-K; sigma = (1/(K-1)) sum |a^(k) - mu|  (MEAN ABSOLUTE deviation, not std — note paper's formula).

6. **Experience collection.** Start with S=5 random seed episodes, train model, add 1 episode every C=100 update steps. Add Gaussian exploration noise eps~N(0,0.3) to actions. Action repeat R (cartpole 8, reacher 4, cheetah 4, finger 2, cup 4, walker 2) — to reduce planning horizon & give clearer learning signal.

7. **Latent overshooting (the generalization; final RSSM agent doesn't need it but it's a contribution).**
   Limitation of std ELBO: stochastic transition p(s_t|s_{t-1}) only trained via ONE-step KL; gradient never traverses a CHAIN of transitions. With limited capacity, training on 1-step preds != best multi-step preds. For planning we need accurate MULTI-step preds.
   - Multi-step prior: p(s_t|s_{t-d}) := integral prod_{tau=t-d+1..t} p(s_tau|s_{tau-1}) ds = E_{p(s_{t-1}|s_{t-d})}[p(s_t|s_{t-1})]. d=1 recovers one-step.
   - d-step bound (eq dstep): ln p_d(o_{1:T}) >= sum_t [ E_{q(s_t|o_<=t)}[ln p(o_t|s_t)]  -  E_{p(s_{t-1}|s_{t-d}) q(s_{t-d}|o_<=t-d)}[ KL( q(s_t|o_<=t) || p(s_t|s_{t-1}) ) ] ].  (Second >= by moving log inside multi-step prior via Jensen on the recursion.)
   - Latent overshooting (eq latov): average over d=1..D, with beta_d weights (beta-VAE style):
     (1/D) sum_d ln p_d >= sum_t [ recon - (1/D) sum_d beta_d E[ KL(q(s_t|o_<=t) || p(s_t|s_{t-1})) ] ].
     Regularizer purely in latent space (no extra images). STOP GRADIENT on posteriors for d>1 (multi-step preds trained toward informed posteriors, not vice versa). beta_{>1} set equal for simplicity.
   - Data-processing-inequality conjecture: latent chain Markov -> for d>=1, I(s_t;s_{t-d}) <= I(s_t;s_{t-1}) -> H(s_t)-H(s_t|s_{t-d}) <= H(s_t)-H(s_t|s_{t-1}) -> E[ln p(s_t|s_{t-d})] <= E[ln p(s_t|s_{t-1})] -> E[ln p_d(o_{1:T})] <= E[ln p(o_{1:T})]. So any bound on multi-step pred dist is also a bound on one-step (in expectation over data).

## Hyperparameters (appendix, verified)
- Conv/deconv from Ha & Schmidhuber (World Models). GRU 200 units. All other fns = 2 FC layers of 200, ReLU.
- Latent: 30-dim diagonal Gaussian (mean + softplus stddev + min_stddev 0.1).
- Images preprocessed to 5-bit depth (Glow-style).
- Adam, lr 1e-3, eps 1e-4, grad clip norm 1000. B=50 chunks of length L=50.
- KL not scaled vs reconstruction, but 3 FREE NATS: clip divergence below 3 (free-bits).
- CEM: H=12, I=10, J=1000, K=100. S=5 seed eps, collect every C=100, noise eps~N(0,0.3).

## Code grounding
- Official TF: google-research/planet/planet/models/rssm.py — _transition (prior) and _posterior. posterior calls transition with zero obs to get belief, concats [belief, obs], fc, mean + softplus stddev + min_stddev. features = concat([sample, belief]). GRUBlockCell(belief_size). num_layers FC of embed_size with elu (paper says ReLU; reimpl uses ReLU). 
- PyTorch reimpl cross32768/PlaNet_PyTorch: Encoder conv (3->32->64->128->256, k=4 s=2 -> 1024 embed), RecurrentStateSpaceModel (prior/posterior), ObservationModel deconv, RewardModel (300 hidden, 3 layers here — but PlaNet paper says reward=2 FC of 200; will follow paper for the FC sizing in answer, note reimpl uses ELU vs ReLU). Training loop: posterior rsample, KL clamp(min=free_nats), obs_loss 0.5*MSE, reward_loss 0.5*MSE, Adam clip_grad_norm. CEMAgent: posterior start state, Normal(0,1) action dist, sample N_candidates, open-loop prior rollout summing reward, top-k refit with mean & MAD/(K-1), return first action mean, replan.

## Code framework scaffold (pre-method)
Existing: env loop -> replay; conv encoder/deconv decoder (exist, from prior video-model work); GRU; reparameterized diagonal Gaussian; Adam+clip; KL divergence; CEM-style population optimizer primitive. MISSING slot: the latent transition (what carries memory + uncertainty), the training objective tying posterior to prior, and how the planner scores sequences in latent space.

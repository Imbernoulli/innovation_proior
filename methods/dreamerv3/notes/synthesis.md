# DreamerV3 synthesis (grounded in retrieved arXiv 2301.04104 src + danijar/dreamerv3 code)

## Pain point / research question
RL works but each new domain needs hyperparameter retuning (PPO robust but weak; SAC needs entropy tuning; MuZero complex). Goal: ONE algorithm, FIXED hyperparameters, masters >150 tasks across domains differing by orders of magnitude in reward scale, signal frequency, observation modality (pixels/vectors), action space (discrete/continuous), 2D/3D. The lineage (Dreamer V1/V2) already learns behavior in imagination from a world model — but V1/V2 required per-domain tuning (esp. the KL/representation-loss scale, reward scale, entropy scale). So the research question reduces to: what makes the previous Dreamer brittle across domains, and what robustness machinery removes each knob?

## Lineage (carried over, treated as background)
- Dyna: learn model / plan in model / act loop.
- PlaNet RSSM: deterministic recurrent h_t + stochastic z_t; sequence model h_t=f(h_{t-1},z_{t-1},a_{t-1}); prior p(z_t|h_t); posterior q(z_t|h_t,x_t). Carried over.
- Dreamer V1: actor-critic in imagination, lambda-returns, pathwise (analytic) actor gradient through dynamics. Continuous control only.
- Dreamer V2: discrete (categorical) latents with straight-through gradients; surpassed human Atari. KL balancing introduced.
- c51 / categorical return distribution (Bellemare 2017); twohot from MuZero (Schrittwieser 2019).
- Reinforce (Williams 1992), entropy reg (Williams 1991 maxent).

## The robustness techniques (THE contribution) — design-decision -> why
1. **symlog squared loss** for decoder(vector)/reward/value reads.
   - symlog(x)=sign(x)ln(|x|+1), symexp(x)=sign(x)(exp(|x|)-1). Inverse pair.
   - Why: targets vary by orders of magnitude across domains. Squared loss on raw large targets -> divergence. Abs/Huber -> stagnates (gradient constant). Normalizing by running stats -> non-stationarity. symlog compresses both signs, ~identity near 0, lets net move fast to large values without truncation/non-stationarity. Predict f(x)=symlog(y); read yhat=symexp(f).
   - Why not plain log: can't represent negatives. symlog is bi-symmetric log family, symmetric about origin, preserves sign.
2. **symexp twohot loss** for stochastic targets (reward predictor + critic).
   - Net outputs logits over exponentially spaced bins B=symexp([-20..20]) (255 bins in code). Read yhat=softmax(f)^T B. Train: -twohot(y)^T logsoftmax(f). twohot: a vector all 0 except indices k,k+1 of two nearest bins, weights = linear interpolation, sum 1.
   - Why: return dist can be multimodal + span orders of magnitude. Categorical decouples gradient size from target size (gradient depends only on bin probabilities, not bin values). Weighted-avg readout can land between bins -> continuous.
   - Output weights init to ZERO so agent doesn't hallucinate rewards/values at init (delays "large predicted reward" stall).
3. **KL balancing + free bits** for the world-model latent KL.
   - dyn loss = max(1, KL[sg(q(z|h,x)) || p(z|h)]); rep loss = max(1, KL[q(z|h,x) || sg(p(z|h))]). beta_dyn=1, beta_rep=0.1.
   - Why two terms (KL balance, from V2): want prior to move toward posterior FASTER than posterior toward prior (prior must chase the richer posterior; don't want posterior collapsing to a lazy prior). sg on different sides + different scale (1.0 vs 0.1).
   - Why free bits (clip at 1 nat ~ 1.44 bits): prevents degenerate solution where dynamics trivially predictable but states carry no info (posterior collapse). Disables KL once already small, focuses on prediction loss. This replaces the per-domain representation-loss scaling V1 needed (3D worlds need strong reg to drop irrelevant detail; static-bg games need weak reg to keep fine pixels). free bits + small rep scale resolves the dilemma at fixed hparams.
4. **1% unimix** on all categoricals (encoder, dynamics predictor, actor): dist = 0.99*softmax + 0.01*uniform. Why: prevents deterministic categoricals -> prevents KL spikes (seen in deep VAEs) and infinite logprobs / zero probs.
5. **Percentile return normalization** for the actor entropy trade-off.
   - S = EMA(Per(R^lambda,95) - Per(R^lambda,5), 0.99). Scale returns by 1/max(L,S) with L=1 (only scale DOWN large returns, leave small ones untouched). Fixed entropy eta=3e-4.
   - Why: entropy scale's right value depends on reward scale+frequency; want explore-more-when-sparse, exploit-when-dense, but invariant to arbitrary reward rescaling. Normalizing ADVANTAGES (PPO) puts fixed emphasis regardless of reachability -> amplifies noise under sparse rewards, stalls exploration. Normalizing by std fails under sparse (std~0 -> blows up). Percentile range robust to outliers (multimodal/randomized returns); EMA smooths; L=1 floor stops over-amplifying small returns. Subtracting offset doesn't change gradient -> only divide by range.
6. **Critic = categorical return distribution + EMA self-regularization.**
   - Critic predicts distribution of R^lambda (c51-style). Critic regularized toward EMA of own params (decay 0.98) instead of a frozen target net -> can compute returns with current critic. Also apply critic loss to REPLAY trajectories (beta_repval=0.3) using imagination returns at rollout start states as on-policy value annotations.
7. **Actor uses Reinforce for BOTH discrete and continuous** (V3 change — V1 used pathwise/dynamics-backprop for continuous). Surrogate: -sg((R^lambda - v)/max(1,S)) logpi - eta*H. Why Reinforce uniformly: a single estimator across discrete+continuous = fewer code paths, fixed hparams; works because the value baseline + return norm tame variance.
8. **Architecture**: RSSM with **Block GRU** (block-diagonal recurrent weights, 8 blocks) -> many memory units w/o quadratic params. **RMSNorm + SiLU**. CNN encoder stride-2 to 6x6/4x4, transposed conv decoder sigmoid out. Vector enc/dec 3-layer MLP on symlog inputs. Actor/critic 3-layer MLP; reward/continue 1-layer MLP.
9. **Optimizer**: AGC (clip per-tensor grad to 30% of weight L2 norm, eps 1e-3) decouples clip threshold from loss scale; LaProp (RMSProp then momentum, beta1=0.9,beta2=0.99,eps=1e-20) instead of Adam (momentum & normalizer both on raw grad) -> smaller eps, avoids Adam instabilities. lr 4e-5.
10. **Continue predictor** c_t in {0,1} (episode continuation), via logistic regression; used in lambda-return discount: R^lambda_t = r_t + gamma c_t ((1-lambda)v_t + lambda R^lambda_{t+1}), R^lambda_T = v_T. gamma=0.997 (discount horizon 333). H=16 (paper text; hparams table imagination horizon 15 — note: T=16 model states from H=15 steps). lambda=0.95.
11. Replay: uniform buffer + online queue, store latent states to init WM on replay, capacity 5e6. Replay ratio = train steps per env step.

## World model loss (eq:wm)
L(phi) = E_q[ sum_t beta_pred L_pred + beta_dyn L_dyn + beta_rep L_rep ]
L_pred = -ln p(x|z,h) - ln p(r|z,h) - ln p(c|z,h)
L_dyn  = max(1, KL[sg(q(z|h,x)) || p(z|h)])
L_rep  = max(1, KL[q(z|h,x) || sg(p(z|h))])
beta_pred=1, beta_dyn=1, beta_rep=0.1.

## Eval settings (pre-method, no outcomes)
Benchmarks (yardsticks): Atari57 (sticky actions), DMLab30 (fixed action space), ProcGen (hard, unlimited levels), Atari100k (26 tasks, 400k env / 100k after action-repeat 4), DMC Visual Control 20 tasks (action repeat 2), DMC Proprioceptive, BSuite, Minecraft Diamond (sparse, 12 milestones +1 each, 100M steps). Metric: episode return vs env steps; human-normalized for Atari. Baselines: PPO (Acme, IMPALA net), IMPALA, Rainbow, plus tuned expert algos per benchmark. 5 seeds.

## Code structure (for scaffold correspondence)
- RSSM: img_step (prior), obs_step (posterior), get_feat
- symlog/symexp, twohot, TwoHotDist (bins, .pred()=softmax^T B, .loss=cross-entropy), MSEDist symlog
- lambda_return backward scan
- Moments/Normalize: percentile EMA -> retnorm (for adv/return), valnorm
- imagine_ahead, imag_loss (policy = reinforce w/ return-norm advantage + entropy; value = twohot cross-entropy to sg(return), + slow critic reg), world-model loss
- AGC + LaProp opt

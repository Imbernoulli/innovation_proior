# Synthesis — Decoupled Weight Decay (AdamW / SGDW)

## The pain point (state of the field ~2017)
- Adaptive gradient methods (AdaGrad, RMSProp, Adam, AMSGrad) are the default for training feed-forward and recurrent nets — fast, robust to learning-rate choice, little tuning.
- BUT: state-of-the-art on CIFAR-10/100 and ImageNet is still held by **SGD with momentum**. Wilson et al. (2017) "The marginal value of adaptive gradient methods": across image classification, char-LM, constituency parsing, adaptive methods generalize WORSE than tuned SGD+momentum, even when they reach lower training loss.
- Hypotheses for the gap floating around: sharp vs flat minima (Keskar 2016), Dinh 2017 (sharp minima can generalize too), inherent problems of adaptive methods (Wilson). None pinned down a concrete, fixable mechanism.
- Practical pain: practitioners must pick optimizer + hyperparameters per task. Want Adam's ease-of-use AND SGD's generalization.

## Load-bearing ancestors

### Weight decay — Hanson & Pratt (1988)
Original definition: at each step shrink weights multiplicatively before/with the gradient step:
  θ_{t+1} = (1 − λ) θ_t − α ∇f_t(θ_t).
λ = decay rate per step. Motivation: minimal network construction, bias toward small weights.

### L2 regularization
Add penalty to the loss: f^reg(θ) = f(θ) + (λ'/2)‖θ‖². Gradient gets an extra λ'θ term:
  ∇f^reg = ∇f + λ'θ.
Then plug into whatever optimizer. This is what every DL library calls "weight decay".

### SGD (plain / with momentum)
θ_{t+1} = θ_t − α ∇f_t(θ_t). Momentum: m_t = μ m_{t-1} − α∇f_t; θ_{t+1}=θ_t+m_t (or β1 form).

### AdaGrad (Duchi 2011)
Per-parameter learning rate: divide gradient by sqrt of sum of past squared gradients. v_t = Σ g²; update θ -= α g / (√v_t + ε). Problem: v_t grows without bound → LR → 0.

### RMSProp (Tieleman & Hinton 2012)
Fix AdaGrad's vanishing LR with an EMA: v_t = β2 v_{t-1} + (1−β2) g²; θ -= α g/(√v_t + ε).

### Adam (Kingma & Ba 2014)
Combine momentum + RMSProp + bias correction:
  m_t = β1 m_{t-1} + (1−β1) g_t
  v_t = β2 v_{t-1} + (1−β2) g_t²
  m̂_t = m_t/(1−β1^t), v̂_t = v_t/(1−β2^t)   [bias correction because m,v init at 0]
  θ_t = θ_{t-1} − α m̂_t/(√v̂_t + ε)
Defaults: α=0.001, β1=0.9, β2=0.999, ε=1e-8.
Key structural fact: the update is PRECONDITIONED — each coordinate's effective step is divided by √v̂ (its historical RMS gradient magnitude). Write the preconditioner as a diagonal matrix M_t = diag(1/(√v̂+ε)); update is θ -= α M_t m̂. M_t ≠ kI in general.

### SGDR — Loshchilov & Hutter (2016)
Cosine annealing of the LR multiplier + warm restarts:
  η_t = η_min + 0.5(η_max − η_min)(1 + cos(π T_cur/T_i)).
Restart: when T_cur reaches T_i, reset T_cur=0 (η jumps back up), keep current θ, multiply budget by T_mult. Improves anytime performance, gives state-of-the-art when combined with Shake-Shake etc.

### Aitchison (2018) — Bayesian filtering view
Posted after the preliminary arXiv version; gives a principled justification. Treats optimizing θ_i given θ_{-i} as a Bayesian filtering / tracking problem. Gaussian state-transition prior + approx conjugate likelihood → posterior mean update μ_post = μ_prior + Σ_post g, i.e. the preconditioner IS the posterior covariance (more uncertain → bigger step). Adam/RMSProp/K-FAC are special cases. The state-transition prior P(θ_{t+1}|θ_t) = N((I−A)θ_t, Q): A is a shrink-toward-zero regularizer. A = λI gives EXACTLY decoupled weight decay (multiply mean by (1−λ)), and it's applied to the PRIOR, not scaled by uncertainty — so weight decay, not L2, is what falls out of the framework naturally.

## The core derivation

### Prop 1: weight decay = L2 for plain SGD (with λ' = λ/α)
SGD on f^reg = f + (λ'/2)‖θ‖²:
  θ_{t+1} = θ_t − α∇f − αλ'θ_t.
SGD with decoupled decay on f:
  θ_{t+1} = (1−λ)θ_t − α∇f = θ_t − λθ_t − α∇f.
Identical iff αλ' = λ, i.e. λ' = λ/α. KEY: the equivalence requires λ' tied to α. So even for SGD, if you tune via the L2 coefficient λ', the best λ' moves whenever you change α — the two hyperparameters are COUPLED. (Explains SGD's reputation for LR sensitivity.)

### Prop 2: weight decay ≠ L2 for any non-trivial preconditioner
Optimizer O with preconditioner M_t. 
L2: θ_{t+1} = θ_t − α λ' M_t θ_t − α M_t ∇f_t   (because the λ'θ term passes through M_t too).
Decoupled WD: θ_{t+1} = (1−λ)θ_t − α M_t ∇f_t.
Equality for all θ_t ⇒ λ θ_t = α λ' M_t θ_t ⇒ M_t = (λ/(αλ'))I = kI. Contradiction unless M_t ∝ I. 
So for Adam (M_t = diag preconditioner ≠ kI) NO L2 coefficient reproduces decoupled weight decay. This is the crux.

### Why L2 fails specifically in Adam (intuition)
With L2, the regularizer gradient λθ is ADDED to g BEFORE the v_t accumulation and BEFORE the √v̂ division. So the decay term gets normalized by √v̂ too. A weight with historically large gradients (large √v̂) has its decay shrunk by the same large denominator → it is regularized LESS, exactly the weights you'd most want to rein in. Conversely small-gradient weights get hammered. To get meaningful decay on big-gradient weights you'd raise λ, but that destroys the small ones. Net: L2 in Adam can't deliver effective uniform regularization → that's a concrete cause of Adam's worse generalization.
With decoupled WD: every weight is multiplied by (1−λ), uniformly, untouched by √v̂.

### Prop 3: with a FIXED preconditioner, decoupled WD = scale-adjusted L2
Let M_t = diag(s)^{-1}, s_i>0 fixed. Claim: O w/ decoupled WD λ runs the same steps as O w/o WD on
  f^sreg(θ) = f(θ) + (λ'/2)‖θ ⊙ √s‖²,  with λ'=λ/α.
Proof. ∇f^sreg = ∇f + λ' (θ⊙s)  [since d/dθ of (λ'/2)Σ s_i θ_i² = λ' s_i θ_i].
O step = θ − α M_t ∇f^sreg = θ − α ∇f/s − α λ'(θ⊙s)/s = θ − α∇f/s − αλ'θ.   [the s cancels in the decay term!]
Decoupled WD step = (1−λ)θ − α∇f/s = θ − α∇f/s − λθ.
Equal iff αλ' = λ. ∎
Interpretation: decoupled WD ⇔ penalizing ‖θ⊙√s‖², i.e. coordinate i is regularized ∝ √s_i — MORE for weights with historically large gradients. (Caveat the paper states: doesn't directly apply to practical Adam since M_t changes each step; intuition only.)

## Additional components

### Normalized weight decay
Empirically the optimal λ depends on the total budget — longer runs (more updates) want smaller λ, since total shrink ≈ (1−λ)^{#updates}. #updates = (B/b)·T (T epochs, batch b, dataset B). Reparameterize:
  λ = λ_norm · √(b/(B·T)).
λ_norm interpretable as "the decay if only one batch pass is allowed." Makes the best setting transfer across budgets and datasets (CIFAR-10 ↔ ImageNet32x32, where epoch is ~24× longer; without normalization λ would be ~5× too large for ImageNet32x32). √ scaling is one choice informed by experiments; the lasting point is that SOME normalization helps.

### AdamWR
AdamW + cosine annealing (η_t per Eq with η_max=1, η_min=0: η_t = 0.5 + 0.5cos(π T_cur/T_i)) + warm restarts (reset T_cur=0 when it hits T_i, T_i ← T_i·T_mult). Normalized WD uses T = epochs in current restart, so a constant λ_norm works across restarts of different length. Restarts had earlier failed for Adam precisely because broken L2 hampered it; fixing decay makes SGDR carry over. Recommend solutions taken right before each restart (η_t=0).

## Design decisions → why
- **Multiply θ by (1−λ) directly, separate from gradient step** (the whole method): restores Hanson&Pratt's exponential decay so the preconditioner never touches the decay; decouples λ from α; uniform shrink. Alternative (add λθ to grad = L2) fails per Prop 2.
- **Schedule both α and λ by η_t**: once decoupled, λ is no longer auto-scheduled via the gradient (as L2 implicitly was), so to keep behavior consistent under a schedule you must apply η_t to the decay step too (Algo line 9/12).
- **Bias correction kept (inherited from Adam)**: m,v init at 0 → early estimates biased toward 0; divide by (1−β^t). Unchanged by decoupling.
- **ε inside the sqrt-denominator only (not added to decay)**: ε guards division; decay needs no ε since it's a plain multiply.
- **√(b/BT) normalization**: total decay ≈ exp(−λ·#updates), #updates ∝ BT/b; want total decay budget invariant → λ ∝ 1/#updates would over-correct, √ chosen empirically to transfer across budget & dataset.
- **Cosine + restarts (AdamWR)**: anytime performance; restarts only work once decay is fixed.

## Canonical implementation (PyTorch torch/optim/adamw.py, _single_tensor_adamw)
Core lines, per parameter:
  step += 1
  param.mul_(1 - lr * weight_decay)               # decoupled decay BEFORE the Adam step
  exp_avg.lerp_(grad, 1 - beta1)                   # m = β1 m + (1-β1) g
  exp_avg_sq.mul_(beta2).addcmul_(grad,grad,1-beta2)  # v = β2 v + (1-β2) g²
  bc1 = 1 - beta1**step; bc2 = 1 - beta2**step
  step_size = lr / bc1
  denom = (exp_avg_sq.sqrt() / sqrt(bc2)).add_(eps)
  param.addcdiv_(exp_avg, denom, value=-step_size)  # θ -= step_size * m / (√(v/bc2)+ε)
Note: grad here is the RAW loss gradient — no λθ added to it (that is the entire point vs torch Adam which has `grad = grad.add(param, alpha=weight_decay)`). PyTorch folds η_t into lr; in the original AdamW algorithm the decay is η_t·λ·θ. The decoupled-decay line and the gradient-step line are disjoint.

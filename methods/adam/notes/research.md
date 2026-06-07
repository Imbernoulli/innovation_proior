# Research notes — Adam (Kingma & Ba, ICLR 2015, arXiv 1412.6980)

Primary paper read in full by main agent: 15-page source PDF (src/0_adam_main.pdf),
including Sections 1-9 and Appendix 10 (full convergence proof: Def 10.1, Lemmas 10.2-10.4,
Theorem 10.5). Math verified against the paper directly.

## Load-bearing ancestors (verified)

### SGD with momentum / Nesterov (Sutskever et al. 2013, "On the importance of initialization and momentum")
- Classical momentum: v_t = mu v_{t-1} + grad ; theta -= lr v_t. EMA of gradient -> first moment.
- NAG: look-ahead gradient, more responsive at high momentum.
- Sutskever 2013: well-tuned momentum + good init can match Hessian-free; momentum schedule
  (increasing then reducing near end) helps. Adam cites this for "decaying beta_1 toward end can help".
- Gap left: a single global learning rate; no per-parameter adaptation; momentum on raw gradient,
  no rescaling by gradient magnitude/variance.

### AdaGrad (Duchi, Hazan, Singer, JMLR 2011, "Adaptive Subgradient Methods")
- Per-parameter rate: theta_{t+1} = theta_t - alpha * g_t / sqrt(sum_{i=1}^t g_i^2).
- Accumulates SUM of squared grads in denominator -> large steps for rare/sparse features,
  small for frequent. Great for sparse, high-dim (NLP) problems.
- Regret O(sqrt(T)); for sparse data adaptive methods can get O(log d sqrt(T)) vs O(sqrt(dT)).
- Online convex optimization / regret framing comes from here (and Zinkevich).
- Gap: monotonically growing denominator => learning rate decays to zero, kills learning on
  non-stationary / non-convex deep nets. Adam relates: AdaGrad = Adam with beta_1=0,
  infinitesimal (1-beta_2), and alpha replaced by annealed alpha*t^{-1/2}
  (gives lim v_hat = t^{-1} sum g^2). [verified p.5 of paper]

### RMSProp (Tieleman & Hinton, Coursera Lecture 6.5, 2012; momentum variant Graves 2013)
- Fix AdaGrad's decaying LR: use EMA of squared grad instead of sum:
  v_t = beta_2 v_{t-1} + (1-beta_2) g_t^2 ; theta -= alpha g_t / (sqrt(v_t)+eps).
- "Forgets" old gradients => works on non-stationary objectives.
- Momentum variant (Graves): momentum applied to the RESCALED gradient.
- Gaps (per Adam paper p.5): (a) no bias-correction term => with beta_2 near 1 (needed for sparse)
  the zero-init bias gives huge early steps and divergence; (b) RMSProp+momentum puts momentum on
  rescaled gradient, whereas Adam uses running averages of first AND second moment directly.

### AdaDelta (Zeiler, 2012, arXiv 1212.5701)
- Also EMA of squared grad; additionally tracks EMA of squared parameter updates to make
  the update unit-correct (no manual global LR). Same EMA-denominator family as RMSProp.
- Gap: still no moment/bias-correction view; unit-matching heuristic.

### Online convex optimization / regret (Zinkevich, ICML 2003)
- OCP framework: adversary reveals convex f_t, predict theta_t, regret
  R(T)=sum_t [f_t(theta_t)-f_t(theta*)]. Online gradient descent with eta_t=t^{-1/2} gives O(sqrt(T)).
- Convexity lower bound f_t(theta)-f_t(theta*) <= g_t^T(theta_t-theta*) is the entry point of
  Adam's regret proof (Lemma 10.2). Adam's bound R(T)=O(sqrt(T)), avg regret -> 0.

### Natural gradient / Fisher (Amari 1998; Pascanu & Bengio 2013)
- NGD preconditions by inverse Fisher information. Adam: v_hat approximates the diagonal of the
  Fisher; preconditioning by sqrt of that diagonal is MORE conservative than vanilla NGD.
- Roux & Fitzgibbon 2010 (fast natural Newton), SFO (Sohl-Dickstein 2014) = quasi-Newton on
  minibatches but memory linear in #minibatch partitions => infeasible on GPU. Motivates a
  cheap first-order method.

## State of the field at the time (2014/early 2015)
- Deep learning boom (AlexNet 2012, speech). SGD+momentum the workhorse but LR tuning painful.
- Adaptive per-parameter methods (AdaGrad/RMSProp/AdaDelta) popular but each with a known wart:
  AdaGrad LR decay-to-zero, RMSProp no bias correction, no unified moment view.
- Dropout (Hinton 2012b) and minibatching make gradients very noisy; sparse high-dim problems common.
- Second-order/quasi-Newton too memory-heavy for GPUs. Need: robust, low-memory, first-order,
  per-parameter, works on non-stationary + sparse + noisy objectives, little tuning.

## Bias-correction derivation (verified, eqs 1-4)
- v_t = (1-beta_2) sum_{i=1}^t beta_2^{t-i} g_i^2 (unroll EMA from v_0=0).
- E[v_t] = E[g_t^2]*(1-beta_2) sum beta_2^{t-i} + zeta = E[g_t^2](1-beta_2^t) + zeta.
- zeta=0 if second moment stationary, else small. So divide by (1-beta_2^t) to de-bias.
  Same for m_t with (1-beta_1^t). Matters most for small (1-beta_2) (sparse case).

## Effective step / SNR / trust region (Sec 2.1, verified)
- Delta_t = alpha * m_hat/sqrt(v_hat). Bounds: |Delta_t| <= alpha*(1-beta_1)/sqrt(1-beta_2)
  when (1-beta_1)>sqrt(1-beta_2), else |Delta_t|<=alpha. Establishes a trust region ~alpha.
- m_hat/sqrt(v_hat) ~ SNR; small SNR => smaller steps => automatic annealing near optimum.
- Scale-invariant: scaling g by c cancels (c*m)/sqrt(c^2*v)=m/sqrt(v).

## Regret proof (Appendix 10, verified)
- Lemma 10.2 convexity hyperplane bound. Lemma 10.3 induction: sum sqrt(g_{t,i}^2/t) <= 2 G_inf ||g_{1:T,i}||_2.
- Lemma 10.4: sum m_hat^2/sqrt(t v_hat) <= (2/(1-gamma)) (1/sqrt(1-beta_2)) ||g_{1:T,i}||_2,
  gamma = beta_1^2/sqrt(beta_2) < 1, uses arithmetic-geometric series sum t gamma^t < 1/(1-gamma)^2.
- Theorem 10.5: with alpha_t=alpha/sqrt(t), beta_{1,t}=beta_1 lambda^{t-1}, get
  R(T) <= D^2/(2alpha(1-beta_1)) sum sqrt(T v_hat_{T,i})
        + alpha(1+beta_1)G_inf/((1-beta_1)sqrt(1-beta_2)(1-gamma)^2) sum ||g_{1:T,i}||_2
        + sum D_inf^2 G_inf sqrt(1-beta_2)/(2 alpha beta_1 (1-lambda)^2).
  => R(T)=O(sqrt(T)); Corollary 4.2: R(T)/T = O(1/sqrt(T)) -> 0.

## AdaMax (Sec 7.1, verified, eqs 6-12)
- Generalize L2 to Lp on the second moment; let p->inf: u_t = max(beta_2 u_{t-1}, |g_t|).
- Update theta -= (alpha/(1-beta_1^t)) m_t / u_t. No bias correction needed for u (max, init 0 ok).
  |Delta_t| <= alpha.

## Temporal averaging (Sec 7.2)
- Polyak-Ruppert; EMA of parameters theta_bar = beta_2 theta_bar + (1-beta_2) theta, de-bias by (1-beta_2^t).

## Canonical code (code/)
- adam_pytorch_v0.4.py (PyTorch v0.4.1 torch/optim/adam.py): clean canonical step():
  exp_avg.mul_(beta1).add_(1-beta1, grad); exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad);
  bias_correction1=1-beta1**t; bias_correction2=1-beta2**t;
  step_size=lr*sqrt(bc2)/bc1; p.addcdiv_(-step_size, exp_avg, exp_avg_sq.sqrt().add_(eps)).
  Uses the "efficient" reordering from the paper (folds bias corrections into step_size).
  NOTE: contains amsgrad branch = POSTERIOR (2018, Reddi et al.) -> exclude from context/reasoning.
- adamax_pytorch.py, adam_pytorch.py (v2.0.0) also fetched for reference.

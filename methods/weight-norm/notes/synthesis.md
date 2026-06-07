# Synthesis — Weight Normalization

## Pain point / research question
First-order SGD success depends on the *curvature* (condition number of the Hessian) of
the loss. Pathological curvature → slow progress. Curvature is **not invariant to
reparameterization**: equivalent parameterizations of the same model can be far easier or
harder to optimize. So: find a *reparameterization* of the standard neuron
y = φ(w·x + b) that improves conditioning of the gradient, without the drawbacks of
batch-coupled normalization.

## Background / field state at the time
- Natural gradient: left-multiply gradient by approx inverse Fisher → whitened gradient.
  KFAC (Kronecker-factored Fisher, Martens & Grosse 2015), FANG (Cholesky of inverse
  Fisher, Grosse & Salakhutdinov 2015), PRONG (whiten layer inputs, Desjardins 2015).
  All expensive / need to estimate & invert curvature.
- Reparameterization route (cheaper): change params so plain SGD gradient ≈ natural
  gradient. Raiko et al. 2012: transform neuron outputs to zero mean & zero slope on
  average → approximately diagonalizes Fisher → whitens.
- BatchNorm (Ioffe & Szegedy 2015): normalize pre-activation t = v·x by minibatch mean and
  std: t' = (t − μ[t])/σ[t]. Reduces "covariate shift", suggested to bring Fisher closer to
  identity. Speeds up training, allows higher LR.
- max-norm / weight clipping (Srebro & Shraibman 2005): normalized the weight norm, BUT
  optimized in w and applied normalization *after* each SGD step — fundamentally different
  from reparameterizing and doing SGD in the new params.

## BatchNorm's drawbacks (the gap WN attacks)
1. Couples the examples in a minibatch — the output for one example depends on the others.
2. Adds noise to gradients (minibatch μ, σ are stochastic estimates). High variance for
   small minibatch.
3. Bad fit for RNN/LSTM (recurrent: normalizing cell states diminishes their ability to
   carry information across steps; same weights reused each timestep).
4. Bad for noise-sensitive apps: RL (DQN destabilizes), generative models.
5. Computational overhead (≈16% slower in their CIFAR setup), extra memory for stats.

## The method
Reparameterize each weight vector:
    w = (g / ‖v‖) · v,
with v a k-vector, g a scalar. Then ‖w‖ = g independent of v. SGD is done directly in
(v, g) (and bias b), NOT in w. Decouples magnitude (g) from direction (v/‖v‖).
Optional: g = e^s log-scale (tried, no benefit, slightly slower → not used).

## Gradients (DERIVE — self-checked)
Let r = ‖v‖.
- ∂w_i/∂g = v_i/r ⇒ ∇_g L = (∇_w L · v)/‖v‖.
- ∂w_i/∂v_j = g[δ_ij/r − v_i v_j/r³] (using ∂r/∂v_j = v_j/r).
  ⇒ ∇_v L = (g/‖v‖)∇_w L − (g/‖v‖³)(∇_w L·v) v
          = (g/‖v‖)∇_w L − (g ∇_g L/‖v‖²) v   [since ∇_w L·v = ‖v‖ ∇_g L].
- Projection form: ∇_v L = (g/‖v‖) M_w ∇_w L, with M_w = I − w w'/‖w‖²
  (projects onto complement of w; uses v/‖v‖ = w/‖w‖).
Two effects: (1) SCALE gradient by g/‖v‖; (2) PROJECT gradient orthogonal to w.
Both push gradient covariance toward identity.

## Geometry / self-stabilizing norm
- ∇_v L ⟂ v always (∇_v L · v = 0; verify: M_w v = 0 since v ∥ w). So a steepest-descent
  step Δv ∝ ∇_v L is orthogonal to v.
- ‖v'‖ = ‖v + Δv‖ = √(‖v‖² + c²‖v‖²) = √(1+c²)‖v‖ ≥ ‖v‖ with c = ‖Δv‖/‖v‖ (Pythagoras).
  ⇒ ‖v‖ grows monotonically (plain GD, no momentum). Growing ‖v‖ shrinks scale g/‖v‖ ⇒
  self-stabilizing effective LR. Noisy grad → large c → fast ‖v‖ growth → shrink scale.
  Small grad → c→0 → ‖v‖ stops. ⇒ robustness to LR choice. (Not strict for Adam/momentum
  but qualitatively holds.)
- Noise removal: cov of grad in v is D = (g²/‖v‖²) M_w C M_w where C = cov of ∇_w L. If w is
  ≈ dominant eigenvector of C, projecting it out makes D closer to identity.

## Relation to BatchNorm
For a single-layer net with whitened inputs x (zero mean, unit var, independent):
μ[t] = 0, σ[t] = ‖v‖ where t = v·x (since Var(v·x) = Σ v_i² Var(x_i) = ‖v‖²). Then
BN's t' = (t−μ)/σ = (v·x)/‖v‖ = (w·x)/g — i.e. BN ≡ WN in this special case. WN = cheaper,
deterministic, non-stochastic approximation to BN (CNNs have fewer weights than
pre-activations → normalizing weights cheaper).

## Data-dependent initialization
WN does NOT fix the *scale* of features per layer (BN does). So initialize:
- Sample v ~ N(0, 0.05²) elementwise.
- Feedforward ONE minibatch X. At each neuron compute t = (v·x)/‖v‖, with batch mean μ[t],
  std σ[t]. Set g ← 1/σ[t], b ← −μ[t]/σ[t]. Then y = φ((t−μ[t])/σ[t]) = φ(w·x + b) has
  zero-mean unit-var pre-activations on that minibatch. Only for the init minibatch.
- Same as concurrent LSUV (Mishkin & Matas 2015) / Krähenbühl 2015, but for the v,g param.
- Can't apply to RNN/LSTM → fall back to standard init there.

## Mean-only batch normalization
WN fixes scale (≈) but the *mean* of activations still depends on v. So combine with a BN
that subtracts mean only, no variance division:
    t = w·x,  t̃ = t − μ[t] + b,  y = φ(t̃),  μ[t] running-averaged for test.
Backprop: ∇_t L = ∇_{t̃} L − μ[∇_{t̃} L] — centers the backpropagated gradients. Cheap. Noise
added is approx Gaussian (CLT on μ) → light tails, vs full BN's high-kurtosis variance-est
noise. WN + mean-only BN best CIFAR result (regularizing noise + good conditioning).

## Canonical code grounding
- PyTorch torch.nn.utils.weight_norm: registers v ("weight_v"), g ("weight_g"=norm), and a
  forward_pre_hook recomputes w = g·v/‖v‖ (via _weight_norm) before each forward. norm taken
  over all dims except `dim` (output-channel dim).
- OpenAI Theano/Lasagne (openai/weightnorm): WeightNormLayer with V (init N(0,0.05)),
  g, b; init pass computes m=mean, stdv=sqrt(mean(square)) over axes_to_sum, sets
  init_updates = [(b, -m/stdv), (g, g/stdv)].
- Code framework scaffold (pre-method): a generic Linear/Conv module storing a raw weight
  Parameter + bias, an SGD/Adam optimizer, MSE/CE loss, a training loop, plus an empty
  "reparameterize the weight" hook stub and an empty "data-dependent init from one batch"
  stub. Final code fills exactly these.
```

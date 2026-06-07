# Lion — synthesis (grounded in arXiv 2302.06675 release.tex + lucidrains/lion-pytorch + google/automl/lion)

## Verified arXiv id: 2302.06675 (Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023)
Name: EvoLved Sign Momentum. Canonical impl: github.com/google/automl/tree/master/lion; widely-used clean reimpl lucidrains/lion-pytorch (downloaded to code/lion_pytorch.py).

## Pain point / research question
AdamW (Adam + decoupled weight decay) and Adafactor are de facto standard optimizers for SOTA language/vision/multimodal models, but they are handcrafted. Adam keeps TWO extra buffers (m and v) -> 2x optimizer memory of params. Want: discover (not handcraft) a first-order optimizer that generalizes from small proxy tasks to SOTA-scale, ideally simpler / lower memory.

## Method discovery framing (program search)
Optimizers represented as PROGRAMS (imperative, NumPy/JAX-like) operating on n-d arrays. train(w, g, v1, v2, lr) -> update, v1, v2; outer loop w = w - update. Same I/O signature as AdamW so memory footprint <= AdamW (2 extra buffers max). 45 math funcs incl interp(x,y,a) := (1-a)*x + a*y. Regularized evolution (tournament selection), warm-start population = AdamW, restarts. Pruning: abstract execution -> drop syntax/shape errors, hash for functionally-equiv caching (~10x), redundant-statement removal (~70% redundant, ~3x shorter). Funnel selection on meta-validation (larger than proxy) to fight meta-overfitting.

## Initial program = AdamW (Program 1, eps & bias-correction omitted)
def train(w, g, m, v, lr):
  g2 = g*g
  m = interp(g, m, 0.9)       # m = 0.9*m + 0.1*g  (NOTE interp(x,y,a)=(1-a)x+ay so interp(g,m,0.9)=0.1*g+0.9*m)
  v = interp(g2, v, 0.999)
  sqrt_v = sqrt(v)
  update = m / sqrt_v
  wd = w * weight_decay
  update = update + wd
  update = update * lr
  return update, m, v

## Raw discovered program -> simplified (Program 2, "renamed for clarity")
def train(w, g, m, v, lr):
  g = clip(g, lr); g = arcsin(g)
  m = interp(g, v, 0.899)
  m2 = m*m
  v = interp(g, m, 1.109)
  abs_m = sqrt(m2)
  update = m / abs_m          # = m/|m| = sign(m)  (the three red statements = sign)
  wd = w*0.4602
  update = update + wd
  lr = lr*0.0002
  m = cosh(update)            # redundant: m reassigned next iter
  update = update*lr
  return update, m, v

## Simplification -> Lion (Program 0)
- cosh removed (m overwritten next iter).
- arcsin & clip removed (no quality drop).
- the 3 red statements (m2=m*m; abs_m=sqrt(m2); update=m/abs_m) collapse to update = sign(m).
- v and m: both used, but v ONLY changes how momentum updated. Two interps with constants ~0.9 and ~1.1 on (g, m, v) are equivalent to ONE interp with ~0.99 -> v need not be tracked separately. So momentum tracked by single EMA with beta2~0.99.

## Final Lion (Program 0) -- THE METHOD
def train(weight, gradient, momentum, lr):
  update = interp(gradient, momentum, beta1)   # = (1-beta1)*gradient + beta1*momentum
  update = sign(update)
  momentum = interp(gradient, momentum, beta2)  # = (1-beta2)*gradient + beta2*momentum
  update = update + weight_decay                # (gray) = update + w*lambda ... decoupled wd
  update = update * lr
  return update, momentum
Defaults beta1=0.9, beta2=0.99 (derived from search). Outer: w = w - update.

## Appendix pseudocode (exact, Algorithm Lion)
given beta1, beta2, lambda, eta, f
  c_t  <- beta1 * m_{t-1} + (1-beta1) g_t        # interpolated momentum used for the update
  theta_t <- theta_{t-1} - eta_t ( sign(c_t) + lambda theta_{t-1} )   # update + decoupled wd
  m_t  <- beta2 * m_{t-1} + (1-beta2) g_t        # momentum EMA (separate, slower)
AdamW for contrast: m_t<-beta1 m + (1-beta1)g; v_t<-beta2 v +(1-beta2)g^2; update m_t/(sqrt v_t)+eps; +decoupled wd.

## Analysis / why (Section: Derivation and Analysis)
- Sign => uniform magnitude update across ALL dimensions (element-wise +/-1 ignoring wd), unlike adaptive optimizers (per-coord scaling). Differs in principle from adaptive.
- Sign adds NOISE to the update -> acts as REGULARIZATION, helps generalization (cf. Neelakantan 2017 adding gradient noise; SAM Foret 2021). Flatter minima.
- Two distinct interpolations (beta1 for the update, beta2 for the momentum EMA) DECOUPLE momentum tracking from how momentum is applied. beta2=0.99 -> momentum remembers ~10x longer gradient history than 0.9; beta1=0.9 on (g,m) BEFORE sign puts MORE weight on current gradient in the actual update. Best of both: long memory in the buffer, recent emphasis in the step.
- Ablation: single interp `m=interp(g,m,beta); update=sign(m)` with beta=0.9 or 0.99 (Ablation_0.9, Ablation_0.99) both worse than Lion -> both betas necessary.
- Larger update norm: sign output is +/-1 elementwise -> ||update|| larger than SGD/adaptive -> need lr 3-10x SMALLER than AdamW. Effective weight decay = lr*lambda (since update += w*lambda then *= lr), so to keep same effective wd, lambda must be 3-10x LARGER than AdamW.
- Memory: only momentum m tracked (1 buffer) vs Adam's m+v (2 buffers). Simpler: no eps, no factorization hyperparams.

## Relation to existing optimizers
- signSGD (Bernstein 2018) and momentum variant: also sign the update, but DIFFERENT momentum rule (signSGD-momentum signs an EMA of g; Lion signs an interpolation of g and m, and tracks m with a different, slower beta2). Rprop (Riedmiller 1993) sign of gradient too.
- NAdam (Dozat 2016): combines updated first moment with the gradient for the update — Lion also combines g and m for the update (via beta1) but additionally DECOUPLES the momentum tracking (beta2) from the application.
- AutoML-discovered PowerSign/AddSign (Bello 2017): used RL/MC over restricted trees, couldn't modify how momentum is tracked/contributes -> didn't reach SOTA. Lion's search space lets momentum tracking itself be searched.

## Load-bearing ancestors
- Adam (Kingma 2014) / AdamW (Loshchilov 2019): EMA of g (m) and g^2 (v), update m/sqrt(v), decoupled weight decay (wd applied to weights not through the gradient/v). Lion's warm-start = AdamW; removing v is the key simplification.
- signSGD (Bernstein 2018): sign(g) update, communication-efficient; majority-vote variant; large-batch friendly. Theoretical: sign update robust to gradient scale heterogeneity.
- AutoML-Zero (Real 2020): search whole ML pipeline from scratch via regularized evolution on toy tasks. Lion = same evolutionary-program-search spirit but aimed at SOTA-generalizing optimizers, with the AdamW signature constraint.
- Regularized evolution (Real 2019): tournament selection, remove oldest (age-based).

## Design decisions -> why
- Program representation (not black-box net like L2O): analyzable, transferable, length = complexity proxy -> select simpler/generalizable.
- AdamW signature / 2-buffer cap: ensures discovered optimizer no heavier than AdamW.
- interp() primitive: compact way to express EMAs/convex combos -> short programs.
- Warm-start = AdamW: huge prior; pure-random search over 2M programs still worse than AdamW.
- Funnel/meta-validation selection: proxy tasks ~20min/1 TPU vs target >1e4x; meta-overfitting real; later-meta-overfit runs generalize better.
- sign on (interp of g & m): uniform-magnitude step + injected noise => regularization; large-batch robust (gain grows with batch size).
- beta1=0.9 (update) < beta2=0.99 (momentum): separate roles; long memory + recent emphasis. Tuning suggestion beta1=0.95,beta2=0.98 for stability when AdamW beta2<0.999.
- lr 3-10x smaller, lambda 3-10x larger than AdamW: compensate larger sign-update norm; keep effective wd = lr*lambda.

## Code (lucidrains, faithful to google/automl)
update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
  p.data.mul_(1 - lr*wd)                                   # decoupled (stepweight) weight decay
  update = exp_avg.clone().mul_(beta1).add(grad, alpha=1-beta1).sign_()  # sign(beta1*m + (1-beta1)*g)
  p.add_(update, alpha=-lr)                                # w -= lr*update
  exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)            # m = beta2*m + (1-beta2)*g
Lion(Optimizer): lr=1e-4, betas=(0.9,0.99), weight_decay=0.0. state['exp_avg']=zeros. (Note lucidrains applies wd as p*=(1-lr*wd) which is the decoupled form; equivalent to theta -= eta*lambda*theta in the appendix algorithm.)

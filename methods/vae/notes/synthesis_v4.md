# Synthesis V4 — Auto-Encoding Variational Bayes (notes-first)

This is the composing source for results_v4. Everything in context/reasoning/answer is transcribed
FROM here. V3 was thin (~16.6k context, ~30k reasoning) and — critically — its context.md
Code-framework leaked the method (named `reparameterize`, presupposed `z=μ+σ⊙ε`, mu/logvar heads).
V4 fixes that: the scaffold must presuppose NOTHING about the inference method.

================================================================================
PART A — The single most important framing question
================================================================================
Q: What is the precise first-principles object, and what is the central difficulty?
A: Object = marginal log-likelihood log p_θ(x) = log ∫ p_θ(z) p_θ(x|z) dz, for a directed model
   z~p(z), x~p_θ(x|z) where p_θ(x|z) is a NONLINEAR neural net. Difficulty: with a nonlinear
   likelihood the marginal integral has no closed form (can't evaluate or differentiate it), so the
   posterior p_θ(z|x)=p(x|z)p(z)/p(x) is also intractable (shares the same uncomputable denominator).
   PLUS scale: N huge → must update on minibatches; any per-datapoint inner optimization or sampling
   chain is disqualified.
   The whole method is forced by ONE recurring obstruction: how to take ∇_φ of an expectation
   E_{q_φ(z|x)}[f(z)] whose MEASURE depends on φ. Every classical tool dies at exactly this object.

================================================================================
PART B — Load-bearing ancestors (verified against primary text §1, §2, related work)
================================================================================

[ELBO identity / EM] (Dempster 1977; Roweis & Ghahramani 1999 = [Row98] for the linear-Gaussian/PCA instance)
- Exact decomposition for ANY q: log p(x) = L + KL(q(z)‖p(z|x)), L = E_q[log p(x,z) − log q(z)].
  KL≥0 ⇒ L is lower bound; tight iff q=p(z|x). DERIVE in reasoning from KL definition + substitution
  log p(z|x)=log p(x,z)−log p(x).
- WHY maximize L not log p(x): log p(x) has the uncomputable ∫dz inside the log; L turns it into an
  EXPECTATION (log inside the expectation, samplable), and is a GUARANTEED underestimate
  (log p(x)−L = KL ≥ 0). One quantity does both jobs: improving q ↓ KL(q‖p(z|x)) (inference);
  improving model ↑ a guaranteed lower bound (learning).
- EM: E-step sets q=p(z|x;θ_old) (KL=0, tight); M-step maximizes E_q[log p(x,z|θ)] (complete-data LL,
  log inside expectation). REQUIRES a writable posterior for the E-step.
- WALL: nonlinear-net likelihood ⇒ no closed-form posterior ⇒ E-step can't even start. EM dead at door.
- Roweis: PCA = ML solution of linear-Gaussian p(z)=N(0,I), p(x|z)=N(Wz,εI) as ε→0. Establishes the
  lineage linear-AE → generative latent model, but ONLY linear/Gaussian (tractable posterior).

[Mean-field VI / Stochastic VI] (Hoffman, Blei, Wang, Paisley 2013 = [HBWP13]; review Blei 2016)
- Replace intractable posterior with q from tractable family, maximize ELBO. Mean-field q=∏_j q_j(z_j),
  CAVI update log q_j*(z_j) = E_{−j}[log p(x,z)] + const.
- WALL 1 (conjugacy): the CAVI expectation E_{−j}[log p(x,z)] is closed-form in z_j only for
  conditionally-conjugate exponential-family models; a nonlinear network in log p(x|z) blows it up.
- WALL 2 (correlations): factorial ∏_j q_j can't represent posterior correlations.
- WALL 3 (scale): per-datapoint local variational params optimized to convergence before each global
  update; param count grows with N; new test point needs a fresh inner optimization. SVI fixes scale
  (stochastic natural-gradient on globals) but STILL needs conjugacy/analytic local updates.

[Score-function / black-box VI gradient] (Blei,Jordan,Paisley 2012 = [BJP12]; Ranganath BBVI 2014 = [RGB13])
- Drop conjugacy: estimate gradient directly. Log-derivative identity ∇_φ q_φ = q_φ ∇_φ log q_φ ⇒
  ∇_φ E_{q_φ}[f(z)] = E_{q_φ}[ f(z) ∇_φ log q_φ(z) ] ≈ (1/L)Σ f(z^l) ∇_φ log q_φ(z^l). Unbiased,
  black-box (only evaluates f), works for discrete z. = REINFORCE.
- WHY HIGH VARIANCE (must be lived, not asserted): estimator = f(z)·s(z), s=∇_φ log q_φ. Score has
  zero mean: E_q[s] = ∫ q ∇log q = ∫∇q = ∇∫q = ∇1 = 0. So it's a zero-mean score weighted by f(z).
  f(z) is a SCALAR WEIGHT that NEVER enters the φ-differentiation — all gradient signal rides the noisy
  score. Thought experiment: if f≈c (near-constant), true gradient ∇E[f]≈0, but estimator ≈ c·s(z) with
  variance c²·Var(s) — does NOT vanish, scaled by c. The estimator throws away df/dz (the smoother,
  more informative channel). Paper calls it "impractical for our purposes." Needs control variates /
  Rao-Blackwell / baselines just to be usable. → STRUCTURAL diagnosis: "φ is in the measure, f doesn't
  enter the differentiation." This diagnosis directly points at the cure.

[Wake-sleep / Helmholtz machine] (Hinton, Dayan, Frey, Neal 1995 = [HDFN95])
- The ONLY prior online method for the same general class of continuous-latent directed models.
  Already has the SHAPE we want: a separate recognition net q(z|x) (encoder) approximating the
  posterior, trained alongside generative net p(x|z).
- Wake: sample z~q(z|x), update generative weights to raise log p(x,z). Sleep: "dream" (x,z)~p, update
  recognition weights to predict z from x.
- WALL: TWO distinct objectives that DON'T jointly correspond to optimizing one bound on log p(x). The
  sleep phase minimizes a REVERSED KL(p‖q) on FANTASY data, not KL(q‖p(z|x)) on REAL data. No single
  coherent objective, no guarantee the marginal likelihood improves; the two nets can drift incoherently.
- BUT it gives us the AMORTIZATION idea: classical VI gives each x^(i) its own free local params
  (μ^(i),σ^(i)) optimized to convergence; wake-sleep PREDICTS the local params with a single shared-weight
  net q_φ(z|x). Inference = one forward pass, param count fixed in N, generalizes to new x. This is the
  scaling lever. We keep the amortized encoder; we discard the two-objective scheme.

[Salimans & Knowles 2013] (= [SK13]) — closest prior use of the reparameterization-style trick
- Used a reparameterization-like change of variables inside a fixed-form SVI scheme to learn the natural
  params of exponential-family approximating distributions.
- WALL: tied to exp-family natural parameters, NOT a general NN recognition model; no amortized
  network-based posterior trainable jointly with a nonlinear generative model.

[Monte Carlo EM + HMC] (Duane et al. 1987 = HMC) — sampling-based baseline
- Intractable E-step posterior → sample with gradient MCMC using ∇_z log p(z|x)=∇_z log p(z)+∇_z log p(x|z)
  (computable: normalizer drops out). Then M-step.
- WALL: one sampling chain per datapoint → not online, doesn't scale. Small-scale reference only.

[Autoencoder lineage] (Vincent 2010 = [VLL+10]; Bengio 2013 = [BCV13]; PSD Kavukcuoglu 2008 = [KRL08];
 infomax Linsker 1989)
- Denoising/contractive/sparse AEs: reconstruction + AD-HOC regularizer; infomax reading: reconstruction
  lower-bounds I(X;Z). PSD = predictive encoder for sparse coding (inspiration for amortization).
- WALL: reconstruction alone insufficient for useful representations; regularizers carry nuisance
  hyperparameters with no probabilistic meaning; no marginal-likelihood objective behind them.
- PAYOFF the method delivers: its KL term is a PRINCIPLED regularizer set by log p(x) itself — no
  nuisance hyperparameter.

[Rezende, Mohamed, Wierstra 2014 = [RMW14]] — concurrent/independent, NOT a predecessor (DLGM /
 stochastic backprop). Mention only as concurrent perspective; do not lean on it.

================================================================================
PART C — The two ways to differentiate an expectation (the conceptual hinge)
================================================================================
(1) Score-function: φ in measure → ∇_φ E_{q_φ}[f] = E_{q_φ}[f ∇_φ log q_φ]. High variance (Part B).
(2) Pathwise/reparameterized: write z = g_φ(ε,x), ε~p(ε) FIXED (φ-independent). Change of variables
    (probability conservation): q_φ(z|x) ∏_i dz_i = p(ε) ∏_i dε_i ⇒
    ∫ q_φ(z|x) f(z) dz = ∫ p(ε) f(g_φ(ε,x)) dε ⇒ E_{q_φ(z|x)}[f] = E_{p(ε)}[f(g_φ(ε,x))]. Same
    expectation, φ-FREE measure ⇒ ∇_φ moves INSIDE: ∇_φ E_{p(ε)}[f(g_φ)] = E_{p(ε)}[∇_φ f(g_φ)] =
    E_{p(ε)}[ ∂f/∂z · ∂g_φ/∂φ ]. The cure for the structural diagnosis: it puts φ in the INTEGRAND,
    so the pathwise df/dz channel — the one score-function discards — becomes the gradient's backbone.
    Same sample count, much lower variance.
- Univariate Gaussian minimal example: z~N(μ,σ²), ε~N(0,1), z=μ+σε; differentiable in μ (directly) and
  σ (multiplies ε).
- Coverage (so it's not a Gaussian-only trick): (i) tractable inverse CDF — ε~U(0,I), g=inverse CDF
  (Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Gompertz, Gumbel, Erlang); (ii) any
  location-scale family — ε=standard (loc 0, scale 1), g=loc+scale·ε (Laplace, Elliptical, Student-t,
  Logistic, Uniform, Triangular, Gaussian); (iii) composition — Log-Normal (exp of normal), Gamma (sum
  of exponentials), Dirichlet (weighted Gamma), Beta, Chi², F. When all fail: good numerical
  inverse-CDF approximations at PDF-comparable cost (Devroye 1986).

================================================================================
PART D — Assembling the estimator (SGVB A/B), minibatch, L=1
================================================================================
- ELBO two equivalent forms:
  Form 1 (joint): L(x) = E_{q_φ(z|x)}[ log p_θ(x,z) − log q_φ(z|x) ].
  Form 2 (KL+recon): L(x) = − KL(q_φ(z|x)‖p_θ(z)) + E_{q_φ(z|x)}[ log p_θ(x|z) ].
  Derive form 2: E_q[log p(x|z)+log p(z)−log q] = E_q[log p(x|z)] + E_q[log p(z)−log q] and the second
  block = −KL(q‖p(z)).
- SGVB estimator A (generic, plug reparam into form 1):
  L̃^A(x) = (1/L)Σ_l [ log p_θ(x,z^l) − log q_φ(z^l|x) ], z^l=g_φ(ε^l,x), ε^l~p(ε). Differentiable in
  θ AND φ. FULLY GENERIC: even log q_φ only needs evaluation; KL need not be closed-form ⇒ works for
  non-Gaussian prior/posterior.
- SGVB estimator B (lower variance, plug reparam into form 2): when KL is analytic, keep it EXACT (don't
  hand a precisely-known quantity to Monte Carlo — that only injects variance) and sample only recon:
  L̃^B(x) = − KL(q_φ(z|x)‖p_θ(z)) + (1/L)Σ_l log p_θ(x|z^l). Rao-Blackwell-flavored: analytic part stays
  analytic. Exposes the AUTOENCODER connection: term 1 = regularizer pulling q to prior; term 2 =
  expected reconstruction LL; g_φ encodes x+noise → z, decoder reconstructs.
- Minibatch scaling: L(X) = Σ_{i=1}^N L̃(x^i) ≈ L̃^M(X^M) = (N/M) Σ_{i=1}^M L̃(x^i). WHY N/M: uniform
  sample of M from N ⇒ E[Σ_{minibatch}] = M·(1/N Σ_all) = (M/N)L(X), so ×(N/M) corrects expectation
  back to full-dataset sum ⇒ unbiased minibatch gradient. (Same factor as the N· in the Full-VB f_φ.)
- WHY L=1 suffices with large M: total samples = M·L, estimator variance ~ 1/(M·L). Under a fixed
  compute budget, L samples of the SAME x are highly correlated (share x and (μ,σ)); spending the budget
  on MORE DISTINCT x averages harder (more independent). Empirically L=1 fine if M large (e.g. M=100).
  ⇒ each step = one plain forward pass: per point draw ε, compute recon, add analytic KL, one backward
  for ∇_{θ,φ}, feed SGD/Adagrad.
- Algorithm: init θ,φ; repeat { minibatch X^M; noise ε; g←∇_{θ,φ} L̃^M(X^M,ε); update θ,φ (SGD/Adagrad) }.

================================================================================
PART E — Concretizing the model (design-decision → WHY table)  [the depth budget]
================================================================================

DECISION                         | WHY THIS, NOT THE OBVIOUS ALTERNATIVE
---------------------------------|--------------------------------------------------------------
Prior p(z)=N(0,I) (centered      | (a) parameter-free → no prior to learn, no prior↔posterior
isotropic Gaussian)              | mutual-collapse to degenerate solutions; (b) isotropic → no
                                 | preferred direction, rotation-symmetric, no imposed structure;
                                 | (c) MOST IMPORTANT computationally: with Gaussian q it gives the
                                 | cleanest closed-form KL. "Too simple?" No — the nonlinear DECODER
                                 | deforms a simple Gaussian into arbitrarily complex data densities;
                                 | complexity belongs in p(x|z), not the prior.
q_φ(z|x) = N(μ^i, σ^i² I)         | Gaussian: location-scale family ⇒ cleanest reparam z=μ+σ⊙ε; pairs
diagonal-covariance Gaussian,    | with N(0,I) for closed-form KL. DIAGONAL not full: full needs
μ,σ from encoder MLP             | J(J+1)/2 params + Cholesky for sampling/KL per point — too costly;
                                 | diagonal = J μ's + J logσ²'s, sampling/KL/grad all elementwise,
                                 | linear in J. Cost: can't model posterior correlations — but this is
                                 | a SIMPLIFYING choice, not a limitation; swap richer q if needed.
sample z = μ + σ ⊙ ε, ε~N(0,I)    | location-scale reparam (Part C); the concrete g_φ for diagonal
                                 | Gaussian. ⊙ = elementwise.
encoder outputs log σ² (not σ)   | network can output any real; exp(·) makes it positive automatically;
                                 | numerically stable (no positivity constraint to enforce, no log of a
                                 | possibly-negative number).
Decoder Bernoulli for binary x   | reconstruction term MUST match data type or the likelihood is wrong.
                                 | Binary pixels ∈{0,1}: multivariate Bernoulli, y=f_σ(W₂tanh(W₁z+b₁)+b₂),
                                 | log p(x|z)=Σ_d[x_d log y_d+(1−x_d)log(1−y_d)] = NEGATIVE per-pixel BCE —
                                 | the BCE loss is the Bernoulli log-likelihood, not an ad-hoc choice.
                                 | sigmoid output ∈(0,1) = probabilities.
Decoder Gaussian for cont. x     | continuous (0,1) data (Frey faces): h=tanh(W₃z+b₃), μ=W₄h+b₄,
                                 | logσ²=W₅h+b₅, log p(x|z)=log N(x;μ,σ²I) ≈ −½Σ[(x−μ)²/σ²+logσ²] =
                                 | learned-variance squared error. Mean squashed to (0,1) by sigmoid.
                                 | WHY match: Gaussian/SE on binary pixels wastes capacity estimating
                                 | variance at 0/1; Bernoulli on continuous grey is undefined.
KL closed form (DERIVE FULLY)    | with p=N(0,I), q=N(μ,σ²I): both integrals carry −(J/2)log2π which
−KL = ½Σ_j(1+logσ_j²−μ_j²−σ_j²)   | CANCELS (same density space) ⇒ KL is elementary in μ,σ, directly
                                 | differentiable. ∫q log p = −(J/2)log2π − ½Σ(μ_j²+σ_j²) [via E[z_j²]=
                                 | μ_j²+σ_j²]. ∫q log q = −(J/2)log2π − ½Σ(1+logσ_j²) [neg entropy; via
                                 | E[(z_j−μ_j)²]=σ_j² ⇒ each quadratic term =1]. Subtract → result. If
                                 | prior/posterior were different families the log2π would NOT cancel and
                                 | we'd fall back to estimating KL by sampling (estimator A, higher var).
                                 | Sanity: σ→1,μ→0 ⇒ each term 0 ⇒ KL=0 (q=p); −μ² pulls μ→0; logσ²−σ²
                                 | maximized at σ=1 pulls σ→1. = "regularize to prior."
weight decay on θ = N(0,I) prior  | training maximizes L plus a small quadratic pull on θ → approximate
on θ → approximate MAP            | MAP, not pure ML. Cheap, principled.
L=1, M=100                        | see Part D.
Optimizer Adagrad/SGD (orig);     | objective is just a differentiable scalar; any SGD-family optimizer
Adam in canonical code            | works. Canonical pytorch/examples uses ReLU+Adam (faster); original
                                 | used tanh/sigmoid + Adagrad. Interchangeable over the same objective.

================================================================================
PART F — Full-VB extension (appendix) — repeat reparam at the parameter level
================================================================================
- Also do VI on global θ (not just point-estimate). Hyperprior p_α(θ), approx posterior q_φ(θ).
  Two-layer ELBO: log p_α(X)=KL(q_φ(θ)‖p_α(θ|X))+L(φ;X), L(φ;X)=∫q_φ(θ)(log p_θ(X)+log p_α(θ)−log q_φ(θ))dθ;
  log p_θ(X)=Σ_i log p_θ(x^i), each with inner z-bound.
- Reparam BOTH layers: θ=h_φ(ζ), ζ~p(ζ); z=g_φ(ε,x), ε~p(ε).
- Shorthand f_φ(x,z,θ)=N·(log p_θ(x|z)+log p_θ(z)−log q_φ(z|x)) + log p_α(θ) − log q_φ(θ).
  The N· makes the per-point inner block dimensionally match the once-per-dataset global term
  log p_α(θ)−log q_φ(θ) (sibling of the N/M minibatch factor).
- Estimate L(φ;X) ≈ (1/L)Σ_l f_φ(x^l, g_φ(ε^l,x^l), h_φ(ζ^l)); both noise layers φ-independent ⇒
  differentiable in φ. If all four (p_α(θ),p(z),q_φ(θ),q_φ(z|x)) Gaussian, four KL subterms analytic ⇒
  each layer's ½Σ(1+logσ²−μ²−σ²). PUNCHLINE: reparam is a GENERAL operator for "differentiating a
  parameterized measure," reused verbatim across levels — not a Gaussian-latent special case. Main line
  keeps MAP on θ (enough, simple).

================================================================================
PART G — Marginal-likelihood estimator (appendix; eval-only, goes to context Evaluation, NOT reasoning)
================================================================================
- For scoring trained models, low-dim latent (≤5): (1) sample {z^l} from posterior via gradient MCMC
  (HMC, ∇_z log p(z|x)=∇_z log p(z)+∇_z log p(x|z)); (2) fit density estimator q(z) to samples;
  (3) fresh posterior samples → p(x) ≈ ( (1/L)Σ_l q(z^l)/[p(z)p(x|z^l)] )^{-1}, z^l~p(z|x). Derivation:
  1/p(x) = ∫ q(z) dz / p(x) = ∫ p(z|x) q(z)/p(x,z) dz ≈ (1/L)Σ q(z^l)/[p(z)p(x|z^l)]. (This is eval
  machinery, pre-method-agnostic; lives in context.md Evaluation settings, kept out of reasoning.md.)

================================================================================
PART H — CODE-FRAMEWORK = MINIMAL PRE-METHOD SCAFFOLD (the V4 fix)
================================================================================
Presuppose NOTHING about the inference method. NO reparameterization, NO SGVB, NO amortized-encoder-as-
the-trick, NO "ELBO" named, NO method names, NO "reference implementation"/"official repo" wording.
What IS genuinely known before the method:
 - we have a directed latent-variable generative model, want to fit it by (approx) max-likelihood;
 - a prior p(z)=N(0,I) (a standard modeling choice that predates the method);
 - we need a net z→params(x) (a generative/decoder net) and — since we'll want fast per-point inference —
   a net x→variational-params (a recognition net), both as generic MLPs;
 - we need SOME training objective (a bound on log p(x)) and SOME way to draw a latent and get gradients,
   BOTH UNKNOWN → bare # TODO stubs;
 - mature substrate: MLP/backprop primitives, Adagrad/SGD optimizer, minibatch data loader.

Scaffold (pre-method vocabulary; corresponds piece-for-piece to final code):
```python
import torch
from torch import nn, optim
from torch.nn import functional as F

# Mature substrate that already exists: minibatch loader, MLP layers, autograd, an adaptive SGD optimizer.
train_loader = ...  # yields random minibatches X^M of M datapoints from the dataset

def log_prior(z):
    # p(z) = N(0, I): a standard, parameter-free modeling choice that predates this work.
    return -0.5 * (z ** 2 + torch.log(2 * torch.pi * torch.ones_like(z))).sum(-1)

class Decoder(nn.Module):
    # Generative net: maps a latent z to the parameters of p(x|z) (an MLP).  # TODO: define layers
    def forward(self, z):
        ...  # TODO: z -> parameters of the observation model p(x|z)

class Encoder(nn.Module):
    # Recognition net: maps a datapoint x to the parameters of a variational distribution over z (an MLP),
    # so per-point inference is one forward pass (shared weights across the dataset).  # TODO: define layers
    def forward(self, x):
        ...  # TODO: x -> parameters of q(z|x)

def draw_latent(enc_out):
    # TODO: how to sample z from the variational distribution AND keep it differentiable w.r.t. the
    #       recognition-net parameters — the gradient estimator we still have to design.
    raise NotImplementedError

def objective(x, encoder, decoder):
    # TODO: a tractable, low-variance, minibatch-friendly bound on log p(x) plus its gradient estimator
    #       w.r.t. BOTH the generative and recognition parameters — to be designed.
    raise NotImplementedError

encoder, decoder = Encoder(), Decoder()
optimizer = optim.Adagrad(list(encoder.parameters()) + list(decoder.parameters()))

for x, _ in train_loader:                 # minibatch X^M
    optimizer.zero_grad()
    loss = -objective(x, encoder, decoder)
    loss.backward()
    optimizer.step()
```
The final code FILLS these stubs: Encoder→two heads (μ, logvar); draw_latent→z=μ+σ⊙ε reparam;
objective→ −KL(closed form) − recon (SGVB B); Decoder→sigmoid Bernoulli. Same structure, TODOs become bodies.

================================================================================
PART I — Final code (grounded in pytorch/examples vae/main.py), written FIRST then hollowed
================================================================================
(see code/main.py). Encoder fc1 784→400, heads fc21→μ(20), fc22→logvar(20); reparameterize
std=exp(0.5 logvar), eps=randn_like, z=mu+eps*std; decode fc3 20→400, fc4 400→784 sigmoid;
loss = BCE(recon,x,sum) + KLD, KLD=−0.5·Σ(1+logvar−μ²−exp(logvar)); Adam lr=1e-3, M=128.
README: ReLU+Adam (canonical) vs original tanh/sigmoid+Adagrad — interchangeable.

================================================================================
PART J — STANDING RULES checklist for the three files
================================================================================
- In-frame: no source-paper artifact (no "this paper"/authors/venue/arXiv/title); prior-art citations
  (wake-sleep, EM, mean-field, Salimans&Knowles, Roweis, Hoffman, Blei, Ranganath, Duchi, Devroye) FINE.
- context.md must NOT name "VAE"/"AEVB"/"SGVB"/"reparameterization" in research-question/background framing;
  five section headers exactly: Research question / Background / Baselines / Evaluation settings / Code framework.
- context.md Code framework = Part H scaffold (NO method names, NO "reference implementation"/"official repo").
- reasoning.md: continuous first-person Chinese, ZERO real markdown headers, all main-text + appendix
  derivations inline (ELBO identity, score-function variance lived, reparam change-of-vars, KL full
  derivation with log2π cancellation, A/B, N/M, L=1, Bernoulli/Gaussian decoders, Full-VB two-layer).
  Insight-before-method everywhere (no state-then-justify). Dead ends marked (撞墙) + aha (我笑了).
  Target ≥28k chars. End on real code + causal-chain recap. No experiment numbers, no hindsight, no meta footer.
- answer.md: open with the method (VAE) named; clean algorithm + objective; faithful PyTorch code; no
  citation header, no arXiv links in code comments.

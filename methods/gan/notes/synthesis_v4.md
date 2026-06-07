# Synthesis V4 — GAN (Generative Adversarial Nets, arXiv 1406.2661)

This is the notes-first synthesis the V4 deliverables are composed FROM (not from memory).
Built on the verified V3 base (research_notes.md) + a re-read of adversarial.tex (no separate
appendix; all proofs inline in Sec.4) + the goodfeli/adversarial Theano-Pylearn2 `AdversaryCost2`
+ the eriklindernoren clean PyTorch reference. All "why"s are reconstructed in-frame; explainers
(NIPS-2016 tutorial, density-ratio framing) were read only to UNDERSTAND, never cited.

VERIFIED FACTS (from source):
- Value fn Eq.1: min_G max_D V = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1-D(G(z)))].  (tex L250)
- Optimal D (Prop.1): D*_G = pdata/(pdata+pg). Pointwise a·log y + b·log(1-y) max at a/(a+b);
  D need not be defined off Supp(pdata)∪Supp(pg).  (tex L417-420)
- Inner D-objective = max log-likelihood of conditional P(Y=y|x), y=1 data / y=0 pg.  (tex L423)
- Thm1: C(G)=−log4+KL(pdata‖m)+KL(pg‖m)=−log4+2·JSD(pdata‖pg), m=(pdata+pg)/2;
  global min −log4 iff pg=pdata.  (tex Thm1)
- Prop2 convergence: U(pg,D) convex in pg (linear in pg for fixed D); subderivative of a sup of
  convex functions includes the derivative at the argmax; sup_D U convex w/ unique optimum ->
  small steps converge pg->pdata.  (tex L473-477)
- Non-saturating: text (L268-273) — train G to MAXIMIZE log D(G(z)) instead of minimize
  log(1-D(G(z))) for stronger early gradients.
- DISCREPANCY confirmed: Algorithm-1 box (tex L368) says G *descends* its gradient of
  log(1-D(G(z))) (minimax form); released code (goodfeli __init__.py L359)
  g_obj = d.layers[-1].cost(y1, y_hat0) = label fakes REAL -> minimize BCE = MAXIMIZE log D(G(z))
  (non-saturating). So box=minimax, code=non-saturating.
- Algorithm 1: k D-steps (k=1 used) then 1 G-step; minibatch SGD with momentum.  (tex L344-375)
- "Helvetica scenario" = mode collapse: G over-trained vs stale D collapses many z onto few x.
- goodfeli: G = MLP (ReLU + sigmoid), D = maxout + dropout. Parzen-window LL eval (sigma=0.2 default).

================================================================================
DESIGN-DECISION -> WHY TABLE  (each lived out in-frame in reasoning.md, insight-before-method)
================================================================================

DD1. Choose an IMPLICIT generator x=G(z) (only sample, never write p(x)).
  WHY: the generative taxonomy is a tree organized by how you get a handle on p(x). Explicit &
  exactly tractable (FVBN, p=∏p(x_i|x_<i)) -> exact likelihood but sequential O(d) sampling, no
  latent. Explicit up to something (Boltzmann: intractable Z; VAE: variational bound + inference
  net). Implicit: keep ONLY the ability to draw a sample. Every explicit branch pays a tax
  (Z / analytic unnormalized density / bound+encoder). Picking implicit BY CONSTRUCTION means you
  never normalize, never integrate, never write a density -> the tax is structurally avoided.
  REJECTED ALTERNATIVES & failure: FVBN (slow sequential sampling); Boltzmann (Z + MCMC mixing);
  VAE (bound + encoder, blurry from per-pixel reconstruction); score matching (still needs analytic
  unnormalized p̃, undefined for deep multi-latent models).

DD2. The downstream training signal = a LEARNED classifier's success (not a likelihood).
  WHY: implicit models had no obvious thing to maximize. NCE already showed a *classifier* can be
  the learning signal for a generative model (turn density estimation into logistic regression).
  Reuse that idea but drop the requirement that the model be an analytic unnormalized density.
  REJECTED: maximum likelihood (needs explicit p), variational bound (needs encoder + bound).

DD3. The contrast = the GENERATOR ITSELF, learned, not a fixed distribution/statistic.
  WHY: NCE's fatal flaw is that its noise is FIXED -> once the model is approximately right on a few
  easy features, telling data from fixed noise is trivial, classifier saturates, signal dies. Same
  disease for a fixed two-sample statistic (MMD): once G matches data on the features the statistic
  sees, the signal flattens even if distributions still differ on unseen features. A *learned*
  contrast is an ADAPTIVE measuring stick: it actively hunts whatever feature currently separates
  fake from real and re-sharpens as G closes each gap, so the task never goes slack. This is the
  single move that simultaneously fills the implicit-model "no loss" hole AND the NCE "pushover" hole.
  REJECTED: fixed noise (NCE), fixed MMD statistic — both static, both go slack.

DD4. The adversary is a CLASSIFIER with sigmoid output, not a direct density-ratio regressor.
  WHY: distinguishing two distributions is governed by the ratio pdata/pg; the Bayes-optimal
  real-vs-fake classifier (equal priors) outputs pdata/(pdata+pg) — the ratio squashed into (0,1).
  Raw ratio ranges (0,∞): tiny here, huge there, numerically miserable, must clip. Sigmoid classifier
  gives the same information, bounded & stable. And it is well-defined even when neither density can
  be written in closed form -> a classifier stands in for two intractable densities.
  REJECTED: regress pdata/pg directly (unbounded, unstable, requires clipping).

DD5. The objective V = E[log D(x)] + E[log(1-D(G(z)))], minimax min_G max_D.
  WHY: D wants to assign the correct label — log D(x) high on data, log(1-D(G(z))) high on fakes
  (1-D = prob fake). That sum is exactly the (negative) BCE of a real-vs-fake classifier (label 1
  data, 0 fakes) — the natural loss for the well-behaved maxout/dropout units, no new machinery. G
  wants D to fail on fakes -> minimize the very thing D maximizes -> one value function, two players.
  Corner checks: pg=pdata -> D≡1/2 -> V=−log4; G terrible -> D≈1 on data,≈0 on fakes -> V->0 (max).

DD6. Optimal discriminator D*_G = pdata/(pdata+pg) (inner max solved nonparametrically).
  WHY/HOW: rewrite 2nd term as expectation under pg (LOTUS / pushforward of pz through G — the
  payoff of defining pg as pushforward: never need its formula). V=∫[pdata log D + pg log(1-D)]dx.
  Maximize pointwise: a log y + b log(1-y), a=pdata b=pg; d/dy = a/y − b/(1-y)=0 -> y=a/(a+b);
  2nd deriv −a/y²−b/(1-y)²<0 -> concave -> max. D* literally = Bayes-optimal squashed ratio (closes
  DD4 loop). Inner objective = fitting logistic posterior P(Y=1|x) — same skeleton as NCE but the
  negative class is the LEARNED pg, not fixed noise.

DD7. C(G)=max_D V collapses to −log4 + 2·JSD(pdata‖pg) (line-by-line).
  WHY/HOW: C(G)=E_pdata[log(pdata/(pdata+pg))]+E_pg[log(pg/(pdata+pg))]. Subtract
  −log4 = E_pdata[−log2]+E_pg[−log2]. Fold log2 in: log(pdata/(pdata+pg))+log2 =
  log(2pdata/(pdata+pg)) = log(pdata/((pdata+pg)/2)) -> E_pdata[...] = KL(pdata‖m), m=(pdata+pg)/2;
  symmetric second term = KL(pg‖m). Sum of two KLs-to-mixture = 2·JSD. JSD≥0, =0 iff equal ->
  global min −log4 unique at pg=pdata. The JSD was NOT engineered — wrote "fool a Bayes classifier",
  solved inner max, JSD fell out. D at optimum secretly hands G a symmetric divergence to descend.

DD8. The divergence that emerges is JSD (symmetric, bounded), not KL — and that is GOOD.
  WHY: real-vs-fake is a symmetric two-class problem, so a symmetric divergence (JSD) emerges, not
  asymmetric KL with its mode-covering/mode-seeking baggage; the mixture m has support wherever
  either does, so logs never blow up -> JSD finite even on disjoint supports. Symmetry+boundedness
  are exactly the properties a sane training signal wants — gotten for free from the game structure.

DD9. Alternating updates = subgradient descent on a convex functional.
  WHY/HOW: view U(pg,D)=V as fn of pg. For fixed D, pg enters only via ∫pg log(1-D)dx — LINEAR in pg
  -> convex. C(pg)=sup_D U is a sup of fns convex in pg -> convex. Subgradient fact: if f=sup_α f_α,
  each f_α convex, then ∂f_β ⊂ ∂f when β=argsup. So gradient of U at D=D* is a valid subgradient of C
  -> computing grad at the OPTIMAL D and stepping pg downhill IS subgradient descent on convex C with
  unique optimum -> small steps converge pg->pdata. CAVEAT: real optimization is over θ_g through an
  MLP -> limited family + many critical points -> convexity holds in DISTRIBUTION space only, not
  parameter space. Justification for MLPs anyway = brute empirical fact they optimize well (same leap
  every deep net rests on).

DD10. k discriminator steps per 1 generator step (k=1 in practice).
  WHY: theory wants D AT its optimum, but optimizing D to completion each step is prohibitive AND on
  finite data a fully-optimized D overfits (memorizes which points are real) -> useless distributional
  gradient. Patch: take k D-steps then 1 G-step; if G moves slowly D stays NEAR optimal across outer
  steps (track, don't re-burn-in). This is exactly SML/PCD persistent-chain logic (carry the inner
  state across learning steps), with D's params as the carried state. k=1 keeps D close enough &
  cheapest. Convergence only needs D NEAR optimal, which this buys.

DD11. Mode collapse ("Helvetica") defense = same synchronization schedule.
  WHY: if G trains hard vs a STALE D, G can collapse many z onto a few outputs that fool current D —
  wins local battle, throws away diversity, pg stops covering pdata. Defense = don't let G outrun D,
  keep D fresh -> same balance the k-step schedule already imposes. So the schedule does DOUBLE duty:
  keeps D near-optimal for the theory AND prevents collapse.

DD12. Non-saturating generator loss: maximize log D(G(z)) instead of minimize log(1-D(G(z))).
  WHY/HOW: minimax G-term log(1-D(G(z))). Early, G garbage -> D rejects confidently -> D(G(z))≈0.
  d/dD log(1-D) = −1/(1-D) ≈ −1 near D=0, and the curve is FLAT near 0 -> gradient into θ_g tiny
  EXACTLY when G is losing & most needs a push. NCE's saturation sneaking back via G's own loss.
  Fix WITHOUT changing the fixed point: G doesn't need to literally minimize log(1-D); it needs a
  signal pushing D(G(z))->1 with strong gradient when losing. So maximize log D(G(z)):
  d/dD log D = 1/D -> BLOWS UP as D->0 -> huge signal when crushed, vanishes as fight evens. Same
  fixed point (both optimized by D(G(z))->1). This is the heuristic that actually TRAINS. (Tie to
  DISCREPANCY: Alg-1 box descends log(1-D); released code maximizes log D(G(z)).)

DD13. Noise prior pz + fully differentiable deterministic G.
  WHY: only stochasticity is z~pz; everything after is deterministic G(·;θ_g). That structure is what
  lets backprop carry D's gradient through G's outputs into θ_g (reparameterization-style). If
  randomness were INSIDE G as a stochastic unit, no clean gradient through sampling. pz arbitrary
  (gaussian/uniform/spherical) since G reshapes it into pg; noise only at G's bottom layer in practice
  (theory permits all layers). No feedback loop in generation -> piecewise-linear units usable freely
  (the very thing the GSN recurrence couldn't afford).

DD14. G = ReLU/rectifier + sigmoid; D = maxout + dropout.
  WHY: D is a powerful classifier trained on finite data against a MOVING target -> dropout to keep it
  from overfitting the current G; maxout for the cleanest piecewise-linear gradients in the classifier.
  G uses rectifier + sigmoid output (bound to data range). All feedforward, no recurrence -> units free.

================================================================================
LOAD-BEARING ANCESTORS (write-ups; each elaborated, with the gap that motivates the next step)
================================================================================

A. RBM/DBM (Smolensky86, Hinton06, Salakhutdinov+Hinton09). Undirected energy-based p=exp(−E)/Z.
   ML gradient = −E_data[∂E/∂θ] + E_model[∂E/∂θ]; the negative phase (expectation under MODEL) is
   what Z hides -> MCMC (CD/PCD). GAP: intractable Z never leaves; learning at mercy of chain MIXING
   between modes (local operators can't cross low-prob deserts -> biased negative samples, slow/fragile).
   Sampling itself needs a chain. = the canonical "explicit-probability tax."

B. DBN (Hinton06). Hybrid: undirected RBM top over directed layers, greedy layerwise. GAP: inherits
   the computational difficulties of BOTH directed and undirected worlds.

C. Score matching (Hyvärinen 2005). Match ∇_x log p model-vs-data; ∇_x log Z=0 so Z cancels. DAE
   (Vincent08)/CAE learning rules ≈ score matching on an RBM. GAP: still needs the unnormalized
   density p̃ written ANALYTICALLY (need ∇_x log p̃); for multi-latent deep models no tractable p̃ ->
   doesn't apply to the models we care about. Same wall, repainted.

D. NCE (Gutmann+Hyvärinen 2010). CLOSEST in spirit. Density estimation -> logistic regression:
   classifier built from model's own log p̃(x;θ) through a sigmoid, Z a learned parameter, discriminating
   data from FIXED noise. Core reusable idea: a classifier's success = learning signal for a generative
   model; model improves by becoming hard to distinguish from data. GAPS: (1) still needs analytic
   unnormalized density -> deep latent models out; (2) noise FIXED -> once model approx right on easy
   features, task trivial, classifier saturates, learning stalls. Hanging question -> "what if the
   contrast kept improving so the task never goes slack?" -> answered by DD3.

E. GSN (Bengio ICML2014) / generalized denoising AE (Bengio NIPS2013). Give up explicit density, train
   a generative MACHINE (sampler): parameterize one step of a generative Markov chain, trainable by
   backprop. Closest implicit/sample-only precedent. GAPS: sampling still runs a Markov chain (mixing
   returns); recurrent feedback loop hostile to piecewise-linear units (unbounded activation fed back
   can blow up) -> loses the best ingredients.

F. VAE / auto-encoding variational Bayes (Kingma+Welling 2014) + stochastic backprop (Rezende 2014).
   Same-era backprop-into-a-generator: reparameterization -> gradients through sampling; maximize ELBO
   with a learned inference (encoder) net regularized to the prior. GENUINELY close: differentiable
   generator, pure backprop, no chain. GAPS: still likelihood-based (a BOUND), still needs encoder at
   train time, per-pixel reconstruction term -> blurry samples.

G. SML/PCD (Younes99, Tieleman08). Persistent negative chains carried across learning steps instead of
   re-burning-in each step. = the template for keeping an inner quantity NEAR optimum while outer params
   move slowly -> justifies k=1 D-step (DD10).

H. Wake-sleep (Hinton95). Separate recognition net learned to invert a generator — template for bolting
   learned approximate inference on after the fact (a possible extension, not core).

I. Density ratio / Bayes-optimal classifier. Telling two distributions apart is governed by pdata/pg;
   equal-prior optimal classifier outputs pdata/(pdata+pg), a monotone transform of the ratio, defined
   even when neither density is closed-form -> a classifier stands in for two intractable densities (DD4,DD6).

J. Reparameterized differentiable generator. x=G(z), z~pz, deterministic differentiable -> backprop
   through sampling. Old in stats (derivative identities Price 1958, Bonnet 1964); active in stochastic
   backprop / VAE. Open hole it leaves: WHAT downstream loss, if not a likelihood/bound? (-> DD2/DD3.)

K. Info-theoretic divergences. KL(p‖q)=∫p log(p/q) asymmetric, can be ∞ on disjoint supports;
   JSD=½KL(p‖m)+½KL(q‖m), m=(p+q)/2, symmetric, ≥0, =0 iff equal, finite even on disjoint supports.
   The symmetric/bounded properties matter for which divergence the symmetric game produces (DD8).

================================================================================
EVALUATION SETTINGS (pre-method facts; NO outcomes)
================================================================================
- Datasets: MNIST (LeCun98), Toronto Face DB (Susskind10), CIFAR-10 (Krizhevsky+Hinton09). FC + conv.
- Metric for sample-only models: Gaussian Parzen-window log-likelihood — fit isotropic-Gaussian KDE to
  generated samples, pick bandwidth σ by CV on validation, report test mean log-lik. (Breuleux+Bengio
  2011; Rifai12; Bengio ICML13/14.) Known high-variance & poor in high-dim but the accepted yardstick.
- Qualitative: fair random draws (not cherry-picked), each sample's nearest training example (anti-
  memorization), and linear interpolation in z-space decoded along the path (manifold smoothness).

================================================================================
CODE FRAMEWORK (pre-method scaffold) <-> FINAL CODE correspondence
================================================================================
The V4 context.md Code-framework section must be a MINIMAL pre-method scaffold that presupposes NOTHING
about the adversarial signal. Known-before-the-method primitives ONLY:
  - Theano: T.grad(cost, params) returns gradient of one scalar expr wrt named params; MRG_RandomStreams
    .normal/.uniform noise sources. (Bergstra10, Bastien12)
  - Pylearn2: MLP/Layer (incl maxout), dropout_fprop, Cost abstraction (expr/get_gradients), SGD+momentum.
    (Goodfellow13)
  - Piecewise-linear units (ReLU Jarrett09/Glorot11, maxout Goodfellow13) + dropout (Hinton12).
Scaffold = bare IMPLICIT-GENERATIVE-MODEL harness ONLY:
  - z ~ p(z) noise prior sampler
  - class Generator:  # TODO net mapping z -> sample (differentiable)
  - def train_step(real_batch):  # TODO the training signal we'll design  (PASS/TODO body)
  - an SGD(+momentum) optimizer
NO discriminator, NO minimax, NO BCE-twice cost, NO density-ratio, NO method names in context.md.
NO "reference implementation"/"official repo" wording.

FINAL CODE FILLS IN train_step (derive scaffold by hollowing out the final code):
  (i) goodfeli Theano/Pylearn2 Cost mapping: d_obj = 0.5*(cost(y1,D(X)) + cost(y0,D(G(z)))),
      g_obj = cost(y1, D(G(z))) [non-saturating], T.grad each wrt disjoint param sets, gate G once/k.
  (ii) clean PyTorch: Generator (Linear+LeakyReLU(+BN)->Tanh), Discriminator (Linear->LeakyReLU->Sigmoid),
       BCELoss, two Adam opts (lr2e-4, betas(0.5,0.999)), G-loss = BCE(D(G(z)),valid=1) [non-saturating],
       D-loss = (BCE(D(real),1)+BCE(D(G(z).detach()),0))/2. z~N(0,1). detach keeps D's grad off G.
  Note the Alg-1-box (minimax, descend log(1-D)) vs released-code (non-saturating, maximize log D(G(z)))
  discrepancy in answer.md.

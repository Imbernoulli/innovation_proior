# WGAN — synthesis notes (Phase 1.5)

## The pain point
We want to learn a distribution P_r over high-dim data (images) by pushing a noise
prior z~p(z) through a neural net g_theta to get P_theta, then making P_theta ≈ P_r.
The standard GAN (Goodfellow 2014) does this with a discriminator minimax game whose
equilibrium minimizes the Jensen–Shannon divergence JS(P_r, P_theta). But GAN training
is notoriously unstable: vanishing generator gradients, mode collapse, need to carefully
balance D and G, no usable loss curve to monitor convergence.

## Load-bearing ancestors

### Original GAN (Goodfellow et al. 2014)
- Value: V(D,G) = E_{x~P_r}[log D(x)] + E_{z}[log(1 - D(g(z)))].
- For fixed G, optimal D* = P_r/(P_r + P_g). Plug back ⇒ the generator minimizes
  2·JS(P_r,P_g) − 2 log 2. So GAN ≡ minimize JS at the inner optimum.
- "−log D" trick: instead of min E[log(1−D(G(z)))] (saturates early), maximize
  E[log D(G(z))] to get stronger gradients early.
- Gap: JS objective behaves terribly when supports don't overlap (see below).

### Towards Principled Methods (Arjovsky & Bottou 2017, arXiv:1701.04862) — THE diagnosis
This is the prequel; its theorems are the motivation.
- Real data lies on a low-dim manifold; g_theta(Z) with z low-dim and g smooth also
  produces a measure supported on a countable union of low-dim manifolds. Two such
  manifolds in general position intersect on a measure-zero set ⇒ P_r and P_g are
  effectively disjoint / overlap on measure zero.
- Perfect-discrimination theorems: if supports are disjoint (or intersect on a
  measure-0 set), there EXISTS a smooth D* with accuracy 1 and ∇_x D* = 0 on both
  supports. The discriminator can become perfect.
- Divergence theorem: under that disjointness, JS(P_r,P_g)=log 2 (constant), KL=+∞ both
  ways. So JS is constant ⇒ ∇_theta JS = 0.
- Vanishing-gradients theorem: as ||D − D*|| → 0, the generator gradient
  ∇_theta E[log(1−D(g(z)))] → 0 (bounded by M·ε/(1−ε)). "Either D updates are inaccurate
  or they vanish." THIS is why you must keep D weak — a fundamental bind.
- −log D analysis: E[−∇_theta log D*(g(z))] = ∇_theta[ KL(P_g||P_r) − 2 JS(P_g||P_r) ].
  The KL is the reverse (mode-seeking, huge cost for fake-looking samples, ~no cost for
  mode dropping ⇒ explains mode collapse), and the −2JS term has the WRONG sign (pushes
  distributions apart). Plus the optimal D* is a singular ratio that doesn't exist when
  supports are disjoint ⇒ as D improves toward it the gradient has exploding variance
  → massively unstable updates. Empirically gradient norms blow up as D trains.
- Proposed fix in that paper: add noise to both P_r and P_g to make them absolutely
  continuous with overlapping support, smoothing JS. WGAN is the cleaner alternative.

### f-divergences / f-GAN (Nowozin et al. 2016)
- GANs can minimize any f-divergence via a variational lower bound. But ALL f-divergences
  share the disjoint-support pathology: they depend on the density ratio dP_r/dP_g, which
  is ill-defined / saturated when supports don't overlap. So "use a different
  f-divergence" doesn't escape the problem. Need a fundamentally different geometry.

### IPMs / MMD / EBGAN (context for Related Work)
- Integral Probability Metric: d_F(P,Q) = sup_{f∈F} E_P f − E_Q f. Different F ⇒
  different metric. F = 1-Lipschitz ⇒ Wasserstein-1 (Kantorovich–Rubinstein). F =
  functions bounded in [−1,1] ⇒ total variation (same bad topology as JS). MMD: F = unit
  ball of an RKHS (kernel trick, no separate net, but O(samples^2) and needs huge batches
  in high dim). EBGAN ≈ total variation (Appendix: optimal energy D gives L_G = (m/2)·δ).
  So among IPMs, the Lipschitz ball is the one giving a weak, well-behaved topology.

### DCGAN (Radford et al. 2015)
- The convolutional architecture (transpose-conv generator, strided-conv discriminator,
  batchnorm, ReLU/LeakyReLU) that became the standard GAN backbone. WGAN reuses it as the
  generator/critic body — the change is the objective and training loop, not the net.

## The core derivation

### Distances and the parallel-lines example
TV δ = sup_A |P_r(A)−P_g(A)|; KL = ∫ log(P_r/P_g) P_r; JS = ½KL(P_r||P_m)+½KL(P_g||P_m),
P_m=(P_r+P_g)/2; EM/Wasserstein-1:
  W(P_r,P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ} ||x−y||,
Π = couplings with the right marginals. γ = optimal transport plan.

Parallel lines: P_0 = (0,Z), P_theta = (theta,Z), Z~U[0,1]. Then
- W(P_0,P_theta) = |theta|  (move each point horizontally by |theta|).
- JS = log 2 if theta≠0 else 0;  KL = +∞ if theta≠0 else 0;  TV = 1 if theta≠0 else 0.
So as theta_t→0, only W→0 continuously. JS/KL/TV are discontinuous at 0 ⇒ no gradient.
This is exactly the disjoint-support case. W gives ∇ = sign(theta) everywhere ≠ 0.

### Why W is "weaker" / continuous (Theorem 1 + Corollary)
If g_theta is continuous in theta then W(P_r,P_theta) is continuous; if g locally
Lipschitz with E_z[L(theta,z)]<∞ (Assumption 1) then W is continuous everywhere &
differentiable a.e. Proof: use coupling γ = law of (g_theta(Z), g_{theta'}(Z)), so
  W(P_theta,P_theta') ≤ E_z||g_theta(z) − g_theta'(z)||;
continuity of g + bounded convergence (X compact ⇒ bound M) ⇒ →0; local Lipschitz ⇒
≤ L(theta)||theta−theta'||; reverse triangle |W(P_r,P_theta)−W(P_r,P_theta')| ≤
W(P_theta,P_theta'); Rademacher ⇒ diff a.e. Corollary: any feedforward net with smooth
Lipschitz nonlinearities + E||z||<∞ satisfies Assumption 1 (the layer-product Jacobian
bound), so W is a legitimate, almost-everywhere-differentiable loss. None of this holds
for JS/KL (parallel lines is the counterexample).

### Topology ordering (Theorem 2)
KL → JS ↔ TV → W (KL strongest, W weakest). Proven via: δ(P_n,P)→0 ⇔ JS→0 (Radon–Nikodym
bound both directions + Pinsker δ ≤ 2√JS); W→0 ⇔ convergence in distribution (Villani);
KL→0 ⇒ δ→0 (Pinsker); δ→0 ⇒ W→0. So W converges in strictly more cases ⇒ easier to make
theta↦P_theta continuous ⇒ a real loss.

### Kantorovich–Rubinstein duality (the key move that makes W tractable)
The inf over couplings is intractable (search over all joint distributions). KR duality:
  W(P_r,P_g) = sup_{||f||_L ≤ 1} E_{x~P_r}[f(x)] − E_{x~P_g}[f(x)],
sup over all 1-Lipschitz f. Replacing ||f||_L≤1 by ≤K scales by K. Intuition for the
duality: the transport LP has a dual; the dual variable is a "potential" f and the
transport-cost constraint ||x−y|| on the primal becomes the Lipschitz constraint
|f(x)−f(y)|≤||x−y|| on the dual. (Full proof is in Villani; we use it as a known OT fact.)

### From duality to an algorithm (Theorem 3 + clipping)
Parameterize f by a neural net f_w with w in a compact set W (so all f_w are K-Lipschitz
for some K depending only on W). Solve
  max_w  E_{x~P_r}[f_w(x)] − E_{z}[f_w(g_theta(z))]    ≈ K·W(P_r,P_theta).
Gradient of W w.r.t. theta: with f the optimal critic (fixed), envelope theorem ⇒
  ∇_theta W(P_r,P_theta) = −E_z[∇_theta f(g_theta(z))].
(Theorem 3 proof: V(f,theta)=E_r f − E_z f(g_theta); KR gives a maximizer f∈X*(theta);
Milgrom–Segal envelope thm ⇒ ∇_theta W = ∇_theta V(f,theta) for any optimal f; then swap
∇ and E via Rademacher (f∘g Lipschitz, diff a.e.) + dominated convergence with dominator
2L(theta_0,z), Fubini to fix a good theta_0.)

Enforcing the Lipschitz constraint: keep w in a box W=[−c,c]^l by CLIPPING w←clip(w,−c,c)
after every critic update. c=0.01. Compact box ⇒ uniformly K-Lipschitz family. The paper
is explicit that this is "a clearly terrible way" — too-large c ⇒ slow to reach the limit,
hard to train critic to optimality; too-small c ⇒ vanishing gradients through many layers.
Tried projecting to a sphere, little difference; kept clipping for simplicity. (Later GP
exists but is OUT OF FRAME — derivation time only knows clipping.)

## Design decisions → why (with rejected alternatives)

1. Objective = E[f(real)] − E[f(fake)] with f 1-Lipschitz (not log-D classifier).
   Why: it IS the KR dual of W. Linear in f ⇒ no sigmoid/log ⇒ never saturates ⇒ gradient
   to G stays alive even when the critic is excellent. Rejected: JS/log-D objective →
   saturates to log 2 on disjoint supports (vanishing-grad theorem).

2. Lipschitz constraint via weight clipping to [−c,c], c=0.01.
   Why: KR needs f∈{||f||_L≤K}; compact weight box ⇒ uniformly K-Lipschitz. Simplest thing
   that works. Rejected: sphere projection (no gain); nothing principled known at the time.
   Known cost: c too big = slow critic; c too small = vanishing grads.

3. Train critic to optimality: n_critic = 5 critic steps per generator step (and 100 in
   warmup/every 500 iters). Why: gradient identity ∇_theta W = −E[∇_theta f] only holds at
   the optimal f; the better the critic, the more accurate the W estimate and its gradient.
   Because W is continuous & diff a.e., a better critic HELPS (opposite of JS, where a
   better D kills the gradient). This removes the D/G balancing act.

4. Rename discriminator → "critic". Why: it is no longer a classifier outputting a
   probability; it is a real-valued function whose scaled difference estimates W. No
   sigmoid on the output.

5. RMSProp, lr 5e-5, no momentum (β1=0). Why: the critic loss is highly non-stationary;
   with Adam (β1>0) training blew up — when it did, cos(Adam step, gradient) went negative.
   RMSProp handles non-stationary objectives (A3C). Rejected: Adam/high lr ⇒ instability.

6. Critic output = mean over batch of scalar (no sigmoid); loss implemented by
   backward(one) on real and backward(mone) on fake ⇒ maximizes E[f(real)]−E[f(fake)];
   generator backward(one) on E[f(fake)] ⇒ minimizes −E[f(fake)] = the W gradient.

7. Architecture = DCGAN body reused for both G and critic. Why: the contribution is the
   objective+loop, not the net; reusing DCGAN isolates the change and shows WGAN is robust
   to architecture (works even without batchnorm / as an MLP, where standard GAN fails).

8. Meaningful loss curve: E[f(real)]−E[f(fake)] tracks W up to the constant K ⇒ lower loss
   correlates with better samples — a real validation/debug signal GANs never had.

9. Mode collapse impossible when critic is trained to optimality: mode collapse arises
   because for a FIXED discriminator the optimal generator collapses to deltas at the
   argmax of D; if we instead optimize the critic to optimality each step there is no fixed
   target to collapse onto.

## Canonical implementation (martinarjovsky/WassersteinGAN, PyTorch)
- main.py training loop: per outer step, run Diters critic updates: clamp params to
  [clamp_lower,clamp_upper]; errD_real=netD(real).backward(one); fake=netG(noise) detached;
  errD_fake=netD(fake).backward(mone); errD=errD_real−errD_fake; optimizerD.step(). Diters
  =100 if gen_iter<25 or %500==0 else 5. Then G update: freeze D; fake=netG(noise);
  errG=netD(fake).backward(one); optimizerG.step(). RMSprop lr 5e-5 default; --adam optional.
- netD (DCGAN_D / MLP_D): same conv/MLP body as DCGAN but NO sigmoid; forward returns
  output.mean(0).view(1) — a scalar critic value.
- netG: standard DCGAN transpose-conv or 4-layer ReLU MLP (512 units in paper).
- nz=100, batch 64, image 64, weights init N(0,0.02).

## In-frame reminders for Phase 2
- Never name "this paper"/authors/arXiv. May name the method "WGAN" (mainly answer.md) as
  the thing being built. Cite ancestors (Goodfellow 2014, Arjovsky&Bottou 2017, Nowozin
  2016, Radford 2015, Villani, Müller, Gretton) freely.
- No proposed-method eval results (no LSUN sample-quality wins). Motivating diagnostics
  (vanishing-grad theorem, parallel-lines, gradient blowup with −logD) ARE in scope.
- Gradient-penalty / improved-WGAN / later work is posterior ⇒ excluded.

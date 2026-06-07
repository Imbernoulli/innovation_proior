# Research notes — GAN (Goodfellow et al. 2014, arXiv 1406.2661)

Primary paper read in full (adversarial.tex). It is a 6-page NIPS paper with NO separate
appendix; all proofs are inline in Section 4 (Theoretical Results). Verified directly:

- Value function Eq.1: min_G max_D V = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1 - D(G(z)))].
- Optimal D (Prop.1): D*_G(x) = pdata(x) / (pdata(x) + pg(x)). Proof: for fixed G, integrand
  per x is a·log y + b·log(1-y) with a=pdata, b=pg, maximized over y in [0,1] at a/(a+b).
- C(G) = max_D V = E_pdata[log D*] + E_pg[log(1-D*)].
- Theorem 1: global min of C(G) iff pg=pdata, value -log4. Proof: subtract
  E_pdata[-log2]+E_pg[-log2] = -log4 from C(G) to get
  C(G) = -log4 + KL(pdata || (pdata+pg)/2) + KL(pg || (pdata+pg)/2) = -log4 + 2·JSD(pdata||pg).
  JSD >= 0, =0 iff equal.
- Convergence (Prop.2): U(pg,D) convex in pg; subgradient of sup of convex functions includes
  the gradient at the argmax; sup_D U convex with unique optimum -> small steps converge.
- Non-saturating trick: instead of G minimizing log(1-D(G(z))) (saturates when D confident
  early), G maximizes log D(G(z)). Same fixed point, stronger early gradients.
- Algorithm 1: k D-steps (k=1 used) then 1 G-step, minibatch SGD with momentum.
- Helvetica scenario = mode collapse: G collapses many z to same x if trained too far w/o D update.

## Load-bearing ancestors (from .bbl) and their gap

- RBMs (Smolensky86, Hinton06), DBMs (Salakhutdinov+Hinton 2009): undirected, energy-based.
  p(x) = exp(-E)/Z. Partition function Z and its gradient intractable; need MCMC (CD/PCD).
  Mixing between modes is the pain point (Bengio ICML2013/2014). GAN sidesteps Z entirely.
- DBNs (Hinton06): hybrid undirected top + directed layers; fast greedy layerwise training,
  but inherits difficulties of BOTH directed and undirected models.
- Score matching (Hyvarinen 2005): fits model up to normalization by matching gradient of
  log-density (avoids Z), but REQUIRES analytically specified unnormalized density.
- NCE (Gutmann+Hyvarinen 2010): turns density estimation into logistic regression discriminating
  data from a FIXED noise distribution; learns Z as a parameter. Closest ancestor in spirit —
  "discriminative criterion to fit a generative model". Two gaps: (1) requires unnormalized
  density analytically specified up to Z; (2) noise is FIXED, so once model approximates data
  on a small subset, the classification task gets easy and learning slows dramatically.
  GAN's key move: replace fixed noise with the generator itself, and replace the unnormalized
  density with a separate discriminator net (no analytic density needed at all).
- Denoising autoencoders (Vincent 2008), contractive AE: learning rules ~ score matching on RBMs;
  define implicit models.
- GSN (Bengio ICML2014) / generalized denoising AE (Bengio NIPS2013): train a generative MACHINE
  (sampler) directly, parameterizing one step of a Markov chain, trainable by backprop. Gap:
  still needs a Markov chain to sample; feedback loop limits use of piecewise-linear units.
- VAE / auto-encoding variational Bayes (Kingma+Welling 2014) and stochastic backprop
  (Rezende 2014): train a generator by backprop via the reparameterization trick, maximize a
  variational lower bound (ELBO) with an approximate inference (encoder) network. Same era,
  also backprop-into-a-generator, but still likelihood-based (variational bound) and needs an
  inference network; samples tend to be blurry. GAN needs no inference net and no explicit
  likelihood.
- SML/PCD (Younes 1999, Tieleman 2008): persistent Markov chain across learning steps — the
  analogy Goodfellow uses to justify k=1 D-step (keep D near optimum as G moves slowly).
- Wake-sleep (Hinton 1995): inference net learned to invert generator — cited as way to add
  learned approximate inference as a GAN extension.

State of field at time (2014): supervised deep nets booming (AlexNet, dropout, ReLU/maxout,
backprop). Deep GENERATIVE models lagging because of intractable partition functions /
intractable inference and reliance on MCMC mixing. Prevailing wisdom = either pay for MCMC
(Boltzmann family) or optimize a variational bound with an inference net (VAE). GAN's pitch:
sidestep both — an implicit model, sampled by pure forward prop, trained by pure backprop,
with the loss supplied by a learned adversary instead of an explicit likelihood.

## Canonical code

- Official: github.com/goodfeli/adversarial (Theano + Pylearn2). Heart = class AdversaryCost2
  in __init__.py. d_obj = 0.5*(cost(y1, D(X)) + cost(y0, D(G(z)))); the non-saturating
  generator objective is g_obj = cost(y1, D(G(z))) — i.e. label the FAKE samples as REAL (1)
  and minimize BCE => maximize log D(G(z)). Confirms the paper's non-saturating trick is what
  the released code actually uses (not log(1-D)). G is MLP (ReLU+sigmoid), D is maxout+dropout.
- Clean PyTorch reference for Phase 2 code: eriklindernoren/PyTorch-GAN implementations/gan/gan.py.
  G: Linear blocks (LeakyReLU, BatchNorm) -> Tanh. D: Linear -> LeakyReLU -> Sigmoid.
  adversarial_loss = BCELoss. g_loss = BCE(D(G(z)), valid=1)  [non-saturating].
  d_loss = (BCE(D(real), 1) + BCE(D(G(z).detach()), 0))/2. z ~ N(0,1), Adam lr=2e-4, betas(0.5,0.999).

## V3 DEPTH pass — additional research (NIPS 2016 GAN tutorial intuitions + explainers)

Sources read: Goodfellow NIPS 2016 Tutorial (arXiv 1701.00160) abstract+summaries; Daniel Takeshi
"Understanding GANs" blog; Toronto CSC2541 GAN-foundations slides; deep-generative-models GAN tutorial
notes; "Learning in Implicit Generative Models" (Mohamed&Lakshminarayanan 1610.03483) framing; density-ratio
perspective (1610.02920). Used only to UNDERSTAND the why; re-derived in-frame, no posterior citation.

DESIGN-DECISION -> WHY table (each reconstructed in-frame in reasoning.md):

1. Implicit generator x=G(z) (no explicit density) — generative-model taxonomy splits into explicit-density
   (must write p(x): FVBN tractable but slow/sequential; Boltzmann needs Z; VAE needs a tractable bound) vs
   implicit (only need to SAMPLE). Choosing implicit *by construction* sidesteps the partition function Z and
   the analytic-tractability constraint entirely: you never normalize, never integrate, just push noise through G.

2. Learned discriminator vs fixed divergence/MMD/NCE-fixed-noise — a fixed two-sample statistic (MMD) or a
   fixed contrast distribution (NCE) gives a STATIC task: once G is approximately right the signal goes slack.
   A trained D supplies an *adaptive* objective that re-sharpens as G improves; the optimal D recovers the
   density ratio pdata/(pdata+pg) so D is literally estimating the (transformed) ratio rather than us having to
   compute it (which would require the intractable densities, or clip an unstable raw ratio that ranges 0..inf).

3. Minimax V = E[log D(x)] + E[log(1-D(G(z)))] — it is exactly the binary-cross-entropy of the optimal classifier
   for the real-vs-fake label; D maximizes its log-likelihood of the correct label, G minimizes it. Pointwise
   max of a log y + b log(1-y) -> D* = a/(a+b) = pdata/(pdata+pg).

4. -log4 + 2 JSD — add/subtract -log4 = E_pdata[-log2]+E_pg[-log2], fold log2 into the logs to get the two KLs
   to the mixture (pdata+pg)/2, recognize 2*JSD. JSD>=0, =0 iff equal -> unique global optimum pg=pdata.
   Symmetric divergence emerges because the real-vs-fake problem is symmetric; finite even on disjoint supports.

5. Convergence: U(pg,D) linear hence convex in pg for fixed D; sup over D of convex funcs is convex; subgradient
   of the argmax-attaining member is a subgradient of the sup -> evaluating grad at D* and stepping pg is valid
   subgradient descent on a convex functional with unique optimum. Caveat: real optimization is over theta_g via
   MLP -> nonconvex parameter space, multiple critical points; guarantee only holds in distribution space.

6. Non-saturating loss — minimax G-term log(1-D(G(z))): d/dD log(1-D) = -1/(1-D) ~ -1 near D=0, and the curve is
   FLAT near D=0 so gradient through G is tiny exactly when G is losing (D confidently rejects, D(G(z))~0).
   Non-saturating: G maximizes log D(G(z)); d/dD log D = 1/D -> blows up as D->0, strong gradient when losing.
   Same fixed point (both push D(G(z))->1). DISCREPANCY: Algorithm-1 box in the writeup descends log(1-D(G(z)))
   (minimax), but the released goodfeli code's g_obj = cost(y1=REAL, D(G(z))) = -log D(G(z)) (non-saturating).

7. k D-steps per G-step — keep D near D* so the gradient handed to G is meaningful (theory wants D*); optimizing
   D to completion is prohibitive and overfits on finite data; carry D's params across steps like SML/PCD
   persistent chains. k=1 cheapest and works.

8. Noise prior z + differentiable G — z is the only stochasticity; because G is a differentiable deterministic
   map of z, backprop carries D's gradient through G's outputs into theta_g (reparameterization-style). Noise
   only at G's bottom layer in practice (theory permits noise at all layers). Uniform/gaussian/spherical noise.

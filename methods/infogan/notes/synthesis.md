# InfoGAN synthesis (Phase 1.5)

Verified: arXiv 1606.03657, "InfoGAN: Interpretable Representation Learning by Information Maximizing GANs", Chen, Duan, Houthooft, Schulman, Sutskever, Abbeel (NIPS 2016).
Canonical impl: original openai/InfoGAN (TensorFlow), saved in `code/openai_InfoGAN/`. The reference trainer uses standard log GAN losses with a non-saturating generator update and subtracts the MI estimate from both the discriminator/Q loss and the generator loss. The appendix describes a diagonal-Gaussian recognition model for continuous codes; the original MNIST launcher uses fixed std for those continuous codes.

## Pain point / research question
- Unsupervised representation learning wants a DISENTANGLED representation: latent dims each map to ONE semantic factor of variation (digit identity, rotation, width, pose, lighting, glasses...).
- Prior disentangling methods (bilinear models, multi-view perceptron, DC-IGN, disBM, adversarial autoencoders) all need SUPERVISION or weak supervision (clamping, grouping, known factors). Only hossRBM is fully unsupervised but only handles discrete factors and its cost grows exponentially in #factors.
- GAN's input noise z is a single unstructured vector; G is free to use it in a highly ENTANGLED way, so no individual z-dim corresponds to a semantic feature. Want to fix that without supervision.

## Background needed
- GAN (Goodfellow 2014): min_G max_D V(D,G) = E_{x~pdata}[log D(x)] + E_{z~pnoise}[log(1-D(G(z)))]. Optimal D = pdata/(pdata+pG).
- DCGAN (Radford 2015): the stable conv architecture used as the base; "enough to stabilize InfoGAN training, did not introduce new trick."
- Mutual information: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X). I=0 iff independent; maximal if deterministic invertible relation. Interpretation: reduction in uncertainty about X from observing Y.
- Variational Information Maximization (Barber & Agakov 2003): the technique for lower-bounding MI with an auxiliary distribution Q.
- Wake-Sleep (Hinton et al. 1995), Helmholtz machine (Dayan et al. 1995): the interpretation frame in appendix B.

## Core idea (insight-before-method)
1. Split G's input into incompressible noise z and a structured latent code c = (c_1,...,c_L), with factored prior P(c_1,...,c_L)=∏ P(c_i). G becomes G(z,c).
2. Problem: standard GAN lets G IGNORE c (it can satisfy P_G(x|c)=P_G(x), so c is a "trivial code"). Need to force G to USE c.
3. Information-theoretic regularizer: make the mutual information I(c; G(z,c)) HIGH — if c and the output share lots of information, c can't be ignored, and the info in c must be preserved in the generation.
4. Regularized minimax: min_G max_D V_I(D,G) = V(D,G) - λ I(c; G(z,c)).

## The central difficulty and the variational lower bound (THE math)
I(c; G(z,c)) requires the posterior P(c|x), which is intractable (need to evaluate, and to sample from it). Fix: Variational Information Maximization — introduce auxiliary distribution Q(c|x) approximating P(c|x).

Derivation (Eq 4), let x = G(z,c):
I(c; G(z,c)) = H(c) - H(c | G(z,c))
             = E_{x~G(z,c)}[ E_{c'~P(c|x)}[log P(c'|x)] ] + H(c)
             = E_{x~G(z,c)}[ D_KL(P(·|x) ‖ Q(·|x)) + E_{c'~P(c|x)}[log Q(c'|x)] ] + H(c)
             ≥ E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ] + H(c).
The inequality is because D_KL ≥ 0. So the bound is TIGHT when Q = P(·|x), i.e. E_x[D_KL]→0.

Self-check of the KL insertion: H(c|x) = -E_{c'~P(c|x)}[log P(c'|x)]. So -H(c|x) = E[log P(c'|x)]. Add and subtract log Q: E[log P(c'|x)] = E[log Q(c'|x)] + E[log P/Q] = E[log Q(c'|x)] + D_KL(P‖Q). Yes — log P - log Q = log(P/Q), and E_{c'~P}[log(P/Q)] = D_KL(P‖Q) ≥ 0. Correct.

Remaining problem in this form: still need to SAMPLE c' from the posterior P(c|x) in the inner expectation. Removed via Lemma 5.1 (proved in Appendix A.1).

Lemma A.1: For r.v. X,Y and function f(x,y) under suitable regularity:
  E_{x~X, y~Y|x}[f(x,y)] = E_{x~X, y~Y|x, x'~X|y}[f(x',y)].
Proof (Eq 7): E_{x~X,y~Y|x}[f(x,y)] = ∫_x P(x) ∫_y P(y|x) f(x,y) dy dx
  = ∫∫ P(x,y) f(x,y) dy dx
  = ∫∫ P(x,y) f(x,y) [∫_{x'} P(x'|y) dx'] dy dx   (the bracket = 1)
  = ∫_x P(x) ∫_y P(y|x) ∫_{x'} P(x'|y) f(x',y) dx' dy dx
  = E_{x~X, y~Y|x, x'~X|y}[f(x',y)].  QED.

Applying it lets us define the variational lower bound L_I(G,Q) (Eq 5) that needs NO posterior sampling:
  L_I(G,Q) = E_{c~P(c), x~G(z,c)}[log Q(c|x)] + H(c)
           = E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ] + H(c)
           ≤ I(c; G(z,c)).
Now L_I is sampled by: draw c~P(c), z~noise, x=G(z,c), evaluate log Q(c|x). Monte-Carlo simple. Maximize L_I w.r.t. Q directly and w.r.t. G via reparametrization (c and z are sampled from fixed priors, x=G(z,c) differentiable). H(c) treated as CONSTANT (fix the latent-code prior; though it could be optimized since common dists have analytic entropy).

Bound max: for finite discrete codes, L_I(G,Q) = H(c) when the bound is tight and the generated sample determines c; do not state this as an unconditional continuous-code maximum.

Final objective (Eq 6): min_{G,Q} max_D V_InfoGAN(D,G,Q) = V(D,G) - λ L_I(G,Q).

## Implementation (appendix C + canonical code) — design decisions
- Q parameterized as a neural net that SHARES all convolutional layers with D, plus ONE final FC layer to output the params of Q(c|x). So InfoGAN adds negligible compute over GAN. (Q and D share body; D has its own real/fake output head, Q has its own code head.)
- Categorical c_i: softmax nonlinearity → Q(c_i|x) is a categorical; log Q = -cross-entropy. So maximizing L_I term = minimizing cross-entropy between Q's softmax and the sampled one-hot code. λ=1 sufficient for discrete codes.
- Continuous c_j: parameterize Q(c_j|x) as a diagonal/factored Gaussian; the recognition net outputs mean and std, std via exp-transform of the output to ensure positivity. Maximizing log Q = minimizing Gaussian NLL; with fixed std this reduces to MSE plus constants. Smaller λ can be used for continuous codes so λL_I (which involves differential entropy) is on the same scale as V.
- DCGAN architecture: up-convolutional G, leaky-ReLU (rate 0.1) in D hidden layers, ReLU in G, batchnorm after most layers, Adam, lr 2e-4 for D and 1e-3 for G typically, λ=1 default.
- L_I converges faster than the GAN objective → "essentially comes for free."

## Wake-Sleep / "Sleep-Sleep" interpretation (appendix B)
InfoGAN as a Helmholtz machine: P_G(x|c) generative, Q(c|x) recognition.
Wake-Sleep: wake phase max_G E_{x~Data, c~Q(c|x)}[log P_G(x|c)]; sleep phase max_Q E_{c~P(c), x~P_G(x|c)}[log Q(c|x)] (dream samples from generator, not real data).
Optimizing L_I w.r.t. Q is EXACTLY the sleep-phase update. But InfoGAN ALSO optimizes L_I w.r.t. G (forcing G to use c) — which is again a sleep-phase-like update (samples from G, not data). So InfoGAN ≈ "Sleep-Sleep". This highlights InfoGAN's difference: G is explicitly encouraged to convey info in c.

## Experiment latent configs (for grounding numbers; experiments are method-results, keep as forward intent)
- MNIST: 1 categorical Cat(K=10,p=0.1) + 2 continuous Unif(-1,1) + 62 noise = dim 74.
- 3D Faces: 5 continuous Unif(-1,1), 128 noise = 133.
- 3D Chairs: 1 continuous + 3 discrete (dim 20 each) + 128 noise = 189.
- SVHN: 4 ten-dim categorical + 4 continuous + 124 noise = 168.
- CelebA: 10 ten-dim categorical + 128 noise = 228.

## Baselines (prior art to elaborate)
- GAN (Goodfellow 2014): base; entangled noise.
- DCGAN (Radford 2015): stable arch; learns image rep supporting linear algebra in code space but not disentangled by design.
- DC-IGN (Kulkarni 2015): SUPERVISED disentangling, clamping graphics codes for pose/light in 3D. Needs supervised grouping.
- disBM (Reed 2014): higher-order Boltzmann, clamping for known-matching factors. Weak supervision.
- hossRBM (Desjardins 2012): only fully-unsupervised disentangler before; discrete only, exponential cost in #factors.
- Bilinear models (Tenenbaum & Freeman 2000), multi-view perceptron (Zhu 2014), Yang 2015 recurrent, adversarial autoencoders (Makhzani 2015), semi-sup VAE (Kingma 2014): need labels.
- Variational Information Maximization (Barber & Agakov 2003): the lower-bound tool.

## Design-decision → why table
- Split input into z (incompressible noise) + c (structured code) → want some inputs to carry semantics.
- Factored prior P(c)=∏P(c_i) → want each code independent → each can capture an independent factor.
- Add -λ I(c;G(z,c)) regularizer → without it G ignores c (trivial code P_G(x|c)=P_G(x)); high MI forces c to be recoverable from x, hence used.
- Use a LOWER BOUND L_I not I directly → I needs intractable posterior P(c|x).
- Auxiliary Q approximating P(c|x) → variational info max; bound tight as Q→P.
- Lemma A.1 → removes posterior sampling so L_I is Monte-Carlo-able from priors + G.
- H(c) constant → fix the code prior; simpler (could optimize but not needed).
- Q shares D's conv body + 1 FC head → negligible extra compute; the features D learns are exactly what's needed to recover c.
- Softmax Q for categorical → log Q = -CE; natural for 1-of-K.
- Gaussian Q for continuous, std via exp → log Q = -NLL; exp ensures positive std.
- λ=1 discrete, smaller for continuous → continuous L_I involves differential entropy, scale-match to V.
- DCGAN base, no new trick → GAN hard to train; reuse known-stable arch.
```

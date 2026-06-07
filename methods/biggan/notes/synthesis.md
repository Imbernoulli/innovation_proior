# BigGAN synthesis — design-decision → why (pre-Phase-2 notes)

## The pain point / research question
By ~2018, class-conditional ImageNet GANs (SA-GAN) reach IS ~52 vs 233 for real data. The gap to real images in *both* fidelity and variety is huge. GAN training is notoriously brittle: it doesn't "just scale" like supervised learning. Want: high-fidelity, diverse, class-conditional 128–512px ImageNet samples. Central question: GANs collapse/destabilize at high resolution & large batch; how do I get there?

## Load-bearing ancestors (the toolbox available at derivation time)
- **Vanilla GAN (Goodfellow 2014).** min_G max_D E_x[log D(x)] + E_z[log(1−D(G(z)))]. Two-player minimax → Nash. Brittle; saturating G gradient → non-saturating −log D(G(z)). Z drawn from N(0,I) or U[−1,1] arbitrarily.
- **DCGAN (Radford 2016).** Conv G/D, the standard backbone; default init N(0,0.02).
- **Conditional GAN (Mirza 2014).** Feed one-hot y to both G and D (concat). AC-GAN (Odena 2017): concat y to z in G + auxiliary classifier loss in D. Limitation: auxiliary classifier encourages easily-classified (low-variety) samples; concatenation is a weak way to inject class.
- **Conditional BatchNorm / FiLM (Dumoulin 2017 artistic style; de Vries 2017; Perez 2018 FiLM).** Replace BN's single (γ,β) with *per-class* (or per-conditioning) gains and biases. FiLM: a general conditioning layer producing feature-wise affine (γ,β) from conditioning input. This is how class info enters G cleanly: BN(h)·γ(y)+β(y).
- **Projection discriminator (Miyato & Koyama 2018).** D*(x,y)=σ(f), f=log q(x,y)/p(x,y) = log q(y|x)/p(y|x) + log q(x)/p(x). Assume conditional class posteriors are log-linear in feature φ(x): then log-ratio of categoricals = (v_y^q − v_y^p)·φ(x) + (x-only). ⇒ f(x,y)= y^T V φ(x) + ψ(φ(x)): inner product of a class embedding with D's feature + unconditional term. Principled vs concat.
- **Spectral Normalization (Miyato 2018 SN-GAN).** Bound D's Lipschitz constant by dividing each weight by its top singular value σ0(W), estimated by 1 step of power iteration with a persisted u. Keeps D's gradients bounded everywhere → better signal to G. SA-GAN extends SN into G too.
- **Self-Attention GAN (Zhang 2018).** Adds the non-local/self-attention block (Wang 2018) to G and D to model long-range/global structure that stacked small convs can't; uses SN in G & D; uses hinge loss; uses TTUR (different G/D learning rates). This is the BigGAN baseline.
- **Hinge loss (Lim 2017 Geometric GAN; Tran 2017).** D: E[relu(1−D(x))] + E[relu(1+D(G(z)))]; G: −E[D(G(z))]. Margin-based; gives zero gradient on confidently-correct examples (a fact that becomes central in the collapse analysis). Pairs naturally with SN (bounded output scale makes the margin meaningful).
- **IS (Salimans 2016).** exp(E_x KL(p(y|x) || p(y))) via an Inception classifier — rewards confident per-sample class posterior (fidelity/"objectness") and marginal diversity. Does NOT penalize intra-class lack of variety.
- **FID (Heusel 2017).** ||μ_r−μ_g||² + Tr(Σ_r+Σ_g−2(Σ_rΣ_g)^{1/2}) on Inception pool features; penalizes both fidelity and variety (mode dropping).
- **Orthogonal init (Saxe 2014).** Orthogonal weight init keeps signals isometric, dynamical isometry → trainable deep nets.
- **Orthogonal regularization (Brock 2017, Neural Photo Editing).** R(W)=β||W^T W − I||²_F — push filters toward orthonormal; keeps the map well-conditioned / near norm-preserving.
- **EMA of G weights (Karras 2018 ProGAN; Yazici 2018; Mescheder 2018).** Average G's weights over training for sampling → smoother, better samples.
- **R1 gradient penalty (Mescheder 2018).** (γ/2) E_{p_data}||∇D(x)||² — zero-centered GP on real data only; provably aids convergence.
- **ProGAN (Karras 2018).** Progressive growing for high-res single-class. BigGAN will show this is unnecessary if you just scale.
- **Generator conditioning (Odena 2018).** GAN performance correlates with the conditioning (Jacobian singular values) of G — motivates monitoring singular values.

## The three contributions, DERIVED
### 1. Scaling
- **Batch ×8 (256→2048):** each batch covers more modes → better gradient estimates for both nets. Empirically IS +46%, fewer iters to converge. *Side effect:* faster but unstable → complete training collapse. (Report scores from just before collapse.)
- **Width +50% (ch 64→96), ~2× params:** more capacity vs dataset complexity → IS +21%. Depth ×2 did NOT help initially (only later via BigGAN-deep bottleneck blocks).
- **Cross-replica BN:** with batch split across 100s of TPU cores, per-device BN stats are computed on tiny shards; aggregate across all devices for correct statistics.
- **Standing statistics at sampling:** BN running averages depend on test batch size & break under EMA weights (running stats computed with non-averaged weights). Fix: run G forward ~100× over random z, accumulate means/vars → outputs invariant to batch size/device.

### 2. Architectural changes improving scalability / conditioning
- **Shared class embedding (vs per-BN-layer embedding):** a per-layer embedding for every cond-BN layer = huge weight count. Share one embedding, linearly project to each layer's (γ,β). −memory/compute, +37% training speed. (FiLM-style projection.)
- **Skip-z (hierarchical latents):** feed z not only to the first layer but to every res block. Split z into 20-D chunks (BigGAN), concat each chunk to shared class embedding → that block's BN (γ,β). Lets G use the latent to directly modulate features at all resolutions/levels of hierarchy. +~4% perf, +18% speed. (BigGAN-deep: concat whole z, no split.)
- z dimensionality: default 128; works down to z∈R^8, minimal drop at R^32 (class info supplements). Full dim 120/140/160 for 128/256/512 because more res blocks = more chunks.

### 3. Truncation trick + orthogonal regularization (DERIVE INLINE)
- **Truncation trick.** GANs (unlike VAE/flows) don't backprop through latents → free to use ANY prior, and any *sampling* distribution at test time, even different from training. Train with z~N(0,I); at test, resample any z component with |z|>threshold (truncated normal). As threshold→0, z→mode of prior; samples → mode of G's per-class output → higher fidelity, lower variety. *Why it works:* the prior's high-density region is where G is best-trained/best-modeled; staying near it trades variety for fidelity. Gives a post-hoc fidelity/variety knob → variety-fidelity curve (like precision/recall). IS (precision-like, no variety penalty) rises monotonically as you truncate; FID first improves (precision↑) then sharply worsens as variety collapses.
- **Why truncation needs G to be smooth → orthogonal reg.** Feeding z values from a region of latent space NOT seen at training is a distribution shift; many large models break under it → "saturation artifacts." Want: whole z-space maps to good outputs, i.e. G smooth/amenable. Enforce conditioning via Orthogonal Regularization.
  - Full form R_β(W)=β||W^T W − I||²_F (Brock 2017). Too limiting (constrains filter norms too, conflicts with what the net wants).
  - **Modified, off-diagonal-only:** R_β(W)=β||W^T W ⊙ (1−I)||²_F. Removes diagonal → minimizes only pairwise cosine similarity between filters (decorrelate directions) WITHOUT forcing unit norm. β=1e-4 (swept 1e-5…1e-2). Without ortho reg only 16% of models amenable to truncation; with it, 60%.
  - *Gradient form used in code* (direct, skips building the loss): grad = 2·(W Wᵀ ⊙ (1−I)) W ; param.grad += strength·grad. (default_ortho uses (W Wᵀ − I) W.) Note code orthogonalizes rows (W viewed as [out, −1], W Wᵀ is out×out), regularizing W Wᵀ rather than Wᵀ W; mathematically symmetric statement of the same "make filters orthonormal" idea.

## Collapse / stability analysis (DERIVE INLINE)
- Monitor top-3 singular values σ0,σ1,σ2 of each weight (Arnoldi/extended power iteration).
- **G:** most layers well-behaved, but a few (esp. the first, over-complete, non-conv linear) have σ0 that grows throughout training and *explodes at collapse*. Test causation: directly regularize σ0 toward fixed σ_reg or toward r·sg(σ1); or *clamp* via partial SVD: W = W − max(0, σ0 − σ_clamp) v0 u0ᵀ. These stop the σ0 explosion (and sometimes mildly help) but DO NOT prevent collapse ⇒ G conditioning is a *symptom indicator*, necessary-ish but insufficient. Turn to D.
- **D:** spectra noisy, σ0/σ1 well-behaved (~1, slow spectral decay), σ0 grows but only *jumps* at collapse (not explodes). Frobenius norms smooth ⇒ noise concentrated on top singular directions. Posit: adversarial dynamics — G periodically produces batches that strongly perturb D.
- If spectral noise causes instability, regularize D's Jacobian: R1 GP (γ/2)E_{p_data}||∇D(x)||². At γ=10 → stable but IS −45%. Even γ=1 (lowest non-collapsing) → IS −20%. Same tradeoff for ortho reg / dropout / L2 on D: enough penalty buys stability but at large performance cost.
- **D memorizes:** train acc >98%, val acc 50–55% (chance). D's job isn't to generalize but to distill data into a learning signal for G. Memorization explanation for D's spectral spikes: as D nears perfect memorization it gets less signal from real data — both vanilla and hinge loss give *zero gradient on confidently-correct examples*; real-data gradient attenuates → D drifts to a negative output bias → eventually misclassifies many reals → big corrective gradient → impulse-response spike. Suggested fixes: unbounded (Wasserstein) loss (unstable here even w/ GP); changing hinge margin (smaller hurts; larger up to ×3 doesn't fix; >×3 unstable); smaller D / dropout (hurts).
- **Interaction experiments:** freeze G → D stable (loss → 0). Freeze D → G immediately, dramatically collapses (maxes D's loss to >300). ⇒ D must stay optimal w.r.t. G throughout; stability is a property of the *interaction*, not G or D alone. But favoring D (higher lr / more steps) is insufficient to ensure stability and can cripple training. **Conclusion (the honest one):** you *can* force stability by strongly constraining D, but at a dramatic performance cost; better final performance comes from relaxing conditioning and letting collapse happen late (early stopping), by which point the model is good.

## Other concrete knobs → why
- **Hinge loss** (vs minimax/NS/WGAN): margin-based, well-behaved with SN; standard in SN-GAN/SA-GAN.
- **Projection D**: principled class conditioning (above).
- **Adam β1=0, β2=0.999**, constant lr; D lr 2e-4, G lr 5e-5 (halved from SA-GAN's 1e-4/4e-4, with 2 D steps per G step). Higher β1 cripples; raising lr mid-training → instant collapse.
- **2 D steps / G step**: best in sweep (1–6 tried). D must keep up with G.
- **EMA decay 0.9999** of G weights for sampling: smoother G.
- **Orthogonal init** (Saxe) instead of N(0,0.02)/Xavier.
- **ε raised 1e-8→1e-4** in BN & SN: TPU low-precision numerical stability.
- **No progressive growing** needed even at 512 — scaling alone suffices.
- **Non-local/attention block at one resolution** (64×64 at 128px; up one stage at 256). Multiple/other locations: no benefit, more cost.
- **3×3 convs, nearest-neighbor upsample** (bilinear hurt; 5×5 only tiny gain at big cost; dilation hurt).

## Canonical code mapping (BigGAN-PyTorch, Brock)
- `Generator.forward(z,y)`: if hier, split z into chunks, concat each to shared embed → per-block conditioning ys; first Linear → reshape 4×4×16ch; GBlocks (ResBlock up) interleaved with Attention; output BN-ReLU-Conv-Tanh.
- `GBlock`: h=act(bn1(x,y)); upsample; conv1; h=act(bn2(h,y)); conv2; +shortcut(1×1 conv if needed).
- `ccbn`: gain=(1+linear(y)); bias=linear(y); BN(x) (cross-replica/standing) ·gain+bias. gain centered 1, bias centered 0.
- `Discriminator.forward(x,y)`: DBlocks down + Attention; ReLU; global sum pool → h; out = linear(h) + sum(embed(y)·h, dim=1)  ← projection.
- `SN` / `power_iteration`: W_mat/σ0 via 1 power-iter step, persisted u, Gram-Schmidt for >1 sv.
- `losses.loss_hinge_dis/gen`. `utils.ortho` (off-diag), `default_ortho` (full). `ema`. `truncated_z_sample` = truncation·truncnorm.rvs(−2,2). `accumulate_standing_stats`.

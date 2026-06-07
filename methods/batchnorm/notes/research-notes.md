# Research notes — Batch Normalization (Ioffe & Szegedy, 2015, arXiv 1502.03167)

(Main agent read the full paper + appendix directly from src/arx.tex. These notes capture the lineage + explainer + code research used to ground reasoning.md / context.md / answer.md. No subagent dispatch tool was available in this environment, so research was done directly via WebSearch/WebFetch.)

## Load-bearing ancestors (from arx.bbl, filtered to what the method builds on / reacts to)

- **LeCun, Bottou, Orr, Muller — "Efficient BackProp" (1998)** [lecun-backprop]. The cited justification that networks converge faster when inputs are *whitened*: linearly transformed to zero mean, unit variance, and decorrelated. Crucially, LeCun shows mean/variance normalization alone speeds convergence *even when features are not decorrelated*. BN explicitly leans on this to justify per-dimension (non-joint) normalization. Limitation it leaves: it only normalizes the *network inputs* once, statically; it says nothing about the *internal* activations that drift during training. Source: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

- **Wiesler & Ney — "A convergence analysis of log-linear training" (2011)** [loglinear-training]. Also cited for the "whitened inputs converge faster" claim. Same gap: input-level, not internal.

- **Shimodaira — covariate shift (2000)** [covariate-shift]. Defines covariate shift: when the input distribution to a learning system changes between train and test. Handled classically by domain adaptation. BN's conceptual move: extend this notion from the whole system to *sub-networks / layers* → "internal covariate shift." Gap: classical covariate shift is a train/test mismatch, not a within-training drift.

- **Glorot & Bengio — "Understanding the difficulty of training deep feedforward networks" (2010)** [glorot-difficulty]. Careful initialization (Xavier) to keep activation/gradient variances stable across layers at init. Limitation: it only fixes the *starting* distribution; as training proceeds, parameters move and the distribution drifts again. BN is the dynamic counterpart.

- **Saxe, McClelland, Ganguli — "Exact solutions to nonlinear dynamics of learning in deep linear nets" (2013)** [iclr-dynamics]. Orthogonal init; singular values near 1 are good for gradient propagation. BN conjectures (Sec on higher LR) that BN drives layer Jacobian singular values toward 1.

- **Nair & Hinton — ReLU (2010)** [relu]. The standard workaround for saturation/vanishing gradients. BN positions itself as enabling even *saturating* nonlinearities (sigmoid) to train.

- **Srivastava et al. — Dropout (2014)** [dropout]. Stochastic regularizer (drop units with prob p). BN observes its minibatch-coupling acts as a regularizer that can replace/reduce Dropout.

- **Approaches BN reacts against — normalization outside the gradient step**: mean-normalized SGD [mean-normalized-sgd, Wiesler 2014], Raiko et al. 2012, Povey et al. 2014 (natural gradient), Desjardins & Kavukcuoglu (natural neural networks). These normalize/whiten but either modify the optimizer or do normalization as a separate step → the "b grows unbounded / model blows up" failure that BN's framing diagnoses.

- **Lyu & Simoncelli — divisive normalization (2008)** [lyu-simoncelli]. Normalizes per single example / across feature maps at a location. BN's objection: this discards the absolute scale of activations → changes representational ability. BN instead normalizes each example *relative to the whole-dataset (minibatch) statistics*.

- **Gülçehre & Bengio (2013)** [gulcehre] standardization layer — acknowledged as similar but applied to nonlinearity *output*, no learned scale/shift, no conv handling, no deterministic inference. (Discussed in conclusion; pre-existing.)

- **Hyvärinen & Oja — ICA (2000)** [ica]. Cited for the "Wu+b is more Gaussian than u" intuition justifying normalizing pre-activation rather than layer input.

- Plumbing: SGD with momentum [momentum, Sutskever 2013], Adagrad [adagrad], DistBelief distributed training [dist-belief], Inception/GoogLeNet [inception].

## State of the field at the time (2015)

Deep nets trained by SGD + momentum/Adagrad. The pain: training is fragile — needs small learning rates and careful init; deep sigmoid/tanh nets barely train because activations saturate and gradients vanish. The accepted toolkit: ReLU, Xavier/orthogonal init, small LR, Dropout for regularization. Everyone "knew" whitening inputs helps (LeCun 1998) but only applied it to the data, once. Several groups were trying to push normalization/whitening *inside* training (natural gradient, mean-normalized SGD) but doing it as a side computation, which fought the optimizer.

## Key derivation pieces verified against the paper

- Naive "subtract running mean outside the gradient step" failure: with x=u+b, x̂=x−E[x], if the GD step ignores ∂E[x]/∂b, then b←b+Δb leaves u+b−E[u+b] unchanged → loss unchanged → b grows without bound; "model blows up." (paper Sec 2)
- Two simplifications: (1) normalize each scalar feature independently (not joint whitening) — justified by LeCun 1998 and by avoiding singular covariance matrices when minibatch m < #activations; (2) use minibatch statistics so they participate in backprop.
- Restore representational power: y = γx̂ + β; setting γ=√Var[x], β=E[x] recovers identity.
- BN transform forward (Alg 1): μ_B, σ²_B, x̂=(x−μ_B)/√(σ²_B+ε), y=γx̂+β.
- Backward (paper, before simplification): the 6 chain-rule equations — dℓ/dx̂, dℓ/dσ²_B, dℓ/dμ_B, dℓ/dx_i, dℓ/dγ, dℓ/dβ. Three paths into x_i (direct, via σ²_B, via μ_B).
- Inference: use population stats; Var[x] = m/(m−1)·E_B[σ²_B] (unbiased); BN collapses to a single affine map y = (γ/√(Var+ε))·x + (β − γE[x]/√(Var+ε)).
- Conv BN: normalize per feature map jointly over batch AND spatial locations (effective m' = m·p·q); one (γ,β) per feature map. Bias b is redundant (absorbed by β).
- Higher LR: BN(Wu)=BN(aWu); ∂BN((aW)u)/∂u unchanged, ∂BN((aW)u)/∂(aW) = (1/a)·∂BN(Wu)/∂W → scale-invariant; larger weights → smaller gradients → self-stabilizing. Conjecture: Jacobian singular values → 1 (if x̂, ẑ Gaussian/uncorrelated and F≈Jx̂ linear, then JJ^T=I).

## Explainers used

- Kratzert (2016), "Understanding the gradient flow through the BN layer": staged computational-graph backprop (9 steps) + the three-path intuition. https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
- Kevin Zakka (2016), Chris Yeh (2017): pen-and-paper derivation of the simplified single-line dx.
- Simplified dx: dx = (1/N)·γ·(var+ε)^(-1/2)·(N·dout − Σdout − (x−μ)(var+ε)^(-1)·Σ(dout·(x−μ))).

## Canonical code (in code/cs231n_batchnorm.py)

cs231n assignment2 layers.py (cthorey mirror). Contains:
- batchnorm_forward: staged steps 1–9 exactly mirroring Alg 1; running_mean/running_var EMA for test; separate test branch using running stats.
- batchnorm_backward: naive staged backward (matches Kratzert's 9 steps).
- batchnorm_backward_alt: simplified single-line dx.
- Also dropout_forward (for the BN-vs-Dropout regularization point) and conv layers.
Note: cs231n uses an EMA running average for inference rather than the paper's exact m/(m-1) unbiased population pass; both are documented. answer.md will present both the paper's exact procedure and the practical EMA.

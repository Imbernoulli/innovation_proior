# SimCLR — explainer / intuition capture (Phase 1 notes)

Sources consulted (web): mlshark "SimCLR and NT-Xent Loss Explained" (medium), aicompetence "InfoNCE vs Triplet vs NT-Xent", emergentmind NT-Xent topic, Frederik vom Lehn "NT-Xent / InfoNCE" (medium), Aditya Rastogi "The projection head in SimCLR" (medium), learnopencv "Contrastive Learning SimCLR and BYOL", mlhonk substack walkthrough. (Pages skimmed for intuition only; load-bearing facts cross-checked against the primary papers.)

## NT-Xent == InfoNCE, specialized
- The community consensus (and SimCLR's own footnote) is that NT-Xent is InfoNCE/the N-pair loss with: cosine similarity on L2-normalized vectors, temperature τ, and a symmetric treatment of both views (i,j) and (j,i). For each anchor it is literally cross-entropy of a (2N-1)-way classification where the "correct class" is the other view of the same image.
- Numerator = exp(sim(z_i,z_j)/τ); denominator = sum over all k≠i. Self-similarity (k=i) is masked out (the "-LARGE_NUM" trick in code).

## Temperature τ and hard negatives
- Small τ sharpens the softmax → the loss concentrates gradient on the hardest negatives (the ones most similar to the anchor); the per-negative gradient weight is exp(u·v⁻/τ)/Z(u). This is why NT-Xent "learns from hard negatives" automatically, unlike margin/triplet which need explicit semi-hard mining.
- But too-small τ over-penalizes and can break the structure (over-focus on a few hard negatives); too-large τ → uniform weighting, weak signal. SimCLR finds an intermediate τ (≈0.1 on ImageNet, ≈0.5 on CIFAR) best. Without L2 norm the dot products are unbounded → need huge τ and representation quality drops even though contrastive accuracy rises.

## Why the projection head, and why discard it
- g(h)=W2·ReLU(W1·h) is trained so z=g(h) becomes invariant to the augmentations (that's what the loss demands). Invariance = throwing away exactly the info that distinguishes the two views: color, orientation, crop position.
- For downstream tasks that info is often useful → you want it preserved. By putting a nonlinear g between h and the loss, the loss can be satisfied by g while h upstream keeps the richer info. So keep h (before g), discard g.
- SimCLR's own probe (Table critic_invariance): an MLP can recover rotation/color/corruption much better from h than from g(h) — direct evidence g removes that info.
- Recent theory framings (post-hoc, NOT for reasoning.md): "subspace selection" / "geometric buffering" — g induces a metric singularity along augmentation directions so the backbone stays linearly separable. (Hindsight; keep out of context/reasoning.)

## Large batch / many negatives
- Each positive in a batch of N gets 2(N-1) negatives. More negatives = harder, more informative classification task = better representation, and (per InfoNCE bound I ≥ logN − L) a tighter MI lower bound. SimCLR shows large batch helps most when training is short; with long training the gap shrinks. LARS needed because plain SGD with linear LR scaling is unstable at batch 4096–8192.

## Composition of augmentations
- Single augmentation: model can almost perfectly solve the contrastive task yet representation is poor. Composition makes the task harder → forces general features.
- Crop-only shortcut: random crops of one image share a color histogram, so the net can match views by color statistics alone. Adding color distortion destroys that shortcut → must learn shape/semantics. Crop+color is the standout pair.
- Random crop alone already subsumes "global-to-local" and "adjacent-view" prediction tasks that prior work baked into architecture — so augmentation, not architecture, defines the task.
</content>

# SimSiam Synthesis

This note is superseded by `notes/discovery_synthesis.md`, which was rebuilt from the retrieved primary paper, ancestors, explainers, author slides, and canonical code.

Key corrections from the strict audit:

- Stop-gradient is the controlled empirical collapse switch in the reported baseline, not a theorem-level necessary-and-sufficient proof for every setting.
- Negative cosine and normalized squared distance satisfy \(\|\hat p-\hat z\|_2^2 = 2 + 2D(p,z)\), so the equivalence includes an additive constant.
- The predictor's regression optimum is a conditional mean \(h^*(z_1)=\mathbb E[z_2\mid z_1]\), which the paper heuristically relates to the per-image augmentation mean \(\mathbb E_T[f(T(x))]\).
- The canonical projector preserves the original ResNet `fc` layer and disables its bias after construction; the predictor output layer keeps its default bias.

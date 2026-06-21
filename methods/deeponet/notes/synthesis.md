# DeepONet Synthesis

This note is superseded by the strict evidence-backed reconstruction in:

- `notes/source_matrix.md`
- `notes/discovery_synthesis.md`

The refreshed synthesis uses the 2021 Nature Machine Intelligence article, the arXiv v3 source, the Chen-Chen ancestor theorem, the Karniadakis CBMM self-account transcript, third-party explainers, and the DeepXDE PyTorch implementation. It also records two audit-sensitive corrections:

- The sensor theorem's hidden bias has shape `R^n`, not `R^{m+1}`, because `W_1 in R^{n x (m+1)}`.
- DeepXDE applies the trunk activation after the trunk FNN output, then merges branch/trunk outputs by `einsum` and adds a scalar learnable bias.

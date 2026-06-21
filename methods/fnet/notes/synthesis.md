# FNet Synthesis

This compatibility note is superseded by `discovery_synthesis.md` and `source_matrix.md`, which were rebuilt from the primary paper, canonical code, ancestors, explainers, and author self-account.

Key corrections from the previous synthesis:

- The result files now distinguish the paper's unitary DFT-matrix convention from the canonical code path. The official code uses unnormalized `scipy.linalg.dft` matrices to match `np.fft.fftn` / `jnp.fft.fftn`.
- The code snippets now project the feed-forward sublayer back to the original `d_model`, matching `f_net/layers.py`; they no longer use the expanded `d_ff` shape as the output dimension.
- `context.md` no longer leaks the final Fourier/DFT move before the reasoning stage.
- `reasoning.md` is a continuous first-person reconstruction with no markdown headers.

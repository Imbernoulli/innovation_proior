Measured result — the nanoGPT-style baseline (hlb-gpt tag `0.0.0`, the baseline release; numbers from the
tag's `README.md`). Metric: A100 wall-clock seconds to reach the ~3.8 val-loss bar, **lower is better**.

| configuration | val loss / perplexity | hardware | wall-clock to bar |
|---|---|---|---|
| nanoGPT-style baseline (tag 0.0.0) | **~3.8 / ≈44.7** | single A100 | **just over ~6 minutes** |

The `0.0.0` README states this code "achieves under ~3.8 val loss (44.7 perplexity) on WikiText-103 in
just over 6 minutes," describing itself as a "distilled, feature-pruned, relatively faithful
reimplementation of a basic GPT language model as defined in Karpathy's nanoGPT," whose only intended
differences are the attention call, the full-accuracy PyTorch GELU, and the native PyTorch one-cycle
scheduler. The bar is held here; the ~6-minute wall-clock is the clock every later rung has to beat. (Per
the repo's convention the timing lives in the release's README/notes, not in a results table — this rung's
number is the `0.0.0` README's headline figure.)

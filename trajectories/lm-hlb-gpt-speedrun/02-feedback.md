Measured result — throughput speedups added to the baseline (hlb-gpt release `0.1.0`, "beta v0.1.0";
number from the release body). Metric: A100 wall-clock seconds to reach the ~3.8 val-loss bar, **lower is
better**.

| configuration | val loss | hardware | wall-clock to bar |
|---|---|---|---|
| nanoGPT-style baseline (tag 0.0.0) | ~3.8 | single A100 | just over ~6 min |
| + dynamic-batch throughput speedups (release 0.1.0) | ~3.8 | single A100 | **~"nearly in half" (≈3.5 min)** |

The `0.1.0` release body states this release adds "a few features that cuts the training time nearly in
half" (and includes a backwards-compatibility hotfix for torch < 2.0). Halving "just over 6 minutes" lands
the run at roughly three-and-a-half minutes, with the ~3.8 val-loss bar held. (Per the repo's convention
the time lives in the release notes, not a results table; "nearly in half" is the release's own
characterization of the speedup.)

Measured result — sequence-length scheduling + learnable linear positional bias (hlb-gpt release `0.2.0`,
"v0.2.0 beta"). Metric: A100 wall-clock seconds to reach the ~3.8 val-loss bar, **lower is better**.

| configuration | val loss | hardware | notes |
|---|---|---|---|
| + dynamic-batch throughput speedups (release 0.1.0) | ~3.8 | single A100 | ~3.5 min |
| + sequence-length schedule & linear position bias (release 0.2.0) | ~3.8 | single A100 | **further speedup** |

The `0.2.0` release body states this release "add[s] sequence length scheduling and make[s] a few other
tweaks" (the supporting change being the learnable linear positional encoding that replaces the absolute
position embedding). The release does not headline a single wall-clock number — per the repo's convention
the times live across the release notes and tag titles rather than one results table — so this rung's
contribution is the documented schedule/position change carrying the run forward between the `0.1.0`
(~3.5 min) and `0.3.0` (~136–140 s) records, with the ~3.8 val-loss bar held.

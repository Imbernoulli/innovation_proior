Measured result — SiGLU MLP + pure-bfloat16 network (hlb-gpt release `0.3.0`, "v0.3.0 beta (~136–140s)";
numbers from the release title and body). Metric: A100 wall-clock seconds to reach the ~3.8 val-loss bar,
**lower is better**.

| configuration | val loss | hardware | wall-clock to bar |
|---|---|---|---|
| + sequence-length schedule & linear position bias (release 0.2.0) | ~3.8 | single A100 | (further speedup) |
| + SiGLU MLP + pure bf16 (release 0.3.0) | ~3.8 | single A100 | **~136–140 s** |

The `0.3.0` release title is "v0.3.0 beta (~136–140s)," and the body states it "upgrade[s] the MLP a bit to
include the SiGLU activation function (over the default non-linearly-gated GELU function), convert[s] the
network over to pure bfloat16 (from a mixed precision dynamic), and perform[s] various optimizations to
bring our training time down another 18–22 seconds or so." So this rung lands the run at roughly
**136–140 seconds** on the A100, with the ~3.8 val-loss bar held. (Per the repo's convention the wall-clock
lives in the release title/notes, not a single results table.)

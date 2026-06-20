Measured result — `construct:hierarchical-gradient` (β-annealed Adam over hierarchical `20→100→500` lifts,
fixed seeds). `R` from the FFT autoconvolution evaluator on the returned heights. Runtime `~100 s`.

| Stage | pieces `N` | `R` |
|---|---|---|
| coarse seed (prev rung) | 20 | 0.884823 |
| lift ×5 + refine | 100 | ~0.8915 |
| lift ×5 + refine (returned) | 500 | 0.894706 |

Reference points: Matolcsi–Vinuesa 20-step `0.88922`, AlphaEvolve 50-step `0.89628`, Boyer–Li 575-step
`0.901564`, Jaech–Joseph 539-step `~0.9016`, AlphaEvolve-V2 record `0.96102`.

Notes: `+0.0099` over the coarse rung, into the band of the published `50`–`575`-step constructions
(`0.896`–`0.9016`). The lift-and-refine principle holds at every level: each `×5` upscale is exactly
ratio-preserving (free), and the β-annealed Adam refinement carves asymmetric fine structure the previous
resolution could not represent, flattening the autoconvolution's cap. The post-lift multiplicative kick is
load-bearing — without it Adam sits on the degenerate flat-block plateau. The returned `500`-piece value
`0.894706` sits just under the AlphaEvolve `50`-step `0.89628` and below the `~0.9016` of Boyer–Li /
Jaech–Joseph, as expected: those spent vastly more compute (Boyer–Li `~10^6` gradient trajectories) and the
last fraction of a percent is dominated by the irregular fine structure of the true optimum. The gradient
is still paying at `500` pieces, which is the opening for the endpoint: lift once more to thousands of
pieces and spend a long, kicked, sharpening Adam schedule to push toward the published step-function
frontier.

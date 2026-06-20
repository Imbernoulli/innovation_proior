Measured result — `construct:coarse-anneal` (simulated annealing on `N=50` heights + `β`-annealed softmax-Adam
polish, fixed seeds). `R` from the FFT autoconvolution evaluator on the returned `50` heights. Runtime `~7 s`.

| Stage | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| uniform (prev rung) | any | 2.000000 |
| simulated annealing (best restart) | 50 | 1.736925 |
| + β-annealed softmax-Adam polish (returned) | 50 | 1.537084 |

Reference points: flat ceiling `2.0`, AlphaEvolve 600-step `1.5053`, AutoEvolver record 30000-step `1.5028628969`,
provable floor `1.28`.

Notes: `−0.463` below the flat ceiling, reaching `1.537084` — a large first drop, confirming that breaking the
flat symmetry is exactly the lever. The annealing alone (`1.7369`) gets most of the way; the gradient polish on the
softmax surrogate slides it the rest of the way down its basin (`−0.20` more). The returned `50`-piece profile is
genuinely asymmetric and partly sparse — `~9` of the `50` heights are driven to (near-)zero, with mass concentrated
toward the ends — the peak-suppressing structure the literature reports for good `C1` constructions. The rung lands
`~0.032` above the `600`-piece AlphaEvolve `1.5053`, and `~0.034` above the `30000`-piece record `1.5028628969`,
capped by the coarse `N=50` resolution: `50` pieces cannot render a fine enough autoconvolution to push the peak
lower. That coarse-resolution stall is the opening for the next rung — lift this shape to a finer grid and bring a
stronger optimizer that attacks the whole flat top of near-peak nodes at once.

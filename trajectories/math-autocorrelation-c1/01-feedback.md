Measured result — `construct:uniform` (flat step function, the discretized indicator). `R` from the FFT
autoconvolution evaluator. Deterministic; runtime negligible.

| Profile | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| flat (returned) | 50 | 2.000000 |
| flat | 600 | 2.000000 |
| flat | 30000 | 2.000000 |

Cross-checks (same evaluator): AutoEvolver record sequence (`N=30000`) → `R = 1.5028628969`, matching the published
record to 10 digits; flat profile → `R = 2` exactly at every `N`.

Reference points: flat ceiling `2.0`, AlphaEvolve 600-step `1.5053`, AutoEvolver record `1.5028628969`, provable
floor `1.28`.

Notes: the flat indicator scores exactly `2`, matching the hand computation (triangular autoconvolution: discrete
self-convolution peaks at `N`, mass `N`, so `R = 2N·N/N² = 2`) and confirming the `2N` normalization. The value is
independent of piece count — refining a flat profile leaves the triangle and the value `2` unchanged — so piece
count alone is not a lever. The evaluator reproduces the published record `1.5028628969` to 10 digits on the
AutoEvolver sequence, so the harness is trustworthy on both ends. This rung has no internal degree of freedom and
no gradient; the entire descent from `2.0` toward the frontier near `1.50286` must be bought by introducing
asymmetric, structured, peak-suppressing heights — the job of the next rung.

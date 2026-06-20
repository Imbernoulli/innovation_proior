Measured result — rung 2 `priority:best-fit`. Same frozen simulator, same five seeded streams per
family (5000 items each). Mean over the five seeds; rung-1 First-Fit shown for comparison.

| Stream family | C | mean #bins | L1 lower bound | excess over LB | (rung 1) |
|---|---|---|---|---|---|
| Weibull(45,3) | 100 | 2089.80 | 2006.60 | **4.146%** | 4.545% |
| OR-style uniform[20,100] | 150 | 2087.60 | 1999.40 | **4.411%** | 4.651% |

Per-seed #bins — Weibull: `[2100, 2086, 2097, 2085, 2081]` (First-Fit `[2107, 2093, 2109, 2091, 2089]`).
Per-seed #bins — OR-style: `[2087, 2095, 2080, 2068, 2108]` (First-Fit `[2098, 2097, 2081, 2074, 2112]`).

Calibration against the FunSearch repo datasets: Best Fit on OR3 = `5.37%`, on Weibull 5k = `3.98%` —
matched to the digit (published table: `5.37%` / `3.98%`).

Notes: Best-Fit beats First-Fit on every single seed of both families — Weibull `4.545% → 4.146%`
(−0.40 pts, ~8 fewer bins/stream), OR-style `4.651% → 4.411%` (−0.24 pts, ~5 fewer bins/stream). The
gain is consistent but modest, exactly as expected: both rules do greedy reuse, which is most of the
saving; Best-Fit's extra mile is closing each bin off at the tightest available fit and preserving
roomy bins. The residual `~4%` is the dead space in those tight-fit holes — leftover slivers Best-Fit
cannot avoid because it scores each bin in isolation as a monotone function of its own slack. A
side-experiment confirmed the limit literally: replacing `−(r−s)` with `−(r−s)²`, `exp(−(r−s))`, or
any monotone reshaping leaves the bin count *identical* (same argmax ordering), and adding tiny fill-
based tie-breaks changes nothing. Beating Best-Fit requires a score that compares a bin against the
others — the next rung's move.

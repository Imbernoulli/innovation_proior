Measured result — rung 4 ENDPOINT `priority:funsearch` (FunSearch-discovered Weibull heuristic,
verbatim from `google-deepmind/funsearch`). Same frozen simulator, same five seeded streams per
family. Mean over five seeds; all rungs shown.

| Stream family | C | mean #bins | L1 LB | excess | r3 gap-diff | r2 BF | r1 FF |
|---|---|---|---|---|---|---|---|
| Weibull(45,3) | 100 | 2020.00 | 2006.60 | **0.668%** | 0.718% | 4.146% | 4.545% |
| OR-style uniform[20,100] | 150 | 2068.00 | 1999.40 | 3.431% | 3.271% | 4.411% | 4.651% |

Per-seed #bins — Weibull: `[2026, 2022, 2020, 2017, 2015]` (rung 3 `[2028, 2023, 2022, 2018, 2014]`).
Per-seed #bins — OR-style: `[2066, 2068, 2057, 2050, 2099]` (rung 3 `[2066, 2055, 2057, 2048, 2098]`).

Calibration against the FunSearch repo datasets (their instances, same simulator): FunSearch-Weibull
on Weibull 5k = `0.68%`, FunSearch-OR on OR3 = `3.11%` — both matched to the published Table 1 digits
(`0.68%`, `3.11%`), and First/Best Fit reproduced as `4.23%/3.98%` (Weibull 5k) and `5.74%/5.37%`
(OR3).

Notes — honest reading. On the **Weibull** streams (the heuristic's home distribution) the endpoint is
best of all four rungs: `0.668%` excess, essentially at the lower bound, edging past the rung-3 core
`0.718%` and crushing Best-Fit's `4.146%` — a `~70`-bin saving per 2090-bin stream, reproducing the
paper's headline `0.68%`. This is the published claim: the discovered priority function beats Best-Fit,
and on Weibull it gets to within a fraction of a percent of optimal.

On the **OR-style** streams the endpoint (`3.431%`) is slightly *worse* than my hand-built rung-3 core
(`3.271%`), though both still beat Best-Fit (`4.411%`). This is not a regression in the ladder — it is
the expected fingerprint of a *searched* heuristic: this is the function FunSearch evolved on the
**Weibull** distribution (the repo ships a separately-discovered OR heuristic, which scores `3.11%` on
OR3). Its two extra item-power terms are tuned to Weibull item sizes, and off that distribution they do
not help and cost a little against the simpler relational core. The honest summary: the discovered
heuristic dominates exactly where it was searched (Weibull, to within `0.03%` of the bound at 100k per
the paper), and the gap to Best-Fit is the published `~3.3-point` improvement reproduced on our seeded
streams. Endpoint method informed by `google-deepmind/funsearch`, FunSearch (Romera-Paredes et al.,
Nature 2024).

Measured result — rung 3 `priority:gap-diff` (hand-designed relational score: `(r − r_max)² / s`,
feasibility-signed, neighbour-differenced). Same frozen simulator, same five seeded streams per
family. Mean over five seeds; rungs 1–2 shown for comparison.

| Stream family | C | mean #bins | L1 LB | excess | (rung 2 BF) | (rung 1 FF) |
|---|---|---|---|---|---|---|
| Weibull(45,3) | 100 | 2021.00 | 2006.60 | **0.718%** | 4.146% | 4.545% |
| OR-style uniform[20,100] | 150 | 2064.80 | 1999.40 | **3.271%** | 4.411% | 4.651% |

Per-seed #bins — Weibull: `[2028, 2023, 2022, 2018, 2014]` (BF `[2100, 2086, 2097, 2085, 2081]`).
Per-seed #bins — OR-style: `[2066, 2055, 2057, 2048, 2098]` (BF `[2087, 2095, 2080, 2068, 2108]`).

Notes: the relational score beats Best-Fit decisively and on every seed of both families. On the
Weibull streams the jump is large — `4.146% → 0.718%`, i.e. `~69` fewer bins per stream of 2089,
collapsing nearly all of Best-Fit's residual waste. On the OR-style streams the gain is real but
smaller — `4.411% → 3.271%` (`~23` fewer bins/stream). The differencing is confirmed to be the
engine: an ablation that drops `score[1:] -= score[:-1]` and keeps only the per-bin relational term
explodes to `~80%` excess (Weibull) — catastrophic — so the neighbour coupling is doing all the work,
exactly the property a single-bin rule (Best-Fit) cannot have. The asymmetry between families
(Weibull near-optimal, OR-style only partway) is a fingerprint that this hand-picked form
`(r − r_max)² / s` is tuned, implicitly, to the Weibull regime and is not the right form everywhere —
the powers and the coupling were guessed, not searched. Searching that functional form is the final
rung; the published FunSearch heuristic is exactly such a searched form, and adds two further item-
power terms on top of this relational core.

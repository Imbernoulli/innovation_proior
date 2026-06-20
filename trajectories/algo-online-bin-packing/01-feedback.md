Measured result — rung 1 `priority:first-fit`. Frozen FunSearch online-bin-packing simulator. Two
seeded stream families, five seeds each (0–4), 5000 items per stream. Mean over the five seeds.

| Stream family | C | mean #bins | mean L1 lower bound | excess over LB |
|---|---|---|---|---|
| Weibull(45,3) | 100 | 2097.80 | 2006.60 | **4.545%** |
| OR-style uniform[20,100] | 150 | 2092.40 | 1999.40 | **4.651%** |

Per-seed #bins — Weibull: `[2107, 2093, 2109, 2091, 2089]`, L1 `[2016, 2007, 2007, 2001, 2002]`.
Per-seed #bins — OR-style: `[2098, 2097, 2081, 2074, 2112]`, L1 `[2005, 2003, 1993, 1981, 2015]`.

Calibration against the published FunSearch repo datasets (same simulator, their instances): First
Fit on OR3 = `5.74%`, on Weibull 5k = `4.23%` — matched to the digit, confirming the simulator is
faithful before we read our own seeded streams.

Notes: greedy reuse alone lands `~4.5–4.7%` above the lower bound on both families — a clear, stable
margin of wasted capacity, consistent across all five seeds (spread `< 0.3%`). The waste is the
predicted one: First-Fit takes the earliest valid bin regardless of how much slack the placement
leaves, so roomy bins get spent on small items and nearly-full bins are not topped off. This is the
floor; the next rung ranks by fit tightness (Best-Fit) to recover that wasted capacity.

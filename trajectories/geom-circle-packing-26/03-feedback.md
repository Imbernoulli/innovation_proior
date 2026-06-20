Measured result — `construct_packing:multistart-slsqp` (`K=120` random-restart SLSQP, master seed
`12345`, radii LP-re-tightened, best feasible kept). Feasibility verified at `atol=1e-7`.

| Instance | seed | starts | Σ rᵢ | feasible | max constraint violation |
|---|---|---|---|---|---|
| n = 26 | 12345 | 120 | 2.6221020467 | yes | 3.22e-15 |

Reference points: grid `2.5414`, single-SLSQP `2.5949`; AlphaEvolve `2.63586276`, ShinkaEvolve
`2.635983283`, AutoEvolver record `2.635988438568`.

Notes: best-of-`120` restarts lifts the sum `+0.0272` over the single start (`2.5949 → 2.6221`),
confirming the climb is governed by sampling more basins — new bests appeared sporadically
throughout the run (e.g. at starts 5, 57, 84), the order-statistic signature of random multi-start.
It remains `~0.014` below the frontier band: uniform random scatters rarely seed the rare
top-quality basins, so the best-of-many saturates. The plateau is the limitation — blind restarts
have no structure and no memory of the best packing found. The endpoint rung adds structured
initialization (golden-angle spiral + corner/edge seeding) and *exploits* the incumbent via
iterated perturbation chains, the construction the AutoEvolver / ShinkaEvolve frontier results use.

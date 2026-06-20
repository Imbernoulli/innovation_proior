Measured result — `construct:finer-basinhop` (lift `24→60`, upscale `×2→120`, basin-hopping around
annealed SLSQP; seeds `1/3`), AlphaEvolve App. B.5 evaluator. `C` is the true hard-max overlap of the best
returned vector.

| Stage | pieces `n` | `Σ v` (target `n/2`) | `C` (upper bound) |
|---|---|---|---|
| coarse SLSQP (prev rung) | 24 | 12.0 | 0.381240 |
| multistart | 60 | 30.0 | 0.381098 |
| upscale ×2 (free, same `C`) | 120 | 60.0 | 0.381098 |
| basin-hop refine (returned) | 120 | 60.0 | 0.381076 |

Reference points: White lower bound `0.379005`, AutoEvolver record `0.38086945`, AlphaEvolve `0.380924`,
Haugland `0.380927`, Erdős floor `0.5`.

Notes: `−0.000164` over the coarse rung, reaching `0.381076` at `120` cells — within `~1.5×10^{-4}` of
the Haugland/AlphaEvolve landmarks (`~0.38092`) and `~1.9×10^{-4}` of the AutoEvolver record
`0.38086945`. As predicted the gain from `24→120` cells is the fine shaving, not a big jump: the coarse
profile already held the gross structure, and the extra resolution lets basin-hopping carve a slightly
lower worst-overlap envelope. The upscale step is confirmed free (same `C = 0.381098` before and after).
The returned profile is feasible (`Σ v = 60 = n/2`) and near-binary (`~31%` of cells pinned at `0`/`1`).
The two load-bearing ingredients are basin-hopping (the finer landscape has more local minima, and
perturb-and-re-solve jumps basins while keeping good structure) and the sharper `β` ladder (annealed to
`3600`, so the soft-max tracks the hard max at this spikier resolution). The cap is still resolution, now
at the fine end: the published frontier lives at several hundred cells. The endpoint rung lifts once more,
to the `~600`-cell scale the records use, with a longer basin-hopping budget plus an exact-minimax
subgradient polish, pushing toward `~0.3809`.

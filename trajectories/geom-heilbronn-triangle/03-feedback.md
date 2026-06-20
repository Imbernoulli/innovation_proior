Measured result — `construct:sa` (simulated annealing on point positions, `30` restarts × `200,000`
steps, incremental `45`-of-`165` scoring, seed `1`). Exact min triangle area over all triples;
returned config verified inside `[0,1]^2`. Runtime `~103 s`.

| Method | min triangle area | fraction of record |
|---|---|---|
| inscribed 11-gon (rung 1) | 0.021456 | 0.579 |
| random multi-start (rung 2) | 0.010872 | 0.294 |
| **simulated annealing, best of 30 restarts** (returned) | **0.035639** | **0.962** |

Per-restart spread (illustrative): worst restart `0.0241`, median `~0.031`, best `0.035639`.
Reference: Goldberg record `1/27 = 0.037037` (fraction `1.000`).

Notes: a large jump — `0.0109 → 0.0356`, from `0.294` to `0.962` of the record — confirming that
*improving* configurations (annealing) beats merely *sampling* them (random multi-start) by a wide
margin, and that accepting downhill moves escapes the knife-edge traps the minimum-of-`165`
objective creates. The per-restart spread is wide (`0.024`–`0.036`), so multi-restart is doing real
work: the reported number is the best of `30` independent runs, and individual runs land in basins
of varying depth. The best run reaches `0.962` of the record but plateaus just short of `1/27`: near
the optimum several triangles are simultaneously near-tight, and a random single-point Gaussian move
tends to grow one while shrinking another, so annealing cannot coordinate the final squeeze. That is
the endpoint's opening — a smooth, differentiable soft-minimum objective whose gradient pushes all
the near-tight triangles up at once, used to polish the best annealed configuration onto the exact
optimum.

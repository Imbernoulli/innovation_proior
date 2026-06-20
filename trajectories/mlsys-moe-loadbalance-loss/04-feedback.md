Measured result — `shinka-entropy-hinge` (ShinkaEvolve discovered loss, arXiv:2509.19349 Eq. 1:
global-batch LBL + entropy-weighted under-utilization hinge, `τ=0.064/N`,
`s(P_ℓ)=0.5+(1−H(P_ℓ)/log N)`, hinge coefficient `0.1/L`). Same tiny MoE, task, steps, seed,
held-out protocol. Full ladder:

| Variant | `L_CE` | perplexity | `L_imb` | fitness `r` |
|---|---|---|---|---|
| no-balance (control) | 3.7280 | 41.594 | 0.1286 | −3.8566 |
| switch-aux (micro-batch `f·P`) | 3.7281 | 41.599 | 0.0587 | −3.7868 |
| global-lbl (global `f·P`) | 3.7279 | 41.592 | 0.0561 | −3.7840 |
| global-lbl + loss-free bias | 3.7274 | 41.570 | 0.0160 | −3.7434 |
| **shinka-entropy-hinge** | **3.7276** | **41.579** | **0.0188** | **−3.7464** |

Notes: among the **pure differentiable losses** the discovered loss is the best balance/CE point —
it cuts imbalance from global-LBL's 0.0561 to **0.0188** (a 3.0× improvement over plain global, 6.8×
over the control) at essentially unchanged CE (41.579 vs 41.592), lifting fitness from −3.7840 to
−3.7464. This is exactly the predicted mechanism: the entropy-gated hinge actively rescues the
under-floor experts (which the smooth `f·P` term only averages over) without touching the healthy
ones, so balance improves with no specialization cost in CE. The one variant with lower imbalance,
`global-lbl + loss-free bias` (0.0160), wins via a *non-gradient* hard-count selection controller —
a complementary mechanism, not a competing loss; the two are orthogonal and stack in principle.
Among loss functions scored on the ShinkaEvolve fitness, the entropy-hinge is the endpoint.

This is a deliberately small reproduction. The discovered loss's authoritative form is Eq. 1 of
arXiv:2509.19349 (the released repo github.com/SakanaAI/ShinkaEvolve does not ship the MoE example
code; the paper formula was confirmed verbatim). ShinkaEvolve's own eval: 556M/82M-active MoE,
`N=64` top-8, ~2.10B FineWeb tokens, `λ=0.01`, beating DeepSeek/Qwen global-batch LBL on seven
downstream benchmarks — a scale this reproduction reproduces in mechanism and ordering, not in size.

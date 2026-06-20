Measured result — `global-lbl` (global-batch `α·N·Σ_i f_i P_i`, `α=1e-2`, `f` over the full batch),
reported both alone and paired with DeepSeek's auxiliary-loss-free selection bias (`u=1e-3`,
selection-only, no aux gradient). Same tiny MoE, task, steps, seed, held-out protocol.

| Variant | `L_CE` | perplexity | `L_imb` | fitness `r` |
|---|---|---|---|---|
| no-balance (control) | 3.7280 | 41.594 | 0.1286 | −3.8566 |
| switch-aux | 3.7281 | 41.599 | 0.0587 | −3.7868 |
| global-lbl | 3.7279 | 41.592 | 0.0561 | −3.7840 |
| **global-lbl + loss-free bias** | **3.7274** | **41.570** | **0.0160** | **−3.7434** |

Notes: global-batch LBL alone slightly improves on the Switch loss's balance (`L_imb` 0.0561 vs
0.0587) at a hair lower CE (41.592 vs 41.599) — consistent with the global scope not
over-constraining slices, though at this small scale where micro-splits are representative the gap
over Switch is marginal. The decisive effect is the **loss-free bias**: as a direct hard-count
controller on the top-K selection it drives imbalance down **3.5×** (0.0561 → 0.0160), the lowest of
any rung, lifting
fitness to −3.7434 with CE actually slightly improved. This is an honest, expected outcome — a count
controller balances counts more aggressively than a smooth `f·P` gradient. Its limitation is what
motivates the endpoint: the bias balances the *average* count but gives the router no gradient signal
about balance, and neither it nor the smooth term specifically rescues the under-used tail through
the router's own probabilities — the entropy-hinge does.

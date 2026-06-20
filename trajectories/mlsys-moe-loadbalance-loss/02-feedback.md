Measured result — `switch-aux` (Switch/GShard aux loss `α·N·Σ_i f_i P_i`, `α=1e-2`, `f` on 4
micro-splits). Same tiny MoE, task, steps, seed, and held-out protocol as the control.

| Variant | `L_CE` | perplexity | `L_imb` | fitness `r` |
|---|---|---|---|---|
| no-balance (control) | 3.7280 | 41.594 | 0.1286 | −3.8566 |
| **switch-aux** | **3.7281** | **41.599** | **0.0587** | **−3.7868** |

Notes: the aux loss cuts imbalance by **2.2×** (0.1286 → 0.0587) at essentially flat cross-entropy
(Δ`L_CE` = +0.0001, within run noise), so fitness rises from −3.8566 to −3.7868 (+0.0698, almost all
from the imbalance term). The `f·P` surrogate works as designed — the differentiable `P` carries the
balancing gradient steered by the detached counts `f`. The micro-batch `f` is the known limitation
(over-constrains each slice); at this small scale the slices are fairly representative, so the
specialization cost does not yet show as a visible CE penalty — the next rung's global `f` is at
least as good on CE and sets up the under-use rescue.

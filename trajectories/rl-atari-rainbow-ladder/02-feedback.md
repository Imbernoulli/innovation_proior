Measured result — DQN + Double Q-learning target, Atari-57, no-ops regime, 200M frames.

| agent | median human-normalized score |
|---|---|
| DQN | 79% |
| **Double DQN** | **117%** |

Median over the 57 games (higher is better). Fixing the upward-biased max with a decoupled target lifts the
median from 79% to 117% — a broad gain, and enough to carry the typical game above the human reference
(>100%). The value object is still a scalar mean; exploration and replay are unchanged.

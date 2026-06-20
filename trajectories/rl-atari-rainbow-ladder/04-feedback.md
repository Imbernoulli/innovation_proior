Measured result — DDQN + Prioritized Experience Replay, Atari-57, no-ops regime, 200M frames.

| agent | median human-normalized score |
|---|---|
| DQN | 79% |
| Double DQN | 117% |
| Noisy Nets | 118% |
| **Prioritized Replay** | **140%** |

Median over the 57 games (higher is better). Reallocating the replay budget toward surprising transitions
lifts the median from 118% to 140% — a broad gain, as predicted: it improves data efficiency on the typical
game rather than a minority. The largest single step since the decoupled target.

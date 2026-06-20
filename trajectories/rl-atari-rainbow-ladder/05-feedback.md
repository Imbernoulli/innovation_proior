Measured result — DDQN + Dueling architecture, Atari-57, no-ops regime, 200M frames.

| agent | median human-normalized score |
|---|---|
| DQN | 79% |
| Double DQN | 117% |
| Noisy Nets | 118% |
| Prioritized Replay | 140% |
| **Dueling DDQN** | **151%** |

Median over the 57 games (higher is better). Splitting the head so the state value is learned from every
transition lifts the median from 140% to 151% — another broad gain, as predicted, of the same character as
prioritized replay. The value object is still a scalar mean.

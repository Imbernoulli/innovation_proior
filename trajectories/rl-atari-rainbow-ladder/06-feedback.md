Measured result — DQN + Distributional RL (C51), Atari-57, no-ops regime, 200M frames.

| agent | median human-normalized score |
|---|---|
| DQN | 79% |
| Double DQN | 117% |
| Noisy Nets | 118% |
| Prioritized Replay | 140% |
| Dueling DDQN | 151% |
| **C51 (distributional)** | **164%** |

Median over the 57 games (higher is better). Learning the whole return distribution lifts the median from
151% to 164% — the strongest single component in the ladder, and the most a single change to the DQN floor
buys. Six axes have now each been improved in isolation; the finale asks whether combining all of them
compounds.

Measured result — DQN + Noisy Nets (Noisy DQN), Atari-57, no-ops regime, 200M frames.

| agent | median human-normalized score |
|---|---|
| DQN | 79% |
| Double DQN | 117% |
| **Noisy Nets** | **118%** |

Median over the 57 games (higher is better). As anticipated, learned parametric exploration barely moves
the median (118% vs the 117% decoupled-target baseline): the gain is real but concentrated on a minority of
hard-exploration games, so it lifts the tails of the distribution, not its center. Exploration was not what
capped the typical game.

# NGU explainer captures (Lil'Log, xlnwel summary, vitalab, search syntheses)

## Lil'Log (lilianweng.github.io exploration-drl)
- NGU = episodic novelty module (within-episode, kNN over episodic memory M of controllable-state
  embeddings phi(s)) + lifelong novelty module (RND, across-episode, slow).
- Embedding phi via inverse-dynamics features (IDF), identical to ICM encoding; only controllable
  changes captured, filters irrelevant variation.
- Episodic bonus r_episodic ~= 1/sqrt( sum_{k-NN} K(phi(x_t), phi_i) + c ); inverse kernel with
  running average d_m^2 of squared Euclidean distances.
- Lifelong: alpha_t = 1 + (err_RND(x_t) - mu_e)/sigma_e (normalized RND error).
- Multiplicative: r_i_t = r_episodic_t * clip(alpha_t, 1, L).
- Why: RND's weakness is long-horizon / it stops rewarding a state once novelty has vanished
  globally; episodic memory rapidly discourages revisits WITHIN an episode (resets each episode),
  so the agent is always pushed to cover new ground each episode; lifelong RND slowly down-modulates
  states seen across many episodes. Episodic avoids "forgetting" which states explored this episode.

## xlnwel summary / vitalab / search synthesis
- cluster distance xi = 0.008 zeroes out distances that are too small (treat near-identical states
  as the same cluster -> avoid self-similarity inflating counts).
- max similarity s_m = 8: if total similarity s > s_m, set r_episodic = 0 (state already heavily
  visited this episode -> no bonus).
- UVFA: one network Q(x,a,beta_i) approximates a family of value functions for augmented rewards
  r^{beta_i} = r^e + beta_i r^i, beta_0 = 0 (pure exploit), beta_{N-1} = beta (max explore).
  Lets you act greedily wrt Q(x,a,0) to turn off exploration -> retain exploitation on dense games.
- Base agent R2D2 (recurrent replay distributed DQN), Retrace loss; intrinsic reward fed as INPUT
  to the net (to keep it Markov / non-stationary reward handled as observation).

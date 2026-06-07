# Rainbow synthesis (arXiv 1710.02298, AAAI 2018, DeepMind Hessel et al.)

## Pain point / research question
Many independent DQN improvements exist, each fixing a different limitation, all on a shared
value-based/replay/target-net framework. UNCLEAR which are complementary and whether they
combine fruitfully. Goal: integrate six extensions addressing DISTINCT concerns into one agent;
show they are largely complementary; ablate to attribute contribution.

## DQN baseline (the trunk)
Loss (1-step Q-learning): (R_{t+1} + γ_{t+1} max_{a'} q_θ̄(S_{t+1},a') − q_θ(S_t,A_t))^2.
Online net θ (acts, gets gradient), target net θ̄ (periodic copy, frozen). RMSProp; uniform
replay (last 1M transitions); ε-greedy. Conv torso from pixels.

## The six components (each prior art, with the limitation it fixes)
1. **Double Q-learning (DDQN, van Hasselt 2016; Hasselt 2010)**: max in target overestimates
   (same net selects AND evaluates the bootstrap action → positive bias). FIX: decouple —
   select with online, evaluate with target:
   (R + γ q_θ̄(S', argmax_{a'} q_θ(S',a')) − q_θ(S,A))^2.
2. **Prioritized replay (Schaul 2015)**: uniform sampling wastes capacity on already-learned
   transitions. FIX: sample with prob p_t ∝ |TD error|^ω. New transitions inserted at max
   priority. ω shapes the distribution. (IS correction with β needed because sampling is now
   non-uniform → biased gradient.)
3. **Dueling networks (Wang 2016)**: hard to learn which states are valuable independent of
   action. FIX: two streams sharing conv encoder f_ξ:
   q_θ(s,a) = v_η(f_ξ(s)) + a_ψ(f_ξ(s),a) − (1/N_actions) Σ_{a'} a_ψ(f_ξ(s),a').
   Subtracting the mean advantage = identifiability (else v,a only defined up to constant).
4. **Multi-step learning (Sutton 1988)**: 1-step bootstrap propagates reward slowly + high bias.
   FIX: truncated n-step return R_t^{(n)} = Σ_{k=0}^{n-1} γ_t^{(k)} R_{t+k+1}, γ_t^{(k)}=Π_{i=1}^k γ_{t+i}.
   Loss (R_t^{(n)} + γ_t^{(n)} max_{a'} q_θ̄(S_{t+n},a') − q_θ(S_t,A_t))^2. n trades bias/variance.
5. **Distributional RL (C51, Bellemare 2017)**: learn full return distribution not just mean.
   Fixed support z^i = v_min + (i−1)(v_max−v_min)/(N_atoms−1), i=1..N_atoms. Predict masses
   p^i_θ(s,a) via per-action softmax. Target d'_t = (R_{t+1} + γ_{t+1} z, p_θ̄(S_{t+1}, ā*_{t+1})),
   project onto support via Φ_z (L2/linear-interp projection), minimize KL(Φ_z d'_t || d_t).
   ā*_{t+1} = argmax_a z^T p_θ̄(S_{t+1},a) (greedy on mean). N_atoms=51, [v_min,v_max]=[−10,10].
6. **Noisy Nets (Fortunato 2017)**: ε-greedy can't explore deep (Montezuma). FIX: noisy linear
   layer y = (b + Wx) + (b_noisy ⊙ ε^b + (W_noisy ⊙ ε^w)x). Self-annealing, state-conditional
   exploration. Factorised Gaussian noise. σ0=0.5.

## THE INTEGRATION (the actual contribution — how to combine without conflict)
- **Multi-step + distributional**: target distribution contracts value dist at S_{t+n} by the
  cumulative discount and shifts by the n-step return:
  d_t^{(n)} = (R_t^{(n)} + γ_t^{(n)} z, p_θ̄(S_{t+n}, a*_{t+n})).  Loss KL(Φ_z d_t^{(n)} || d_t).
- **+ Double Q**: bootstrap action a*_{t+n} = argmax_a z^T p_θ(S_{t+n}, a)  selected by ONLINE
  net (on the mean), evaluated via TARGET net's distribution p_θ̄(S_{t+n}, a*_{t+n}).
- **+ Prioritized replay**: prioritize by the KL loss itself (what's being minimized), not the
  TD error: p_t ∝ (KL(Φ_z d_t^{(n)} || d_t))^ω. More robust to stochastic returns (KL keeps
  decreasing even when returns nondeterministic).
- **+ Dueling, adapted to distributions**: shared f_ξ(s)=φ; value stream v_η with N_atoms
  outputs; advantage stream a_ψ with N_atoms × N_actions outputs. Per atom i:
  p^i_θ(s,a) = softmax_i( v^i_η(φ) + a^i_ψ(φ,a) − ā^i_ψ(s) ),  ā^i_ψ(s)=(1/N_actions)Σ_{a'} a^i_ψ(φ,a').
  (Dueling aggregation done per-atom BEFORE softmax.)
- **+ Noisy Nets**: replace ALL linear layers with noisy linear (factorised Gaussian). ε=0 (fully
  greedy; exploration entirely from noise).

## Hyperparameters (Table 1)
- Min history to start learning: 80K frames (PER lets you start sooner than DQN's 200K).
- Adam lr = 0.0000625 (= 0.00025/4 = DQN lr /4). Adam ε = 1.5e-4.
- Exploration ε = 0.0 (Noisy Nets does exploration).
- Noisy σ0 = 0.5.
- Target network period = 32K frames.
- Prioritization: proportional, exponent ω = 0.5, IS β: 0.4 → 1.0 (linear over training).
- Multi-step n = 3 (compared 1,3,5; 3 best at end).
- Distributional atoms = 51; min/max = [−10,10].
- Same hyperparameters across all 57 games.

## Why these six (not others)
Each addresses a DISTINCT concern so they're complementary: overestimation (double), sample
efficiency (PER), action-generalization (dueling), reward-propagation/bias-variance (multistep),
representation richness (distributional), exploration (noisy). Deliberately only ONE exploration
method to keep selection manageable.

## Canonical implementation (Kaixhin/Rainbow) — learn():
```
# online net selects a* on the mean of the next-state distribution (n-step ahead)
pns = online_net(next_states); dns = support * pns; a* = dns.sum(2).argmax(1)
target_net.reset_noise(); pns = target_net(next_states); pns_a = pns[range(B), a*]
# distributional Bellman: Tz = R^{(n)} + (gamma^n) * support  (nonterminal mask), clamp [Vmin,Vmax]
Tz = returns.unsqueeze(1) + nonterminals * (discount**n) * support
Tz = Tz.clamp(Vmin, Vmax); b = (Tz - Vmin)/delta_z; l=b.floor(); u=b.ceil()
# project mass onto two nearest atoms
m.index_add_(0, l, pns_a*(u-b)); m.index_add_(0, u, pns_a*(b-l))   # (+ integer-b fix)
# cross-entropy (KL up to const), IS-weighted; priority = per-sample loss
loss = -sum(m * log_ps_a, dim=1); (weights*loss).mean().backward()
mem.update_priorities(idxs, loss)
```
Network: NoisyLinear everywhere; dueling head (value N_atoms, advantage N_atoms×A);
per-atom aggregation v + a − mean(a); per-action softmax over atoms.

## Design-decision table
| choice | why | rejected alt |
|---|---|---|
| combine 6 distinct-concern extensions | each fixes a different DQN limitation; complementary | one-at-a-time (misses synergy) |
| double-Q via online-select/target-eval on dist | reduce overestimation, reuse online greedy | C51's target-net greedy (no decoupling) |
| prioritize by KL loss not |TD| | it's what's minimized; robust to stochastic returns | |TD error| (paper notes KL more robust) |
| dueling aggregation per-atom before softmax | dueling defined per-atom logit; softmax normalizes | aggregate on means (loses distribution) |
| n=3 multistep | best final performance (1,3,5 swept) | n=1 (slow propagation), n=5 (worse at end) |
| Noisy Nets, ε=0 | deep state-conditional exploration; removes ε schedule | ε-greedy (shallow exploration) |
| Adam lr = DQN/4 | Adam less lr-sensitive; /4 chosen from {/2,/4,/6} | RMSProp (DQN default) |
| start learning at 80K | PER lets earlier start (vs DQN 200K) | 200K (unnecessary with PER) |
| multistep+dist: contract z by γ^{(n)}, shift by R^{(n)} | distributional analogue of n-step target | 1-step dist target (slow) |
| ω=0.5 priority exponent | tuned {0.4,0.5,0.7}; robust under KL priority | larger ω (sharper, less stable) |

# Provable self-play for two-player zero-sum Markov games

## Problem

In a tabular episodic two-player zero-sum Markov game
$\mathrm{MG}(H,\mathcal S,\mathcal A,\mathcal B,\mathbb P,r)$ — shared state, simultaneous
actions $a\in\mathcal A$ (max) and $b\in\mathcal B$ (min), reward $r_h(s,a,b)\in[0,1]$,
unknown $\mathbb P,r$, no simulator — learn an approximate **Nash equilibrium** by *self-play*:
the learner controls both players, repeatedly plays its current pair against itself, and updates.
Performance is the **strong regret** against per-episode best responses,
$$\mathrm{Regret}(K)=\sum_{k=1}^K\Big[V^{\dagger,\nu^k}_1(s_1^k)-V^{\mu^k,\dagger}_1(s_1^k)\Big],$$
each summand a non-negative duality gap $\big(V^{\dagger,\nu^k}_1-V^\star_1\big)+\big(V^\star_1-V^{\mu^k,\dagger}_1\big)$.
The target is sublinear regret and a PAC bound for an $\epsilon$-Nash pair,
$V^{\dagger,\hat\nu}_1(s_1)-V^{\hat\mu,\dagger}_1(s_1)\le\epsilon$ — good against best responses,
not merely near in value.

## Key idea

Lift single-agent optimism (UCBVI) to two players with **two-sided optimism**: maintain an upper
estimate $Q^{\mathrm{up}}$ and a lower estimate $Q^{\mathrm{low}}$, each from a bonus-augmented
Bellman backup. A single player cannot be "greedy" in a matrix payoff $Q(s,a,b)$ alone, so at
each state solve a **one-step matrix game** for both policies *jointly*. Because the two players
carry *different* matrices ($Q^{\mathrm{up}}\neq Q^{\mathrm{low}}$), the per-state solve is a
**general-sum Nash equilibrium**. The optimism must bracket the deployed policies' *best-response*
values — $Q^{\mathrm{up}}\ge\sup_\mu Q^{\mu,\nu^k}\ge\inf_\nu Q^{\mu^k,\nu}\ge Q^{\mathrm{low}}$ —
which is exactly what makes the up–low band upper-bound the regret summand.

## VI-ULCB (value iteration with upper–lower confidence bounds)

For each episode, backward sweep over $h$: at every $(s,a,b)$,
$$Q^{\mathrm{up}}_h\leftarrow\min\{r_h+[\widehat{\mathbb P}_hV^{\mathrm{up}}_{h+1}]+\beta_t,\,H\},\qquad
Q^{\mathrm{low}}_h\leftarrow\max\{r_h+[\widehat{\mathbb P}_hV^{\mathrm{low}}_{h+1}]-\beta_t,\,0\},$$
with $\beta_t=c\sqrt{SH^2\iota/t}$, $t=N_h(s,a,b)$, $\iota=\log(SABT/p)$; at every $s$,
$$(\mu_h(\cdot\mid s),\nu_h(\cdot\mid s))=\textsc{Nash\_General\_Sum}(Q^{\mathrm{up}}_h(s,\cdot,\cdot),Q^{\mathrm{low}}_h(s,\cdot,\cdot)),\quad
V^{\mathrm{up/low}}_h(s)=\mu_h(s)^\top Q^{\mathrm{up/low}}_h(s,\cdot,\cdot)\nu_h(s).$$
Then play one episode sampling $a_h\sim\mu_h(s_h),b_h\sim\nu_h(s_h)$ and update counts.

**Theorem (regret).** With probability $\ge1-p$, $\mathrm{Regret}(K)=\tilde O(\sqrt{H^3S^2ABT})$.
**Corollary (PAC).** Sampling $(\hat\mu,\hat\nu)$ uniformly over episodes gives an $\epsilon$-Nash
pair once $K\ge\tilde O(H^4S^2AB/\epsilon^2)$.

*Proof sketch.* (1) **ULCB lemma:** by downward induction in $h$, $Q^{\mathrm{up},k}_h\ge\sup_\mu Q^{\mu,\nu^k}_h$
and $Q^{\mathrm{low},k}_h\le\inf_\nu Q^{\mu^k,\nu}_h$; the $Q$-step uses positivity of $\widehat{\mathbb P}$
plus concentration, the $V$-step uses that $\mu^k$ is the $Q^{\mathrm{up}}$-best-response to $\nu^k$
inside the equilibrium. Hence the sandwich and $\mathrm{Regret}(K)\le\sum_k(V^{\mathrm{up},k}_1-V^{\mathrm{low},k}_1)(s_1^k)$.
(2) **Concentration:** an $\ell_\infty$ $\epsilon$-net of $\{V:\mathcal S\to[0,H]\}$ has size $(1/\epsilon)^S$;
Hoeffding + union bound + $\epsilon=c\sqrt{H^2S\iota/K}$ give $|(\widehat{\mathbb P}-\mathbb P)V|\le c\sqrt{SH^2\iota/N}$
uniformly — fixing $\beta_t$ (the $\epsilon$-net $S$ is the source of the $S^2$).
(3) **Telescope:** $(V^{\mathrm{up}}-V^{\mathrm{low}})(s^k_h)\le(V^{\mathrm{up}}-V^{\mathrm{low}})(s^k_{h+1})+4\beta^k_h+\xi^k_h+\zeta^k_h$
with $\xi,\zeta$ martingale differences; unroll, sum, use $\sum_{t=1}^N\sqrt{1/t}=O(\sqrt N)$ and
Cauchy–Schwarz $\sum_{s,a,b}\sqrt{N_h}\le\sqrt{SAB\cdot K}$, then sum over $H$ steps:
$\tilde O(\sqrt{H^4S^2AB\,K})=\tilde O(\sqrt{H^3S^2ABT})$.

## Recovering polynomial runtime

$\textsc{Nash\_General\_Sum}$ is PPAD-complete, so VI-ULCB is sample-efficient but not poly-time.

- **VI-Explore (explore-then-exploit).** Reward-free exploration (per target state $s$, set
  $\tilde r=\mathbb 1[s'=s]$, treat $(a,b)$ as one action, run a single-agent PAC learner) yields an
  empirical model; value iteration on it needs only $\textsc{Nash\_Zero\_Sum}$ (one matrix, von
  Neumann minimax, LP). Polynomial time, regret $\tilde O((H^5S^2ABT^2)^{1/3})=\tilde O(T^{2/3})$.

## Optimistic Nash Q-learning (model-free, $S$-linear)

Replace the general-sum Nash by a **Coarse Correlated Equilibrium** (always exists, $A+B$ linear
constraints → LP → polynomial; if $\pi=\textsc{CCE}(Q,Q)$, its marginals are a zero-sum Nash
because $\max_a\mathbb E_b Q(a,b)\le\mathbb E_\pi Q\le\min_b\mathbb E_a Q(a,b)$ combines with
the minimax chain $\min_b\mathbb E_a Q(a,b)\le N^\star\le\max_a\mathbb E_b Q(a,b)$), and the model-based
backup by an online Q-learning update so each sample is used once on an independent continuation:
$$Q^{\mathrm{up}}_h(s_h,a_h,b_h)\leftarrow(1-\alpha_t)Q^{\mathrm{up}}_h+\alpha_t(r_h+V^{\mathrm{up}}_{h+1}(s_{h+1})+\beta_t),\quad
\alpha_t=\tfrac{H+1}{H+t},\ \beta_t=c\sqrt{H^3\iota/t},$$
$\pi_h(\cdot,\cdot\mid s_h)=\textsc{CCE}(Q^{\mathrm{up}}_h(s_h,\cdot,\cdot),Q^{\mathrm{low}}_h(s_h,\cdot,\cdot))$.

**Theorem.** $V^{\mathrm{up},k}_h\ge V^\star_h\ge V^{\mathrm{low},k}_h$ and
$\frac1K\sum_k(V^{\mathrm{up},k}_1-V^{\mathrm{low},k}_1)(s_1)\le O(\sqrt{H^5SAB\iota/K})$; the
**certified policy** is $\epsilon$-Nash after $\tilde O(H^5SAB\iota/\epsilon^2)$ episodes.

**Certified policy (why value $\neq$ policy).** Unrolling the update,
$Q^{\mathrm{up},k}_h(s,a,b)=\alpha^0_tH+\sum_{i=1}^t\alpha^i_t[r_h+V^{\mathrm{up},k^i_h(s,a,b)}_{h+1}(\cdot)+\beta_i]$
with $\alpha^0_t=\prod_{j\le t}(1-\alpha_j),\ \alpha^i_t=\alpha_i\prod_{j>i}(1-\alpha_j),\ \sum_i\alpha^i_t=1$.
So $Q^{\mathrm{up}}$ is an upper certificate, with the initial $H$ term and bonuses, for playing the
$\alpha$-weighted mixture of past per-step policies; the certified $\hat\mu$ samples an episode
index and re-samples it from $\{\alpha^i_t\}$ at each node — a non-Markovian **nested mixture**
certified against best responses. (A value-only band does not certify a policy in a game; this
construction is what closes the gap.)

## Optimistic Nash V-learning (near-optimal, $A+B$)

Per state, run **Follow-the-Regularized-Leader** (regret independent of the opponent's action count)
on importance-weighted losses, blended with the Q-learning $V$-backup:
$$\hat\ell_h(s_h,a)=\frac{[H-r_h-V^{\mathrm{up}}_{h+1}(s_{h+1})]\,\mathbb 1[a_h=a]}{\mu_h(a_h\mid s_h)+\eta_t},\quad
\mu_h\leftarrow\arg\min_{\mu\in\Delta_{\mathcal A}}\eta_t\langle L_h,\mu\rangle+\alpha_t\mathrm{KL}(\mu\|\mu_0)\propto e^{-(\eta_t/\alpha_t)L_h},$$
$\eta^{\mathrm{up}}_t=\sqrt{\log A/(At)},\ \beta^{\mathrm{up}}_t=c\sqrt{H^4A\iota/t}$ (min-player symmetric).

**Theorem.** The certified pair is $\epsilon$-Nash after $\tilde O(H^6S(A+B)\iota/\epsilon^2)$ episodes —
matching the lower bound $\Omega(H^3S(A+B)/\epsilon^2)$ up to $\mathrm{poly}(H)$, and decentralized
(each player needs only the opponent's action).

## Lower bounds and why self-play is necessary

Collapsing the game to a single-agent MDP gives $\mathrm{Regret}=\Omega(\sqrt{H^2S(A+B)T})$, so the
above rates are sublinear and (for V-learning) near-optimal in $(S,A,B)$. Crucially, **learning a
best response to a fixed (possibly non-Markovian) opponent is as hard as learning parity with
noise** (conjectured super-polynomial): a deterministic two-action game with two states per layer
can encode a parity instance in the opponent's policy. The no-regret hardness against adversarial opponents follows
from the same construction and remains for non-adaptive Markov policies across episodes. Hence
achieving no-regret against an *adversarial* opponent is computationally hard, which rules out the natural "two no-regret learners against each other"
recipe. Self-play — two coordinated, two-sided-optimistic players solving a per-state equilibrium —
is what makes sample-efficient Nash learning tractable.

## Implementation sketch

```python
import numpy as np

def cce(P, Q):
    """CCE of bimatrix (P,Q) via LP: pi over A*B with
       E_pi P >= max_a* E_pi P(a*,b),  E_pi Q <= min_b* E_pi Q(a,b*).
       A+B linear constraints -> polynomial; CCE(Q,Q)'s marginals are zero-sum Nash."""
    A, B = P.shape
    return solve_cce_lp(P, Q).reshape(A, B)

def nash_q_learning(env, S, A, B, H, K, c, iota):
    Qup  = np.full((H, S, A, B), float(H))      # two-sided optimism: upper Q
    Qlow = np.zeros((H, S, A, B))               #                     lower Q
    N    = np.zeros((H, S, A, B))
    pi   = np.full((H, S, A, B), 1.0 / (A * B))
    Vup  = np.zeros((H + 1, S)); Vlow = np.zeros((H + 1, S))
    for k in range(K):
        s = env.reset()
        for h in range(H):
            a, b = sample_joint(pi[h, s])                       # (a,b) ~ pi_h(.,.|s)
            r, s_next = env.step(a, b); N[h, s, a, b] += 1
            t = int(N[h, s, a, b]); alpha = (H + 1) / (H + t)
            beta = c * np.sqrt(H**3 * iota / t)
            Qup[h, s, a, b]  = (1-alpha)*Qup[h, s, a, b]  + alpha*(r + Vup[h+1, s_next]  + beta)
            Qlow[h, s, a, b] = (1-alpha)*Qlow[h, s, a, b] + alpha*(r + Vlow[h+1, s_next] - beta)
            pi[h, s] = cce(Qup[h, s], Qlow[h, s])              # per-state equilibrium solve
            Vup[h, s]  = float(np.sum(pi[h, s] * Qup[h, s]))   # E_pi Qup
            Vlow[h, s] = float(np.sum(pi[h, s] * Qlow[h, s]))  # E_pi Qlow
            s = s_next
    return pi, N        # eps-Nash policy extracted via the certified (nested-mixture) procedure
```

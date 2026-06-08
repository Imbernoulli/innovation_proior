# TD(λ) and Eligibility Traces

## Problem
Learn the value function v_π(s) = E_π[G_t | S_t = s] from a stream of experience, with a
single knob that interpolates smoothly between one-step TD (fast, online, cheap, but
propagates credit one step at a time) and Monte Carlo (full credit propagation, but
end-of-episode only, high variance, only usable as stated when episodes terminate) — and
do it **online, causally, and with memory/compute that do not grow with how far back
credit must travel.**

## Key idea
Average *all* n-step returns with geometrically decaying weights — the **λ-return** — and
replace the acausal forward-looking update with a backward-running **eligibility trace**
that decays the credit assigned to recently visited states. With weights fixed during an
episode, the forward λ-return view and the backward accumulating-trace view produce
identical total weight changes. A dutch trace gives the exact strict-online counterpart.

## The λ-return (forward view)
Every n-step return G_{t:t+n} = R_{t+1}+γR_{t+2}+…+γ^{n−1}R_{t+n}+γ^n v̂(S_{t+n}) is a valid
target (error-reduction: max_s |E_π[G_{t:t+n} | S_t=s]−v_π(s)| ≤ γ^n max_s|v̂(s)−v_π(s)|),
and so is any convex average of them. The geometric average, normalized to sum to 1, is
the λ-return:

  G^λ_t = (1−λ) Σ_{n=1}^∞ λ^{n−1} G_{t:t+n}
        = (1−λ) Σ_{n=1}^{T−t−1} λ^{n−1} G_{t:t+n} + λ^{T−t−1} G_t.

Weights: the n-step return gets (1−λ)λ^{n−1}. λ=0 ⇒ G^λ_t = one-step return (TD(0));
λ=1 ⇒ G^λ_t = full return G_t (Monte Carlo). Recursive form:
G^λ_t = R_{t+1} + γ[(1−λ)v̂(S_{t+1}) + λ G^λ_{t+1}].
Off-line λ-return algorithm: accumulate α[G^λ_t − v̂(S_t,w)]∇v̂(S_t,w) with w fixed through
the episode, then apply the accumulated change. This is acausal — G^λ_t needs rewards
arbitrarily far ahead.

## TD(λ) (backward view, accumulating trace)
Short-term trace z_t (same dimension as w), bumped by the gradient and faded by γλ:

  z_{−1} = 0,   z_t = γλ z_{t−1} + ∇v̂(S_t,w_t).
  δ_t = R_{t+1} + γ v̂(S_{t+1},w_t) − v̂(S_t,w_t).
  w_{t+1} = w_t + α δ_t z_t.

Tabular trace: e_t(s) = γλ e_{t−1}(s) + 1_{s=S_t}. Causal, online, O(d), no n-step buffer.
λ=0 ⇒ z_t=∇v̂(S_t) ⇒ TD(0); λ=1 leaves only the γ discount decay, and with γ=1 the trace
does not decay in an episodic task. The decay is γλ, not λ or γ alone: a TD error passed k
steps back carries a reward k steps deeper in the return (factor γ^k) weighted by the
λ-return's horizon decay (factor λ^k); their product is (γλ)^k.

## Forward ⇔ backward equivalence theorem (offline, tabular)
With weights held fixed during the episode, total updates coincide for every state s:

  Σ_{t=0}^{T−1} ΔV^{TD}_t(s) = Σ_{t=0}^{T−1} ΔV^λ_t(S_t) 1_{s=S_t}.

**Proof.**
*Left side (backward).* The accumulating trace is, explicitly,
e_t(s) = Σ_{k=0}^t (γλ)^{t−k} 1_{s=S_k}. Hence
  Σ_j ΔV^{TD}_j(s) = Σ_{j=0}^{T−1} α δ_j Σ_{i=0}^j (γλ)^{j−i} 1_{s=S_i}
                   = Σ_{i=0}^{T−1} α 1_{s=S_i} Σ_{j=i}^{T−1} (γλ)^{j−i} δ_j,
by swapping the order of the triangular double sum (visit index i ≤ error index j).

*Right side (forward).* Expand one λ-return error by reward-columns. The reward
R_{t+j+1} collects coefficient γ^jΣ_{n≥j+1}(1−λ)λ^{n−1}=(γλ)^j, and each bootstrap value
term splits into the full value needed for the current TD error plus the −γλ carry that
seeds the next column:
  G^λ_t − V(S_t) = (γλ)^0 δ_t + (γλ)^1 δ_{t+1} + (γλ)^2 δ_{t+2} + … = Σ_{k=t}^{T−1} (γλ)^{k−t} δ_k
(omitted k ≥ T terms vanish: zero post-terminal rewards/values). Therefore
  Σ_t ΔV^λ_t(S_t) 1_{s=S_t} = Σ_{t=0}^{T−1} α 1_{s=S_t} Σ_{k=t}^{T−1} (γλ)^{k−t} δ_k,
identical to the left side. ∎

Online (each update applied immediately), V drifts between steps so the δ's no longer all
align: TD(λ) then *approximates* the λ-return, exactly only as α→0.

## Eligibility traces are not specific to TD (dutch-trace MC)
Even offline linear Monte Carlo / LMS, w_{t+1}=w_t+α[G−w_t^Tx_t]x_t (single terminal G, no
discount), reorganizes into an exact incremental trace. With F_t = I − αx_t x_t^T,
  w_T = a_{T−1} + αG z_{T−1},  z_t = z_{t−1} + (1 − α z_{t−1}^T x_t)x_t,  a_t = F_t a_{t−1},
where z_{−1}=0 and a_{−1}=w_0. These recursions reproduce the end-of-episode update at
O(d)/step with no stored history. The
self-correcting "dutch" trace this reveals,
  z_t = γλ z_{t−1} + (1 − αγλ z_{t−1}^T x_t)x_t,
  w_{t+1} = w_t + αδ_t z_t + α(w_t^T x_t − w_{t−1}^T x_t)(z_t − x_t),
gives **true online TD(λ)**, which reproduces the strict online λ-return target *exactly* in
the linear case, with the same O(d) memory/asymptotic cost as plain TD(λ) plus one extra
inner product.

## Implementation
```python
import numpy as np

class TDLambda:
    """Accumulating-trace TD(λ): offline-exact, online-approximate forward equivalence."""
    def __init__(self, d, gamma, lam, alpha):
        self.w = np.zeros(d); self.gamma, self.lam, self.alpha = gamma, lam, alpha
    def reset_episode(self):
        self.z = np.zeros_like(self.w)
    def step(self, x, r, x_next, terminal):
        v      = self.w @ x
        v_next = 0.0 if terminal else self.w @ x_next
        delta  = r + self.gamma * v_next - v               # δ_t  (TD error)
        self.z = self.gamma * self.lam * self.z + x        # z_t = γλ z + ∇v̂
        self.w += self.alpha * delta * self.z              # w += α δ_t z_t

def lambda_return(rewards, values, gamma, lam):
    """Forward λ-return targets (values[T]=0), with weights fixed for the episode."""
    T = len(rewards); G = np.zeros(T); G[T-1] = rewards[T-1]
    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma * ((1-lam)*values[t+1] + lam*G[t+1])
    return G

class TrueOnlineTDLambda:
    """Dutch-trace TD(λ): exact for the strict online λ-return view, O(d)/step."""
    def __init__(self, d, gamma, lam, alpha):
        self.w = np.zeros(d); self.gamma, self.lam, self.alpha = gamma, lam, alpha
    def reset_episode(self, x):
        self.z = np.zeros_like(self.w); self.v_old = 0.0; self.x = x
    def step(self, r, x_next, terminal):
        x = self.x; v = self.w @ x
        v_next = 0.0 if terminal else self.w @ x_next
        delta  = r + self.gamma * v_next - v
        gl     = self.gamma * self.lam
        self.z = gl * self.z + (1 - self.alpha * gl * (self.z @ x)) * x   # dutch trace
        self.w += self.alpha * (delta + v - self.v_old) * self.z
        self.w -= self.alpha * (v - self.v_old) * x
        self.v_old = v_next; self.x = x_next
```

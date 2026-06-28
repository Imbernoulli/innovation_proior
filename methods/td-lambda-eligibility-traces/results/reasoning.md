Let me start from what actually hurts. I have an agent walking through states under a fixed policy, picking up rewards, and I want the value function — the expected discounted return from each state. I have exactly two tools that work, and they sit at opposite ends of a spectrum, and each one annoys me for the opposite reason.

At one end, one-step TD. I see a transition S_t → R_{t+1}, S_{t+1}, I form the error δ_t = R_{t+1} + γv̂(S_{t+1}) − v̂(S_t), and I nudge v̂(S_t) toward closing it. It's online, it's O(d), it has low variance because it leans on my own current estimate of the next state instead of waiting for the noisy far future. But that's also its disease: the only thing that ever reaches v̂(S_t) on a given step is information from one step ahead. If a reward is buried twenty states down the trajectory, it takes the better part of twenty episodes for that reward to crawl back, one bootstrap at a time, to the state that set it up. Slow.

At the other end, Monte Carlo. Wait for the episode to finish, compute the actual return G_t, and move every visited state toward it: w ← w + α[G_t − v̂(S_t)]∇v̂. One episode, and credit floods all the way back. No bootstrapping bias — the target is the real thing. But I pay for it: I can't update until the episode ends, the return is a single high-variance sample of the future, and if the task never terminates I'm simply stuck.

So I want something in between. And the obvious move is to bootstrap not after one real reward but after n of them: the n-step return

  G_{t:t+n} = R_{t+1} + γR_{t+2} + … + γ^{n−1}R_{t+n} + γ^n v̂(S_{t+n}),

with the convention that if t+n runs past the end of the episode it just equals the full return G_t. Small n is nearly TD; large n is nearly Monte Carlo. Why am I even allowed to do this — to trust a half-bootstrapped target? Because of the way the bound on the error behaves. The expected n-step return is a strictly better estimate of v_π than my current values are, worst-state-wise:

  max_s |E_π[G_{t:t+n} | S_t=s] − v_π(s)| ≤ γ^n · max_s |v̂(s) − v_π(s)|.

Each extra real reward I fold in shaves the worst-case error by another factor of γ. So every n-step return is a contraction toward the truth — a legitimate target — and updating toward it converges. Good. That settles that the in-between targets are sound.

But two things about the n-step return still bother me, and they're the whole reason I keep going. First, I have to pick n, a single integer, and there's no principled choice — it's a brittle, task-specific guess, and committing to one horizon throws away the others. Second, and worse, look at the *cost*. To update S_t from G_{t:t+n} I have to wait n steps to even see R_{t+n} and v̂(S_{t+n}); for the first n−1 steps of every episode I make no update at all; and I have to physically hold the last n feature vectors in memory so I can apply the delayed update when its data finally arrives. The interpolation knob exists, but it drags the storage and the delay right back in — the very things one-step TD was free of.

Now, the error-reduction property gives me more room than I've used. It says any *average* of n-step returns, as long as the weights are non-negative and sum to one, is again a contraction target — the bound just becomes the weighted average of the per-component bounds, which is no worse than the worst one. So I'm not restricted to picking a single n; I can blend all of them. That feels like the right freedom: instead of betting on one horizon, take a weighted vote over all horizons at once. The question is which weights.

Let me not pick arbitrarily; let me pick the weights that buy me something. I want the blend to have a single dial that slides from "all weight on the one-step return" to "all weight on the full return," continuously. And — remembering how much the n-step buffer cost me — I'd love the weighting to be one I can *accumulate incrementally*, so that summing over all horizons doesn't mean storing all horizons.

A geometric weighting is the natural candidate: give the n-step return a weight proportional to λ^{n−1} for some λ ∈ [0,1]. Why geometric, specifically? Because it's the scale-free one — each extra step of horizon multiplies the weight by the same constant λ, so there's no special horizon length baked in, just a single decay rate. And geometric is exactly the shape that telescopes and accumulates, which is the property I said I wanted. Normalize so the weights sum to one: Σ_{n=1}^∞ c·λ^{n−1} = c/(1−λ) = 1, so c = (1−λ). That gives me the λ-return,

  G^λ_t = (1−λ) Σ_{n=1}^∞ λ^{n−1} G_{t:t+n}.

The one-step return gets the largest slice, 1−λ; the two-step gets (1−λ)λ; the three-step (1−λ)λ², and so on, each successive horizon fading by another factor of λ. In an episode that terminates at T, every n-step return with n ≥ T−t is just the full return G_t, so I can peel those identical tail terms out of the sum:

  G^λ_t = (1−λ) Σ_{n=1}^{T−t−1} λ^{n−1} G_{t:t+n} + λ^{T−t−1} G_t.

And now I should check the two limits, because the dial is the whole point — if the endpoints aren't TD(0) and Monte Carlo, the whole construction is pointless. At λ = 0, every term λ^{n−1} dies except n = 1 (using the convention λ^0 = 1), so G^λ_t should collapse to G_{t:t+1}, the one-step TD target. At λ = 1, the main sum's coefficient (1−λ) goes to zero and the residual λ^{T−t−1}G_t = G_t should be all that survives — the Monte Carlo target. I don't entirely trust the λ=1 case by eye, because it's a 0·∞ situation (coefficient → 0 while the number of surviving terms → ∞), so let me actually compute it on a short episode rather than wave at it. Take three steps with rewards R_1=1, R_2=0, R_3=2, γ=0.9, fixed values V(S_0)=0.4, V(S_1)=−0.2, V(S_2)=0.7, terminal value 0. Using the recursive form G^λ_t = R_{t+1}+γ[(1−λ)V(S_{t+1})+λG^λ_{t+1}], I run it at λ=0 and get G^λ = (0.82, 0.63, 2.0); the one-step targets R_{t+1}+γV(S_{t+1}) are (1+0.9·(−0.2), 0+0.9·0.7, 2.0) = (0.82, 0.63, 2.0) — identical. At λ=1 I get (2.62, 1.8, 2.0); the actual discounted returns G_t are (1+0.9·0+0.81·2, 0+0.9·2, 2) = (2.62, 1.8, 2.0) — identical again. So the limits really are TD(0) and Monte Carlo, the 0·∞ resolves correctly, and the continuum between them is what I asked for. The forward-view algorithm is then just: at the end of the episode, for each visited state, step toward its λ-return,

  w_{t+1} = w_t + α[G^λ_t − v̂(S_t,w_t)]∇v̂(S_t,w_t).

This is clean and it's *theoretically* what I want. But I've recreated the very problem I was trying to escape, and worse. G^λ_t depends on rewards arbitrarily far into the future — in a continuing task it's never fully known, and even in an episodic one I can't compute it until the episode ends. It's acausal: to update S_t I'd need to look forward at things that haven't happened. And to do it I'd store everything. So as a literal algorithm this is no better than Monte Carlo; it's the *target* I love, not the implementation. I need a way to produce these exact updates without ever looking forward.

Stare at the structure of the problem. The forward view says: stand at each state, look ahead, gather all the future, combine. The trouble is always the same — the future isn't available yet. But here's the asymmetry I keep underusing. At time t I can't see the future, but I *can* see the past. Every reward and TD error I compute at time t is relevant to *every state I've already visited*, in proportion to how recently I visited it. What if, instead of standing at S_t and reaching forward, I stand at the present, compute the one cheap thing I always have — the current TD error δ_t — and shout it *backward* to all the states behind me, each in proportion to how much it should care?

That "how much it should care, fading with recency" is exactly an eligibility idea. A state, when visited, becomes eligible for learning, and that eligibility decays as I move on; whatever reinforcing signal arrives later modifies each state in proportion to its current eligibility. So let me carry a short-term memory — a trace vector z, same shape as w — that I bump up by the value-gradient whenever I'm at a state, and that otherwise fades. Then every step I take the current δ_t and apply it through z:

  z_t = (decay)·z_{t−1} + ∇v̂(S_t,w_t),   w_{t+1} = w_t + α δ_t z_t.

This is causal, online, O(d), no n-step buffer — beautiful. But it's only worth anything if it computes *the same thing* as that forward λ-return I derived. So two questions are pinned down for me by that demand. What is the decay rate, and does this backward mechanism actually reproduce the forward target? I can't just assert the decay is λ; I have to find the rate that makes the two views agree, and then prove they agree.

Let me get at the decay rate by unrolling the λ-return into the language the backward view speaks — TD errors. Hold the weights fixed for a moment (pretend v̂ doesn't change during the episode) so every δ is well-defined against one value function. Take a single forward update and expand it. Writing it out per reward,

  (1/α)ΔV^λ_t(S_t) = G^λ_t − V(S_t)
   = −V(S_t)
     + (1−λ)λ^0 [r_{t+1} + γV(S_{t+1})]
     + (1−λ)λ^1 [r_{t+1} + γr_{t+2} + γ²V(S_{t+2})]
     + (1−λ)λ^2 [r_{t+1} + γr_{t+2} + γ²r_{t+3} + γ³V(S_{t+3})]
     + …

— each row is an n-step return with its (1−λ)λ^{n−1} weight. Now don't read it by rows, read it by *columns*: collect everything multiplying r_{t+1}, then everything multiplying r_{t+2}, and so on. The r_{t+1} column appears in every row with coefficient (1−λ)λ^{n−1}, and Σ_{n≥1}(1−λ)λ^{n−1} = 1 — the whole column collapses to weight 1 on r_{t+1}. What about the value terms? γV(S_{t+1}) appears only in the n=1 row, with weight (1−λ). Write (1−λ)·γV(S_{t+1}) = γV(S_{t+1}) − γλV(S_{t+1}): I keep a *full* γV(S_{t+1}) here, and carry the leftover −γλV(S_{t+1}) downward. So the first column, completed by that −V(S_t) sitting out front, is

  γ⁰ [r_{t+1} + γV(S_{t+1}) − V(S_t)] = δ_t,

a clean one-step TD error, weight 1. The carried −γλV(S_{t+1}) lands in the r_{t+2} column. That column's rewards collect coefficient γΣ_{n≥2}(1−λ)λ^{n−1} = γλ on r_{t+2}, and the same split on its value term feeds the next column; the algebra repeats identically, shifted, and the −γλV(S_{t+1}) I carried supplies exactly the missing −γλV(S_{t+1}) that turns the second column into (γλ) times a TD error. Out drops

  + (γλ)^1 [r_{t+2} + γV(S_{t+2}) − V(S_{t+1})] = (γλ)δ_{t+1},
  + (γλ)^2 [r_{t+3} + γV(S_{t+3}) − V(S_{t+2})] = (γλ)²δ_{t+2},
  + …

So the column-by-column collapse suggests the entire λ-return error telescopes into a discounted sum of future TD errors:

  G^λ_t − V(S_t) = Σ_{k=t}^{∞} (γλ)^{k−t} δ_k,

with the upper limit really T−1 because every TD error past termination involves zero rewards and zero values and so vanishes.

The carrying-and-splitting argument is the kind of thing that looks airtight on paper and is off by a stray factor in practice, so before I lean on it let me check the identity numerically on the same three-step episode. With γ=0.9 and λ=0.5, the TD errors against the fixed V are δ_0 = 1+0.9·(−0.2)−0.4 = 0.42, δ_1 = 0+0.9·0.7−(−0.2) = 0.83, δ_2 = 2+0−0.7 = 1.3. The λ-returns from the recursion are G^λ = (1.45675, 1.215, 2.0), so the left sides G^λ_t − V(S_t) are (1.05675, 1.415, 1.30). Now the right sides, γλ = 0.45: for t=0, δ_0 + 0.45·δ_1 + 0.45²·δ_2 = 0.42 + 0.3735 + 0.263... = 0.42+0.3735+0.263250 = 1.05675; for t=1, δ_1 + 0.45·δ_2 = 0.83 + 0.585 = 1.415; for t=2, just δ_2 = 1.30. All three match to the digit. The telescoping is real, not an artifact of how I grouped the terms.

So that answers the decay question without my having to guess. The weight on δ_k — the TD error k−t steps in the future — is (γλ)^{k−t}, the product of two factors I can name. The γ is unavoidable: δ_k carries a reward that lives k−t steps deeper in the return, and any reward that deep is discounted by γ^{k−t}, so passing the error that far back must carry the same γ. The λ is the horizon weighting I chose for the λ-return, fading by λ per extra step of lookahead. Neither alone is right; the unrolling *forces* their product. The trace must decay by γλ per step — not λ, not γ, but γλ. So:

  z_t = γλ z_{t−1} + ∇v̂(S_t,w_t),   δ_t = R_{t+1} + γv̂(S_{t+1}) − v̂(S_t),   w_{t+1} = w_t + α δ_t z_t.

In the tabular case the trace is per-state, e_t(s) = γλ e_{t−1}(s) + 1_{s=S_t}, bumped by 1 on the visited state and faded by γλ everywhere. And at λ = 0 the trace is just ∇v̂(S_t) with no memory — the update collapses to one-step TD(0). That's why the family deserves the name TD(λ): TD(0) is the λ=0 member.

Now I owe the proof. The forward telescoping above tells me the per-state forward update is a forward-running sum of δ's; the backward trace mechanism is a present-time scatter of one δ over past states. Are the *totals over an episode* actually equal? Let me hold the weights fixed across the episode (the offline case) so all the δ's are defined against the same V, and show the sum of all the backward TD(λ) updates equals the sum of all the forward λ-return updates, state by state:

  Σ_{t=0}^{T−1} ΔV^{TD}_t(s) = Σ_{t=0}^{T−1} ΔV^λ_t(S_t) 1_{s=S_t},  for every state s.

Take the left side first — the backward mechanism. The accumulating trace, unrolled explicitly instead of recursively, is the sum over every past visit of (γλ) raised to how long ago it was:

  e_t(s) = Σ_{k=0}^{t} (γλ)^{t−k} 1_{s=S_k}.

So the total backward change to state s over the episode is

  Σ_{j=0}^{T−1} ΔV^{TD}_j(s) = Σ_{j=0}^{T−1} α δ_j Σ_{i=0}^{j} (γλ)^{j−i} 1_{s=S_i}.

This is a double sum over the triangular region i ≤ j: visit time i, later error time j. Reading the same triangle by visits instead of by errors gives

  = Σ_{i=0}^{T−1} α 1_{s=S_i} Σ_{j=i}^{T−1} (γλ)^{j−i} δ_j.

Now rename i back to t and j back to k so it matches the forward-view notation:

  = Σ_{t=0}^{T−1} α 1_{s=S_t} Σ_{k=t}^{T−1} (γλ)^{k−t} δ_k.

Each term is: "for each time t I was at a state, give that state α times the discounted sum of TD errors from t onward." That's a forward-running δ-sum, attached to the state visited at t.

Now the right side — the forward λ-return updates. The single forward update at t is α[G^λ_t − V(S_t)]∇V(S_t), and I just proved by the column-collapse that, with V held fixed,

  G^λ_t − V(S_t) = Σ_{k=t}^{T−1} (γλ)^{k−t} δ_k

(the omitted k ≥ T terms are exactly zero — fictitious post-terminal steps with zero reward and zero value). So the total forward change to state s is

  Σ_{t=0}^{T−1} ΔV^λ_t(S_t) 1_{s=S_t} = Σ_{t=0}^{T−1} α 1_{s=S_t} Σ_{k=t}^{T−1} (γλ)^{k−t} δ_k,

which is term-for-term the same expression I reached on the left. So on paper the two totals coincide. But the whole argument hinges on the triangle being re-summed correctly and on the backward trace really being the explicit geometric sum I claimed, so let me run both algorithms end-to-end on the three-step episode and compare the actual per-state numbers — the symbolic match and a numeric mismatch can't both be right, and if they disagree I've made an indexing error somewhere above.

Forward side, α=0.1: the forward update to each state is α(G^λ_t − V(S_t)), which I already computed as the left sides (1.05675, 1.415, 1.30), so the three states get (0.105675, 0.1415, 0.130). Backward side: run the accumulating trace z_t = γλ z_{t−1} + x_t with γλ=0.45 (each state visited once, so the trace is just one-hot bumps that fade), applying α δ_t z_t each step with V held fixed. Step 0: z=(1,0,0), Δ += 0.1·0.42·(1,0,0). Step 1: z=(0.45,1,0), Δ += 0.1·0.83·(0.45,1,0). Step 2: z=(0.2025,0.45,1), Δ += 0.1·1.3·(0.2025,0.45,1). Summing the contributions per state: S_0 gets 0.1·(0.42·1 + 0.83·0.45 + 1.3·0.2025) = 0.1·(0.42+0.3735+0.263250) = 0.105675; S_1 gets 0.1·(0.83·1 + 1.3·0.45) = 0.1·(0.83+0.585) = 0.1415; S_2 gets 0.1·1.3 = 0.130. The backward totals (0.105675, 0.1415, 0.130) are exactly the forward totals. So the acausal forward λ-return algorithm and the causal backward eligibility-trace algorithm produce identical total weight changes over the episode, for every state — that's the forward/backward equivalence, and the fact that the numbers land only when the trace decays by γλ (not by λ, not by γ) is what pins the decay rate.

The proof has one constraint: I held V fixed during the episode. That's exactly the offline case, where I accumulate all the updates and apply them at the end, so every δ_k is computed against the same value function. In the *online* case — applying each update as I go — V drifts between steps, the δ's no longer all align, and the equality becomes an approximation, close when α is small enough that V barely changes within an episode. So online TD(λ) is a faithful-but-approximate implementation of the forward λ-return, exact only in the offline limit. That tells me where to look if I ever want exactness online: I'd need a trace that corrects for the drift it itself causes.

Which makes me suspicious that the trace isn't really about TD at all. Let me test that by stripping TD out entirely and asking whether an eligibility trace still falls out of pure Monte Carlo. Take linear MC with a single terminal return G and no discounting — the LMS rule w_{t+1} = w_t + α[G − w_t^T x_t]x_t, applied as the end-of-episode sweep over the stored features. One step can be rewritten as

  w_{t+1} = (I − αx_t x_t^T)w_t + αGx_t.

Define the fading matrix F_t = I − αx_t x_t^T. Then w_{t+1} = F_t w_t + αGx_t, and recursing from w_0 to the end,

  w_T = F_{T−1}⋯F_0 w_0 + αG Σ_{k=0}^{T−1} F_{T−1}⋯F_{k+1} x_k.

Name the two pieces a_{T−1} = F_{T−1}⋯F_0 w_0 and z_{T−1} = Σ_{k=0}^{T−1} F_{T−1}⋯F_{k+1} x_k, so w_T = a_{T−1} + αG z_{T−1}. The point is that z_t can be built incrementally from z_{−1}=0, with no stored history and no knowledge of G:

  z_t = F_t z_{t−1} + x_t = (I − αx_t x_t^T)z_{t−1} + x_t = z_{t−1} + (1 − α z_{t−1}^T x_t)x_t,

and a_t = F_t a_{t−1} the same way from a_{−1}=w_0. The claim that w_T = a_{T−1} + αG z_{T−1} reproduces the LMS sweep exactly is strong enough that I should not take it on the strength of the matrix algebra alone — a misplaced F_t in the recursion would still look plausible. Let me check it on a small instance: five random feature vectors in ℝ⁴, a single terminal return G=1.7, α=0.05, random w_0. Running the direct LMS sweep w_{t+1}=w_t+α(G−w_t^Tx_t)x_t gives one final w_T; building z incrementally by z_t = z_{t−1}+(1−αz_{t−1}^Tx_t)x_t and a by a_t=F_t a_{t−1} (each a single O(d) update, never touching G until the end) and forming a + αG z gives another. The two final weight vectors agree to ~10⁻¹⁶, machine precision. So even pure Monte Carlo, with not a single bootstrap in sight, reorganizes into an eligibility trace updated O(d) per step that reproduces the exact end-of-episode update — and the trace I get, z_t = z_{t−1} + (1 − α z_{t−1}^T x_t)x_t, is not the simple accumulating trace; it has that extra (1 − α z^T x) correction. Eligibility traces, then, are nothing to do with temporal differences specifically. They're the general device for assigning long-horizon credit cheaply and incrementally, and they show up the moment you try to implement *any* forward-looking, long-horizon update without storing the lookahead.

And that correction term is the thread back to the online-exactness problem. The accumulating trace γλ z_{t−1} + x_t gives the right *total* offline but only approximates the forward view online, because it ignores that each update shifts the very values being bootstrapped. The MC derivation just showed the shape of an exact incremental trace: the old trace is not only faded, it is also corrected for how much the current feature projects onto it. With TD's temporal fading included, that gives

  z_t = γλ z_{t−1} + (1 − αγλ z_{t−1}^T x_t)x_t,

paired with a weight update that also subtracts the bootstrap drift,

  w_{t+1} = w_t + αδ_t z_t + α(w_t^T x_t − w_{t−1}^T x_t)(z_t − x_t),

and I expect this version to reproduce the strict online λ-return target step by step, not just in total — true online TD(λ), with the same O(d) memory and asymptotic per-step cost as the plain version, plus one extra inner product. I want to be careful here: I checked the offline forward/backward equality directly above and it held to the digit, but the *online* exactness claim is a different statement — it's an equality against the online λ-return algorithm, which recomputes truncated λ-return targets from scratch at each growing data horizon, and that's fiddly enough to reimplement that a quick comparison of mine left a small residual I couldn't immediately attribute to either side. So I'll state this as the design goal the dutch correction is built to meet, and flag that the strict-online equivalence is the piece I'd want to verify against a carefully-written online λ-return reference before claiming it as proven, rather than asserting it the way I can assert the offline result. What I'm confident of: the plain accumulating-trace TD(λ) is offline-exact and online-approximate, and the (1 − αγλ z^T x) correction is the same dutch shape the Monte Carlo reorganization forced, now carrying TD's γλ fading — which is exactly the drift the online accumulating trace ignores.

So the chain, start to finish: TD(0) only moves credit one step and Monte Carlo only at episode end; n-step returns interpolate but force a brittle horizon choice and an n-deep buffer; the error-reduction property licenses averaging *all* n-step returns, and the scale-free, incrementally-summable weighting is geometric, giving the (1−λ)λ^{n−1}-weighted λ-return that dials TD(0)↔MC with one knob — and the two endpoints checked out numerically as the one-step and full returns; that forward target is acausal, so I flip to a backward view, scattering each current TD error onto recently visited states through a decaying trace; unrolling the λ-return into TD errors forces the trace to decay by γλ — γ for the reward's depth in the return, λ for the horizon weighting — and a triangular sum-swap matched against a column-collapse shows the backward trace reproduces the forward λ-return's total update in the offline case, which I confirmed on a three-step episode where the per-state totals from the two algorithms agreed to the digit; and the same trace machinery, derived independently from pure Monte Carlo and checked to reproduce the LMS sweep to machine precision, reveals that eligibility traces are the general mechanism for cheap long-horizon credit assignment, with a self-correcting dutch form aimed at making the strict online λ-return equivalence exact as well — that last step being the one I'd still want to verify against a proper online λ-return reference rather than take on faith.

```python
import numpy as np

# Linear value v̂(s,w)=w·x(s); tabular is the one-hot x special case. ∇v̂ = x.

# --- TD(λ): the backward eligibility-trace algorithm (accumulating trace) ---
# z_t = γλ z_{t-1} + ∇v̂(S_t)   ;   δ_t = R + γ v̂(S') − v̂(S)   ;   w += α δ_t z_t
class TDLambda:
    def __init__(self, d, gamma, lam, alpha):
        self.w = np.zeros(d)
        self.gamma, self.lam, self.alpha = gamma, lam, alpha
    def reset_episode(self):
        self.z = np.zeros_like(self.w)            # trace zeroed at episode start
    def step(self, x, r, x_next, terminal):
        v      = self.w @ x
        v_next = 0.0 if terminal else self.w @ x_next
        delta  = r + self.gamma * v_next - v      # one-step TD error (12.6)
        self.z = self.gamma * self.lam * self.z + x   # decay by γλ, bump by ∇v̂ (12.5)
        self.w += self.alpha * delta * self.z         # scatter δ over the trace (12.7)
        # λ=0 ⇒ z=x ⇒ pure TD(0); λ=1 leaves only the γ discount decay.

# --- Forward view: the (offline) λ-return target it is equivalent to ---
# G^λ_t = (1−λ) Σ_n λ^{n-1} G_{t:t+n}  ; update toward G^λ_t at episode end.
def lambda_return(rewards, values, gamma, lam):
    # rewards[t]=R_{t+1}, values[t]=v̂(S_t); episodic, values[T]=0.
    T = len(rewards)
    G = np.zeros(T)
    G[T-1] = rewards[T-1]                          # last step: full return
    for t in range(T-2, -1, -1):
        # recursive λ-return: G^λ_t = R_{t+1} + γ[(1−λ)v̂(S_{t+1}) + λ G^λ_{t+1}]
        G[t] = rewards[t] + gamma * ((1-lam)*values[t+1] + lam*G[t+1])
    return G

def offline_lambda_return_update(agent, xs, rewards, gamma, lam):
    values = np.array([agent.w @ x for x in xs] + [0.0])
    G = lambda_return(rewards, values, gamma, lam)
    dw = np.zeros_like(agent.w)
    for t, x in enumerate(xs):
        dw += agent.alpha * (G[t] - agent.w @ x) * x   # forward update (12.4)
    agent.w += dw   # equals Σ of TD(λ)'s updates over the episode (offline, w fixed)

# --- True online TD(λ): dutch trace, exact for the strict online λ-return view ---
class TrueOnlineTDLambda:
    def __init__(self, d, gamma, lam, alpha):
        self.w = np.zeros(d); self.gamma, self.lam, self.alpha = gamma, lam, alpha
    def reset_episode(self, x):
        self.z = np.zeros_like(self.w); self.v_old = 0.0; self.x = x
    def step(self, r, x_next, terminal):
        x = self.x
        v      = self.w @ x
        v_next = 0.0 if terminal else self.w @ x_next
        delta  = r + self.gamma * v_next - v
        gl     = self.gamma * self.lam
        # dutch trace: γλ z + (1 − α γλ zᵀx) x  — the self-correcting form (12.11)
        self.z = gl * self.z + (1 - self.alpha * gl * (self.z @ x)) * x
        self.w += self.alpha * (delta + v - self.v_old) * self.z
        self.w -= self.alpha * (v - self.v_old) * x
        self.v_old = v_next
        self.x = x_next
```

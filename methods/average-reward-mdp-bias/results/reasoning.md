OK, let me think about the situation I'm actually in. I have a system that never stops. A queue that keeps taking customers; a machine I maintain forever; a controller that cycles through states for as long as it runs. Finite states `S`, finite actions `A`, reward `r(s,a)`, transitions `p(s'|s,a)`. I want the best stationary policy. And the honest thing about my task is that there's no last period — the process just goes.

The tool everyone reaches for is the discounted return, `Σ_{t≥0} γ^t r_t` with `γ < 1`. I know exactly why `γ` is there: an undiscounted infinite sum of bounded rewards diverges, and `γ < 1` tames it into something finite, `V_π = (I − γP_π)^{-1} r_π`, a clean linear solve because `I − γP_π` is invertible — its eigenvalues are `1 − γλ_i` with `|λ_i| ≤ 1`, and since `γ < 1` none of them hit zero. So the whole discounted apparatus works: the Bellman optimality equation `V*(s) = max_a [r(s,a) + γ Σ p(s'|s,a) V*(s')]`, the fact that `T_γ` is a `γ`-contraction in max-norm so value iteration converges geometrically, greedy-on-`V*` is optimal. Beautiful, but the `γ` was never part of *my* problem. I introduced it to make a sum converge. It quietly imposes a horizon of about `1/(1−γ)` steps, and for a task that genuinely never ends that horizon is a fiction.

And it's not just inelegant — it can be wrong. Let me make myself uneasy with a concrete case. Suppose I'm at a state where I commit to one of two cycles. Cycle one pays `5` immediately and then every `5` steps; cycle two pays `10` four steps later and then every `5` steps. Per step that's `1` versus `2`. For a task that runs forever, cycle two is obviously the one I want — twice the long-run yield. Now compute discounted values: `5/(1−γ^5)` versus `10γ^4/(1−γ^5)`. The first wins whenever `5 > 10γ^4`, i.e. whenever `γ < 2^{-1/4} ≈ 0.8409`, purely because its smaller reward lands sooner and discounting overweights "sooner." So discounting picks the long-run-worse policy. I can fight back by shoving `γ` toward `1`... but then `V_γ(s) → ∞` for *every* policy. The ranking I care about becomes the difference of two quantities that both blow up. That's not a knob I want my answer to hinge on.

So let me just say what I actually want and refuse to apologize for it: maximize the long-run reward per step,

```
g = lim_{N→∞} (1/N) E[ Σ_{t=0}^{N-1} r_t ].
```

No `γ`. This is the honest objective for a continuing task. Now — can I build a Bellman-style theory for *this*?

First, what is `g` for a fixed policy `π`? The policy induces a Markov chain `P_π`, reward vector `r_π`. The chain has recurrent states (visited forever) and transient ones (eventually abandoned). Say it's unichain — a single recurrent class plus maybe some transient states; that's the common, clean case. Along a trajectory the time-average of the reward converges, by the ergodic theorem, to its expectation under the stationary distribution `d` (`dP_π = d`): `g = d · r_π`. And here's the first thing to really sit with: this `g` does **not** depend on the start state. From any state, after a finite transient stretch you fall into the one recurrent class and spend all of eternity there, and the eternity is what survives dividing by `N`. The finite reward of the transient prefix gets divided by `N → ∞` and vanishes. So `g` is a single scalar for the whole unichain policy.

That scalar is a problem. If the value of every state is the same number `g`, then `g` tells me *nothing* about which states are preferable, and a greedy improvement step `argmax_a [...]` needs exactly that — it needs to know whether action `a` lands me somewhere better. A constant carries no ranking. So a single function-of-state value, the thing the discounted theory hands me, has collapsed to a constant and lost all its discriminating power. I'm going to need something *besides* `g`.

Let me try to write a Bellman equation anyway and see what breaks, because the breakage will tell me what the missing object is. Naively set `γ = 1` in the policy-evaluation equation: `V = r_π + P_π V`, i.e. `(I − P_π) V = r_π`. Now `I − P_π` is **singular** — `P_π` is stochastic, `P_π 1 = 1`, so `1` is an eigenvalue and `I − P_π` kills the constant vector. Two consequences. One: a solution exists only if `r_π` is in the range of `(I − P_π)`, which a generic reward vector is not — so there's no `V` at all unless I modify the equation. Two: even where a solution exists it's not unique, because I can add any multiple of `1` (the null vector) and stay a solution. The singular matrix is screaming at me: the constant direction is degenerate, and the natural way the constant direction enters a never-ending process is... the average reward. The `1` in the null space and the state-independent `g` are the same phenomenon.

So let me put `g` back in *on purpose* and look for a relative description. I can't ask "what is the total reward from state `s`," that's infinite. But I *can* ask "how much better is starting in `s` than the steady state — what extra, finite reward do I accumulate because I happen to start here?" Concretely, run the process for `N` steps from `s`. Its expected reward should have a common leading term `N · g` (the steady stream), plus a state-dependent offset that captures the transient. Subtract off the part that's common to everyone — the `N · g` — and look at what's left:

```
h(s) := lim_{N→∞} [ E[Σ_{t=0}^{N-1} r_t | s_0=s] − N g ].
```

For an aperiodic unichain this is the same as the ordinary centered sum `E[Σ_{t=0}^{N-1}(r_t−g)|s_0=s]` converging. If the recurrent class is periodic, the centered partial sums can oscillate by phase, so I should not make ordinary convergence carry the theory. The robust object is the finite offset pinned by the first-step balance — equivalently the Abel/Laurent limit as `γ → 1`. That offset is the *relative value*, the *bias*: the cumulative advantage (or deficit) of starting in `s` rather than in the steady state. This is the object that survived when the absolute value died. It's a value function in the only sense that's left to me.

Now does `h` satisfy a recursion? Condition on the first step. Starting in `s`, I get `r(s,π(s))`, move to `s'` with prob `p(s'|s,π(s))`, and from `s'` onward I accumulate `h(s')` of relative value — but I've also "used up" one period of the steady stream, so I owe a `g`. Let me derive it cleanly by telescoping. Write the `N`-step expected reward from `s` as `f_N(s)`. We have `f_N(s) = r_π(s) + Σ_{s'} p(s'|s) f_{N-1}(s')`. Posit the asymptotic form `f_N(s) ≈ N·g + h(s)` (leading linear term `N·g` because the gain is state-independent, plus the finite offset `h`). Substitute:

```
N·g + h(s) ≈ r_π(s) + Σ_{s'} p(s'|s) [ (N−1)·g + h(s') ].
```

The right side is `r_π(s) + (N−1)·g · Σ_{s'} p(s'|s) + Σ_{s'} p(s'|s) h(s') = r_π(s) + (N−1)·g + Σ p h(s')` (the transition probabilities sum to `1`). Match the `O(N)` terms: `N·g` on the left, `(N−1)·g + ...` on the right — wait, those differ by `g`, which is the constant I move to the offset equation. Collecting:

```
N·g + h(s) = r_π(s) + (N−1)·g + Σ_{s'} p(s'|s) h(s').
```

Cancel `(N−1)·g` from both sides — the leading drift matches exactly because `g` is the same constant everywhere, which is *why* I needed it state-independent — and I'm left with the `O(1)` balance:

```
g + h(s) = r_π(s) + Σ_{s'} p(s'|s,π(s)) h(s').
```

There it is. This is the **average-reward policy-evaluation equation** — the Poisson equation for the Markov reward chain. In vector form `g·1 + (I − P_π) h = r_π`. And notice the bookkeeping closes perfectly with the singularity I hit earlier: `(I − P_π) h = r_π − g·1`, and now the right side is forced into the range of `(I − P_π)` precisely by the choice `g = d · r_π` (left-multiply by `d`: `d(I−P_π)h = 0` since `dP_π = d`, so I need `d(r_π − g·1) = 0`, i.e. `g = d·r_π` — consistent). The equation is solvable *because* of `g`, and the leftover non-uniqueness — `h` is determined only up to adding a constant, exactly the null space `span{1}` of `I − P_π` — is harmless: greedy choices depend only on differences `h(s') − h(s'')`, never on the absolute level. I pin it by fixing a reference, `h(reference) = 0`. So it's `|S|` equations in `|S|+1` unknowns (`h` and the scalar `g`), and the reference pin supplies the last constraint. Solve the linear system — done, I have `(g, h)` for the policy.

The pair. That's the resolution of "a scalar can't rank states": the long-run optimum is characterized not by one value function but by a **pair** — the gain `g`, a constant on each recurrent class telling me the *level*, and the bias `h`, the relative-value function telling me the *ranking*. `h` plays the role of value; `g` is the per-step toll.

Now optimization. I have evaluation; I want improvement. Given the current `(g, h)`, define the greedy policy `π'(s) = argmax_a [ r(s,a) + Σ_{s'} p(s'|s,a) h(s') ]`. The intuition: `h` measures relative desirability of next states, so picking the action that maximizes `immediate reward + expected next bias` should not lower the long-run gain, and a strict improvement somewhere should move me to a better stationary policy. Alternate evaluate/improve. This is policy iteration, transplanted from the discounted world, with the discounted linear solve `(I−γP)V = r` replaced by the Poisson solve `g·1 + (I−P)h = r`. Because there are finitely many stationary deterministic policies, Howard's improvement step terminates in finitely many steps at a policy I can't improve in gain. At the fixed point the greedy action equals the current action, which says

```
g* + h*(s) = max_a [ r(s,a) + Σ_{s'} p(s'|s,a) h*(s') ].
```

This is the **average-reward Bellman optimality equation**. A scalar `g*` and a vector `h*` together solve it, and the greedy policy with respect to `h*` achieves the optimal gain `g* = max_π g_π`. For a unichain (or more generally communicating) MDP this pair exists, and a stationary deterministic optimal policy exists. The structure mirrors the discounted optimality equation `V* = max_a[r + γ Σ p V*]`, but with two differences forced by the never-ending horizon: there are two unknowns instead of one, and the `max` is over `r + Σ p h*` offset by the constant `g*` rather than over a discounted bootstrap.

I want a second, independent derivation of this same pair, because I don't fully trust an asymptotic ansatz I pulled out of telescoping. Let me get `(g, h)` to *fall out* of the discounted theory in the limit `γ → 1`. The discounted value is `V_γ = (I − γP_π)^{-1} r_π`. The reason I can't just set `γ = 1` is that `(I − γP_π)^{-1}` has a **pole** at `γ = 1`: `I − P_π` is singular. A function with a pole has a Laurent expansion about that point, not a Taylor expansion. So write `V_γ` as a Laurent series in `(1 − γ)`:

```
V_γ(s) = y_{-1}(s)/(1−γ) + y_0(s) + y_1(s)·(1−γ) + …
```

a `1/(1−γ)` pole term, a constant term, vanishing corrections. Plug this into the discounted Bellman equation `V_γ = r_π + γ P_π V_γ` and match powers of `(1−γ)`. Write `γ = 1 − ε` with `ε = 1 − γ → 0`:

```
y_{-1}/ε + y_0 + y_1 ε + … = r_π + (1−ε) P_π [ y_{-1}/ε + y_0 + y_1 ε + … ].
```

Expand the right side: `r_π + P_π y_{-1}/ε + P_π y_0 + P_π y_1 ε − P_π y_{-1} − P_π y_0 ε − … `. Collect by order of `ε`.

Order `ε^{-1}` (the pole): `y_{-1} = P_π y_{-1}`. So `y_{-1}` is a right eigenvector of `P_π` with eigenvalue `1` — for a unichain chain that's the constant vector, `y_{-1} = g·1` for some scalar `g`. The pole coefficient is a state-independent constant. That **is** the gain: the discounted value blows up like `g/(1−γ)`, and the rate of blow-up is the average reward. (Sanity: `V_γ ≈ g/(1−γ) → g · Σ_{t≥0} γ^t`, the discounted value of getting `g` every step — exactly the steady stream. Good.)

Order `ε^0` (the constant term): from the left, `y_0`. From the right, `r_π + P_π y_0 − P_π y_{-1}`. So `y_0 = r_π + P_π y_0 − P_π y_{-1}`. Substitute `y_{-1} = g·1`, and `P_π (g·1) = g·1`:

```
y_0 = r_π + P_π y_0 − g·1   ⇒   g·1 + (I − P_π) y_0 = r_π.
```

That is *exactly* the Poisson equation I derived by telescoping, with `y_0` in the role of `h`. So the constant term of the Laurent expansion **is** the bias. The limit `γ → 1` doesn't destroy information — it sorts the discounted value into a divergent gain piece and a finite bias piece:

```
V_γ(s) = g/(1−γ) + h(s) + e_γ(s),   with e_γ(s) → 0 as γ → 1.
```

The `O(1)` constant term that survives the limit is forced to be `h`, and the higher-order remainder `e_γ = y_1·(1−γ) + …` vanishes. The pair `(g, h)` isn't an ad-hoc construction; it's literally the leading two coefficients of the discounted value's singular expansion. The discounting that I distrusted for being arbitrary turns out to be the right *analytic probe*: differentiate the discounted world at its singular point and the average-reward objects drop out as residues.

Let me re-check the order bookkeeping once more, because a dropped factor here would be fatal. The pole term contributes `g/(1−γ)` to the value — multiplying `1/(1−γ) → ∞`, so for large `γ`, `g/(1−γ) ≫ h + e_γ`; the value is dominated by the scaled gain. That's the same `V_γ → ∞` I worried about earlier, now decomposed. Two policies with different gains: the `1/(1−γ)` terms differ and, as `γ → 1`, the gain comparison swamps everything — the larger-gain policy eventually wins. Two policies with *equal* gain: the `1/(1−γ)` terms cancel and the comparison drops to the *next* term, the bias `h`. The factors line up.

And that cancellation is the clue to a problem I haven't faced yet. Maximal gain is sometimes *too coarse*. Picture a task where several policies all reach an absorbing goal and then sit there earning `0`: every one of them has long-run average reward `0`, all tied on gain. Gain can't separate them. But I obviously prefer the policy that reaches the goal *fastest*, hoarding the most reward on the way. Where does "fastest" live? In the **bias**. The finite advantage of the transient prefix — the reward you scoop up before settling into the recurrent class — is exactly what `h` measures. So among gain-optimal policies, I should break the tie by choosing the one with the largest `h(s)` at every state. Pictorially this is the `1/(1−γ)` terms canceling and the decision falling to the `O(1)` term: when gains tie, `lim_{γ→1} (V_γ^{π*} − V_γ^π) = h^{π*} − h^π`, so maximizing discounted value in the limit *is* maximizing bias. That refinement — gain-optimal **and** then bias-optimal — is the right notion of best for a continuing task. Howard's policy iteration, which optimizes gain first, lands on a bias-optimal policy when the first gain-optimal policy it finds already has the right recurrent structure; in general bias-optimality is the genuinely sharper criterion.

I can make this ordering systematic, and the Laurent expansion hands me the ladder. Define: `π*` is **`n`-discount-optimal** if `lim_{γ→1} (1−γ)^{-n} (V_γ^{π*}(s) − V_γ^π(s)) ≥ 0` for every state `s` and every competitor `π`. Read off what each `n` means by plugging in the expansion `V_γ = g/(1−γ) + h + e_γ`.

`n = −1`: multiply the difference by `(1−γ)`. The `g/(1−γ)` terms become `g`, the `h` and `e_γ` terms become `O(1−γ) → 0`. So the condition is `g^{π*} − g^π ≥ 0` for all `π` — **`(−1)`-discount-optimal is exactly gain-optimal.** (Let me write the proof out: `(1−γ)(V_γ^{π*} − V_γ^π) = (g^{π*} − g^π) + (1−γ)(h^{π*} − h^π) + (1−γ)(e^{π*} − e^π)`. As `γ → 1` the last two groups vanish since `h` is finite and `e → 0`, leaving `g^{π*} − g^π ≥ 0`.)

`n = 0`: don't multiply at all. Now I must already be restricting to gain-optimal policies — otherwise the un-multiplied difference contains the `(g^{π*} − g^π)/(1−γ)` pole, which dominates and forces `g^{π*} ≥ g^π` anyway. Among gain-equal policies the pole cancels and `lim_{γ→1} (V_γ^{π*} − V_γ^π) = (h^{π*} − h^π) + lim(e^{π*} − e^π) = h^{π*} − h^π ≥ 0`. So **`0`-discount-optimal is gain-optimal *plus* bias-optimal.** The two criteria I motivated by hand are the `n = −1` and `n = 0` rungs of one ladder.

Higher `n` keeps going: each rung peels off the next Laurent coefficient `y_n` as the tie-breaker once all lower coefficients are tied. The whole sequence is nested — `m`-discount-optimal implies `n`-discount-optimal for every `n < m` — so the optimal sets shrink as `n` grows: all policies ⊃ gain-optimal ⊃ bias-optimal ⊃ … The limit of this nesting is the sharpest criterion of all: a policy that is `n`-discount-optimal for *every* `n ≥ −1`. Equivalently, a single policy that maximizes the discounted value for *all* `γ` sufficiently close to `1` — one policy that dominates on a whole interval `(γ*, 1)`, so there's no further tie left to break. That's **Blackwell optimality**, and Blackwell's result is that for a finite MDP such a policy exists: there is a `γ*` below `1` such that one stationary policy is discount-optimal for every `γ ∈ (γ*, 1)`, and it is exactly the policy that maximizes the Laurent coefficients lexicographically — gain first, then bias, then the rest. The discounting I started out distrusting is, in the end, what *defines* the finest optimality: not any single `γ`, but the behaviour as `γ → 1^-`.

Now the computation, because I want to actually solve these, not just characterize them. Policy iteration I already have: evaluate `(g, h)` by the Poisson solve with a reference pin, improve greedily, repeat — finitely terminating. But each evaluation is an `|S|`-dimensional linear solve, expensive when `|S|` is large. Can I iterate the optimality equation directly, the way value iteration iterates the discounted one? Define the average-reward backup `T(V)(s) = max_a [ r(s,a) + Σ_{s'} p(s'|s,a) V(s') ]` and try `V_{k+1} = T(V_k)`. Watch what happens: at the optimum `g* + h* = T(h*)`, so `T` shifts its fixed point *up by `g*` every application* — `T(h*) = h* + g*·1`. The iterates don't converge; they march off to infinity at rate `g*` per step, `V_k ≈ k·g* + h*`. That's the same divergence as the undiscounted sum, reappearing inside the iteration. The absolute values run away.

But the *relative* values — the differences `V_k(s) − V_k(s')` — are exactly what's converging, because the runaway part `k·g*` is a common constant added to every state and cancels in any difference. So I should iterate only the relative values. Subtract off a reference state at each step:

```
V_{k+1}(s) = T(V_k)(s) − T(V_k)(reference).
```

Now `V_{k+1}(reference) = 0` for all `k`, the iterates can't drift, and they converge to `h` (up to the reference offset). I need to keep one raw quantity before the subtraction: if `U_k = T(V_k)`, then the drift vector `U_k − V_k` approaches `g*·1`, and the offset `U_k(reference) − V_k(reference)` approaches `g*` when the reference is pinned to zero. This is **relative value iteration**. And the stopping rule should ignore the common drift too: not the max-norm of the raw increment, which is stuck near `g*`, but the **span seminorm** `sp(V) = max_s V(s) − min_s V(s)`. Stop when `sp(U_k − V_k) < ε`: the span measures the spread of the *relative* update and is blind to any constant added to all states, which is precisely the part I don't care about and can't stop drifting.

Two cautions I should bake in. First, periodicity. If the chain has a recurrent class of period `> 1`, the backup `T` doesn't contract in span and the iterates can oscillate forever instead of settling — value iteration needs the chain aperiodic under all policies (or a state reachable from everywhere). The fix is an aperiodicity transformation: blend in a self-loop, `P̃ = (1−τ)I + τP` with `0 < τ < 1`, and scale the rewards to `r̃ = τr`. Then `T̃(h) = h + τg`, the greedy action is unchanged because the transformed maximand is `(1−τ)h(s) + τ[r(s,a)+P_a h]`, and the transformed gain is `g̃ = τg`; the original gain is the measured drift divided by `τ`. Second, asynchrony. If I update states one at a time (Gauss–Seidel / asynchronous), relative value iteration can actually **diverge** — Tsitsiklis's two-state example shows the relative-value map is not monotone and not a max-norm contraction, so the convergence guarantees of asynchronous *discounted* value iteration do not carry over. That's a real trap: the synchronous sweep is safe; the obvious asynchronous shortcut is not, without further care.

Let me collect what the reasoning has produced, because the pieces now lock together. The pain was a continuing task with no horizon, where discounting injects an arbitrary `γ` that can rank policies wrong and blows the value up as `γ → 1`. The honest objective is the long-run average `g`. For a unichain policy `g` is a state-independent scalar, which carries no ranking — so a single value function is not enough. Trying `γ = 1` directly hits the singularity of `I − P`, whose null space `span{1}` is the same degeneracy as the state-independent gain. Reintroducing `g` as a per-step toll and asking for the finite relative advantage of each start state produces the bias `h` and the Poisson equation `g + h(s) = r(s) + Σ p h(s')`; the same equation drops out independently as the `O(1)` coefficient of the Laurent expansion `V_γ = g/(1−γ) + h + O(1−γ)`, whose `O(1/(1−γ))` coefficient is the gain. Taking the `max` over actions gives the average-reward Bellman optimality equation `g* + h*(s) = max_a [r(s,a) + Σ p(s'|s,a) h*(s')]`, characterizing the optimum by the **pair** `(g*, h*)`. Gain alone is coarse; the Laurent ladder refines it — `(−1)`-discount-optimal = gain-optimal, `0`-discount-optimal = also bias-optimal, and `∞`-discount-optimal = Blackwell-optimal, one policy that wins for all `γ` near `1`. Policy iteration (Poisson-solve + greedy) finds a gain-optimal policy in finitely many steps; the Laurent/Blackwell refinement says how to break equal-gain ties by bias and higher coefficients; relative value iteration with reference subtraction and a span-seminorm stop iterates to `h` and recovers `g*` from the raw drift. That's the landing — the optimality equations plus the gain/bias characterization, with policy iteration and RVI as the concrete solvers.

```python
import numpy as np

# ---- evaluate a fixed policy: the (gain, bias) PAIR via the Poisson equation ----
# g + h(s) = r(s,pi(s)) + sum_s' p(s'|s,pi(s)) h(s'),  pin h(reference)=0.
# This is the average-reward analogue of the discounted linear solve (I - gamma P)V = r,
# but with the gain g made explicit because (I - P) is singular.
def evaluate_policy(mdp, pi, ref=0):
    n = mdp.n
    Ppi = np.stack([mdp.P[pi[s]][s] for s in range(n)])   # (n,n) transition under pi
    rpi = np.array([mdp.R[pi[s]][s] for s in range(n)])    # length-n reward under pi
    # unknowns x = [h(0..n-1), g]; equations: h - Ppi h + g*1 = rpi, and h(ref)=0.
    A = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)
    A[:n, :n] = np.eye(n) - Ppi          # (I - Ppi) h
    A[:n, n]  = 1.0                       # + g
    b[:n]     = rpi
    A[n, ref] = 1.0                       # h(ref) = 0 pins the +c1 freedom
    x = np.linalg.solve(A, b)
    h, g = x[:n], x[n]
    return g, h

# ---- policy iteration: Poisson evaluation plus greedy improvement ----
def policy_iteration(mdp):
    pi = np.zeros(mdp.n, dtype=int)
    while True:
        g, h = evaluate_policy(mdp, pi)
        # greedy on r + E[h]: the average-reward improvement step
        Q = np.stack([mdp.R[a] + mdp.P[a] @ h for a in range(mdp.m)], axis=0)  # (m,n)
        pi_new = Q.argmax(axis=0)
        if np.array_equal(pi_new, pi):
            return g, h, pi                # fixed point: g + h = max_a [ r + P h ]
        pi = pi_new

# ---- relative value iteration: iterate h, subtract a reference to kill the g-drift ----
def relative_value_iteration(mdp, ref=0, eps=1e-9, tau=0.5):
    # aperiodicity transform: P~ = (1-tau)I + tau P, R~ = tau R, so g~ = tau*g
    P = [ (1 - tau) * np.eye(mdp.n) + tau * mdp.P[a] for a in range(mdp.m) ]
    R = [ tau * mdp.R[a] for a in range(mdp.m) ]            # rewards scale with the blend
    V = np.zeros(mdp.n)
    while True:
        Q = np.stack([R[a] + P[a] @ V for a in range(mdp.m)], axis=0)
        raw = Q.max(axis=0)
        drift = raw - V                                    # tends to tau*g times the all-ones vector
        V_next = raw - raw[ref]                            # reference subtraction: V_next[ref]=0
        if drift.max() - drift.min() < eps:                # span seminorm: ignores common drift
            g = 0.5 * (drift.max() + drift.min()) / tau    # undo g~ = tau*g
            pi = Q.argmax(axis=0)
            return g, V_next, pi
        V = V_next
```

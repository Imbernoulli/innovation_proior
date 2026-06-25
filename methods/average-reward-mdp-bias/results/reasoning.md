OK, let me think about the situation I'm actually in. I have a system that never stops. A queue that keeps taking customers; a machine I maintain forever; a controller that cycles through states for as long as it runs. Finite states `S`, finite actions `A`, reward `r(s,a)`, transitions `p(s'|s,a)`. I want the best stationary policy. And the honest thing about my task is that there's no last period — the process just goes.

The tool everyone reaches for is the discounted return, `Σ_{t≥0} γ^t r_t` with `γ < 1`. I know exactly why `γ` is there: an undiscounted infinite sum of bounded rewards diverges, and `γ < 1` tames it into something finite, `V_π = (I − γP_π)^{-1} r_π`, a clean linear solve because `I − γP_π` is invertible — its eigenvalues are `1 − γλ_i` with `|λ_i| ≤ 1`, and since `γ < 1` none of them hit zero. So the whole discounted apparatus works: the Bellman optimality equation `V*(s) = max_a [r(s,a) + γ Σ p(s'|s,a) V*(s')]`, the fact that `T_γ` is a `γ`-contraction in max-norm so value iteration converges geometrically, greedy-on-`V*` is optimal. Beautiful, but the `γ` was never part of *my* problem. I introduced it to make a sum converge. It quietly imposes a horizon of about `1/(1−γ)` steps, and for a task that genuinely never ends that horizon is a fiction.

And it's not just inelegant — it can be wrong. Let me make myself uneasy with a concrete case. Suppose I'm at a state where I commit to one of two cycles. Cycle one pays `5` immediately and then every `5` steps; cycle two pays `10` four steps later and then every `5` steps. Per step that's `1` versus `2`. For a task that runs forever, cycle two is obviously the one I want — twice the long-run yield. Now compute discounted values: `5/(1−γ^5)` versus `10γ^4/(1−γ^5)`. The first wins whenever `5 > 10γ^4`, i.e. whenever `γ < 2^{-1/4}`. Let me actually evaluate that threshold: `2^{-1/4} = 1/2^{0.25}`, and `2^{0.25} ≈ 1.1892`, so the crossover is at `γ ≈ 0.8409`. For any discount below that — and `0.84` is squarely inside the range people actually use — discounting prefers cycle one, purely because its smaller reward lands sooner and discounting overweights "sooner." So discounting picks the long-run-worse policy. I can fight back by shoving `γ` toward `1`... but then `V_γ(s) → ∞` for *every* policy. The ranking I care about becomes the difference of two quantities that both blow up. That's not a knob I want my answer to hinge on.

So let me just say what I actually want and refuse to apologize for it: maximize the long-run reward per step,

```
g = lim_{N→∞} (1/N) E[ Σ_{t=0}^{N-1} r_t ].
```

No `γ`. This is the honest objective for a continuing task. Now — can I build a Bellman-style theory for *this*?

First, what is `g` for a fixed policy `π`? The policy induces a Markov chain `P_π`, reward vector `r_π`. The chain has recurrent states (visited forever) and transient ones (eventually abandoned). Say it's unichain — a single recurrent class plus maybe some transient states; that's the common, clean case. Along a trajectory the time-average of the reward converges, by the ergodic theorem, to its expectation under the stationary distribution `d` (`dP_π = d`): `g = d · r_π`. And here's the first thing to really sit with: this `g` does **not** depend on the start state. From any state, after a finite transient stretch you fall into the one recurrent class and spend all of eternity there, and the eternity is what survives dividing by `N`. The finite reward of the transient prefix gets divided by `N → ∞` and vanishes. So `g` is a single scalar for the whole unichain policy.

That scalar is a problem. If the value of every state is the same number `g`, then `g` tells me *nothing* about which states are preferable, and a greedy improvement step `argmax_a [...]` needs exactly that — it needs to know whether action `a` lands me somewhere better. A constant carries no ranking. So a single function-of-state value, the thing the discounted theory hands me, has collapsed to a constant and lost all its discriminating power. I'm going to need something *besides* `g`.

Let me try to write a Bellman equation anyway and see what breaks, because the breakage will tell me what the missing object is. Naively set `γ = 1` in the policy-evaluation equation: `V = r_π + P_π V`, i.e. `(I − P_π) V = r_π`. Now `I − P_π` is **singular** — `P_π` is stochastic, `P_π 1 = 1`, so `1` is an eigenvalue and `I − P_π` kills the constant vector. Two consequences. One: a solution exists only if `r_π` is in the range of `(I − P_π)`, which a generic reward vector is not — so there's no `V` at all unless I modify the equation. Two: even where a solution exists it's not unique, because I can add any multiple of `1` (the null vector) and stay a solution. The singular matrix is telling me something: the constant direction is degenerate, and the natural way a constant direction enters a never-ending process is through a per-step rate. The `1` in the null space and the state-independent `g` look like the same phenomenon wearing two hats — let me see if I can make that precise rather than just assert it.

So let me put `g` back in *on purpose* and look for a relative description. I can't ask "what is the total reward from state `s`," that's infinite. But I *can* ask "how much better is starting in `s` than the steady state — what extra, finite reward do I accumulate because I happen to start here?" Concretely, run the process for `N` steps from `s`. Its expected reward should have a common leading term `N · g` (the steady stream), plus a state-dependent offset that captures the transient. Subtract off the part that's common to everyone — the `N · g` — and look at what's left:

```
h(s) := lim_{N→∞} [ E[Σ_{t=0}^{N-1} r_t | s_0=s] − N g ].
```

For an aperiodic unichain this is the same as the ordinary centered sum `E[Σ_{t=0}^{N-1}(r_t−g)|s_0=s]` converging. If the recurrent class is periodic, the centered partial sums can oscillate by phase, so I should not make ordinary convergence carry the theory. The robust object is the finite offset pinned by the first-step balance — equivalently the Abel/Laurent limit as `γ → 1`. That offset is the *relative value*, the *bias*: the cumulative advantage (or deficit) of starting in `s` rather than in the steady state. This is the object that survived when the absolute value died. It's a value function in the only sense that's left to me.

Now does `h` satisfy a recursion? Condition on the first step. Starting in `s`, I get `r(s,π(s))`, move to `s'` with prob `p(s'|s,π(s))`, and from `s'` onward I accumulate `h(s')` of relative value — but I've also "used up" one period of the steady stream, so I owe a `g`. Let me derive it cleanly by telescoping. Write the `N`-step expected reward from `s` as `f_N(s)`. We have `f_N(s) = r_π(s) + Σ_{s'} p(s'|s) f_{N-1}(s')`. Posit the asymptotic form `f_N(s) ≈ N·g + h(s)` (leading linear term `N·g` because the gain is state-independent, plus the finite offset `h`). Substitute:

```
N·g + h(s) ≈ r_π(s) + Σ_{s'} p(s'|s) [ (N−1)·g + h(s') ].
```

The right side is `r_π(s) + (N−1)·g · Σ_{s'} p(s'|s) + Σ_{s'} p(s'|s) h(s') = r_π(s) + (N−1)·g + Σ p h(s')` (the transition probabilities sum to `1`). So

```
N·g + h(s) = r_π(s) + (N−1)·g + Σ_{s'} p(s'|s) h(s').
```

Cancel `(N−1)·g` from both sides — the leading drift matches exactly because `g` is the same constant everywhere, which is *why* I needed it state-independent; if `g` had depended on `s'` it would not have pulled out of the sum and this cancellation would have failed. I'm left with the `O(1)` balance:

```
g + h(s) = r_π(s) + Σ_{s'} p(s'|s,π(s)) h(s').
```

So the policy-evaluation equation for the average criterion is a *pair* of objects — the scalar `g` and the vector `h` — tied together by one linear relation. In vector form `g·1 + (I − P_π) h = r_π`. Now let me check whether this actually resolves the singularity I hit a moment ago, because that's the thing that worried me. Rearrange: `(I − P_π) h = r_π − g·1`. The right side now has to land in the range of `(I − P_π)`. A solution to `Ax = b` with `A` singular exists exactly when `b` is orthogonal to the left null space of `A`. The left null vector of `I − P_π` is `d` (since `d(I − P_π) = d − dP_π = 0`). So solvability requires `d · (r_π − g·1) = 0`, i.e. `d·r_π − g·(d·1) = 0`, i.e. `g = d·r_π` since `d·1 = 1`. That is *precisely* the gain I already computed from the ergodic theorem. So the two derivations agree: the value of `g` that the steady-state distribution forces is exactly the value that makes the Poisson equation solvable. The equation is solvable *because* of `g`, not by accident. And the leftover non-uniqueness — `h` is determined only up to adding a constant, exactly the null space `span{1}` of `I − P_π` — is harmless: greedy choices depend only on differences `h(s') − h(s'')`, never on the absolute level. I pin it by fixing a reference, `h(reference) = 0`. So it's `|S|` equations in `|S|+1` unknowns (`h` and the scalar `g`), and the reference pin supplies the last constraint. Solve the linear system and I have `(g, h)` for the policy.

The pair. That's the resolution of "a scalar can't rank states": the long-run optimum is characterized not by one value function but by a **pair** — the gain `g`, a constant on each recurrent class telling me the *level*, and the bias `h`, the relative-value function telling me the *ranking*. `h` plays the role of value; `g` is the per-step toll.

Now optimization. I have evaluation; I want improvement. Given the current `(g, h)`, define the greedy policy `π'(s) = argmax_a [ r(s,a) + Σ_{s'} p(s'|s,a) h(s') ]`. The intuition: `h` measures relative desirability of next states, so picking the action that maximizes `immediate reward + expected next bias` should not lower the long-run gain, and a strict improvement somewhere should move me to a better stationary policy. Alternate evaluate/improve. This is policy iteration, transplanted from the discounted world, with the discounted linear solve `(I−γP)V = r` replaced by the Poisson solve `g·1 + (I−P)h = r`. Because there are finitely many stationary deterministic policies, the improvement step should terminate at a policy I can't improve in gain. At such a fixed point the greedy action equals the current action, which says

```
g* + h*(s) = max_a [ r(s,a) + Σ_{s'} p(s'|s,a) h*(s') ].
```

If this is right, it's the **average-reward Bellman optimality equation** — a scalar `g*` and a vector `h*` together solving it, with the greedy policy achieving `g* = max_π g_π`. The structure mirrors the discounted optimality equation `V* = max_a[r + γ Σ p V*]`, but with two differences forced by the never-ending horizon: there are two unknowns instead of one, and the `max` is over `r + Σ p h*` offset by the constant `g*` rather than over a discounted bootstrap. I've been pushing symbols around, though, and I want to *see* this work before I trust it — a hand example where I can read off the gain, the bias, and watch policy iteration land on the equation.

Let me build the smallest MDP that still has a real choice in it. Two states; call them "cheap" (state 0) and "rich" (state 1). In each state I can play `a=0` (drift toward cheap) or `a=1` (drift toward rich). To keep *every* policy unichain — single recurrent class — I make every move leaky: probability `0.9` in the intended direction, `0.1` the other way, so no state is ever absorbing. Rewards: being in cheap pays `1`, being in rich pays `3` (I'll attach the reward to the action that *sits* in a state). Concretely

```
P[a=0] = [[.9 .1],[.9 .1]]    R[a=0] = [1, 0]   # action 0: drift to state 0
P[a=1] = [[.1 .9],[.1 .9]]    R[a=1] = [0, 3]   # action 1: drift to state 1
```

Four deterministic policies. Let me evaluate each with the Poisson solve (pin `h(0)=0`) and cross-check the gain against `g = d·r_π` from the stationary distribution. Running it:

```
pi=(0,0):  g=0.90  h=[0,-1]   d=[.9 .1]     # always cheap-ward; sits mostly in state 0, pays ~1
pi=(0,1):  g=2.00  h=[0,10]   d=[.5 .5]      # cheap stays cheap, rich stays rich -> 50/50 -> (1+3)/2
pi=(1,0):  g=0.00  h=[0, 0]   d=[.5 .5]      # the "wrong" cross policy
pi=(1,1):  g=2.70  h=[0, 3]   d=[.1 .9]     # always rich-ward; sits mostly in state 1, pays ~2.7
```

Let me sanity-check a couple of these by hand instead of trusting the solver blindly. For `pi=(1,1)` the chain is `[[.1 .9],[.1 .9]]`; its stationary distribution solves `d = dP`, and a chain whose rows are identical `[.1 .9]` lands in `[.1 .9]` immediately, so `g = .1·0 + .9·3 = 2.7`. Matches. For `pi=(0,1)` the chain is `[[.9 .1],[.1 .9]]`, symmetric, so `d=[.5 .5]` and `g = .5·1 + .5·3 = 2.0`. Matches. Good — the solver's gains are right, and the best policy is `(1,1)` with gain `2.7`, which is what I'd expect: spend as much time as possible in the rich state.

Now run policy iteration from `pi=(0,0)` and watch it. Evaluate gives `g=0.9, h=[0,-1]`. Greedy step: `Q[a,s] = R[a] + (P[a] h)[s]`. `P[0] h = [.9·0+.1·(−1), .9·0+.1·(−1)] = [−.1,−.1]`, so `Q[0] = [1,0]+[−.1,−.1] = [.9,−.1]`. `P[1] h = [.1·0+.9·(−1), …] = [−.9,−.9]`, so `Q[1] = [0,3]+[−.9,−.9] = [−.9,2.1]`. Argmax over actions per state: state 0 prefers `a=0` (`.9 > −.9`), state 1 prefers `a=1` (`2.1 > −.1`). So `pi → (0,1)`. Evaluate `(0,1)`: `g=2.0, h=[0,10]`. Greedy again with this `h`: `P[1] h = [.1·0+.9·10, …] = [9,9]`, `Q[1]=[0,3]+[9,9]=[9,12]`; `P[0] h = [.9·0+.1·10,…]=[1,1]`, `Q[0]=[1,0]+[1,1]=[2,1]`. State 0: `9 > 2` → `a=1`; state 1: `12 > 1` → `a=1`. So `pi → (1,1)`. Evaluate `(1,1)`: `g=2.7, h=[0,3]`. Greedy once more: `P[1] h=[.1·0+.9·3,…]=[2.7,2.7]`, `Q[1]=[0,3]+[2.7,2.7]=[2.7,5.7]`; `P[0] h=[.1·3,.1·3]=[.3,.3]`... wait, `P[0] h = [.9·0+.1·3, .9·0+.1·3]=[.3,.3]`, `Q[0]=[1,0]+[.3,.3]=[1.3,.3]`. State 0: `2.7 > 1.3` → `a=1`; state 1: `5.7 > .3` → `a=1`. Greedy returns `(1,1)` — unchanged. Fixed point reached in three improvement steps. And at that fixed point, check the optimality equation directly: `g* + h*(s)` is `2.7+0 = 2.7` and `2.7+3 = 5.7`; `max_a Q[a,s]` is `2.7` and `5.7`. They match exactly, residual zero in both states. So the fixed point really does satisfy `g* + h* = max_a[r + P h*]`. The Bellman optimality equation isn't just plausible from the telescoping — I watched policy iteration converge to a `(g*, h*)` that solves it.

That worked, but it worked on a deliberately leaky MDP. Let me push on the unichain assumption to find out what it's actually buying me, because I leaned on it without testing it. Drop the leak: `a=0` makes state 0 self-loop with certainty (reward 1), `a=1` makes state 1 self-loop with certainty (reward 3), and the off-actions move you across. Now the policy `pi=(0,1)` makes *both* states absorbing — `P_π = I`, two separate recurrent classes, a **multichain**. Try to evaluate it with the same Poisson solve: `(I − P_π) = (I − I) = 0`, so the linear system's top block is all zeros except the `+g` column, and with the reference pin the matrix is

```
A = [[0 0 1],[0 0 1],[1 0 0]],  det(A) = 0.
```

Singular — the solve throws. And it *should*: a multichain policy doesn't have one gain, it has a different gain in each recurrent class (here `1` in class `{0}`, `3` in class `{1}`), so a single scalar `g` cannot describe it. The state-independence of `g` that I leaned on at the very start is exactly the unichain assumption, and here is where it fails concretely. This isn't a bug to patch away cheaply — it's the genuine boundary of the gain-plus-scalar theory. The honest scope is unichain (or communicating, where every policy's chain can be steered into a single class); the leak in my first example was what kept every policy inside that scope, which is why everything closed there. I'll keep the unichain assumption explicit rather than pretend the scalar `g` works in general.

I want a second, independent derivation of the same pair, because I don't fully trust an asymptotic ansatz I pulled out of telescoping, and the hand example only confirms one MDP. Let me get `(g, h)` to *fall out* of the discounted theory in the limit `γ → 1`. The discounted value is `V_γ = (I − γP_π)^{-1} r_π`. The reason I can't just set `γ = 1` is that `(I − γP_π)^{-1}` has a **pole** at `γ = 1`: `I − P_π` is singular. A function with a pole has a Laurent expansion about that point, not a Taylor expansion. So write `V_γ` as a Laurent series in `(1 − γ)`:

```
V_γ(s) = y_{-1}(s)/(1−γ) + y_0(s) + y_1(s)·(1−γ) + …
```

a `1/(1−γ)` pole term, a constant term, vanishing corrections. Plug this into the discounted Bellman equation `V_γ = r_π + γ P_π V_γ` and match powers of `(1−γ)`. Write `γ = 1 − ε` with `ε = 1 − γ → 0`:

```
y_{-1}/ε + y_0 + y_1 ε + … = r_π + (1−ε) P_π [ y_{-1}/ε + y_0 + y_1 ε + … ].
```

Expand the right side: `r_π + P_π y_{-1}/ε + P_π y_0 + P_π y_1 ε − P_π y_{-1} − P_π y_0 ε − … `. Collect by order of `ε`.

Order `ε^{-1}` (the pole): `y_{-1} = P_π y_{-1}`. So `y_{-1}` is a right eigenvector of `P_π` with eigenvalue `1` — for a unichain chain that's the constant vector, `y_{-1} = g·1` for some scalar `g`. The pole coefficient is a state-independent constant. So the discounted value blows up like `g/(1−γ)`. Is that `g` the average reward? Quick check: if `V_γ ≈ g/(1−γ)`, that's `g · Σ_{t≥0} γ^t`, the discounted value of receiving `g` every single step forever — exactly the steady stream of a process whose per-step rate is `g`. So the pole coefficient *is* the gain, consistently with the first derivation.

Order `ε^0` (the constant term): from the left, `y_0`. From the right, `r_π + P_π y_0 − P_π y_{-1}`. So `y_0 = r_π + P_π y_0 − P_π y_{-1}`. Substitute `y_{-1} = g·1`, and `P_π (g·1) = g·1`:

```
y_0 = r_π + P_π y_0 − g·1   ⇒   g·1 + (I − P_π) y_0 = r_π.
```

That is the same Poisson equation I derived by telescoping, with `y_0` in the role of `h`. So the constant term of the Laurent expansion plays the part of the bias. The limit `γ → 1` doesn't destroy information — it sorts the discounted value into a divergent gain piece and a finite bias piece:

```
V_γ(s) = g/(1−γ) + h(s) + e_γ(s),   with e_γ(s) → 0 as γ → 1.
```

Two derivations now point at the same `(g, h)` — one from telescoping the undiscounted partial sums, one from the singular structure of the discounted resolvent. But both are still symbol-pushing, and I'd rather watch the decomposition happen numerically than trust the algebra twice. Take the leaky MDP again, policy `(1,1)`, where I already know `g=2.7` and `h=[0,3]`. Compute the genuine discounted value `V_γ = (I−γP)^{-1}r` at a sequence of `γ` creeping toward `1`, and read off the two coefficients.

First the pole. `(1−γ)·V_γ(s)` should approach `g = 2.7`:

```
γ        (1−γ)V_γ(0)   (1−γ)V_γ(1)
0.9        2.4300        2.7300
0.99       2.6730        2.7030
0.999      2.6973        2.7003
0.9999     2.69973       2.70003
```

Both states march to `2.7` as `γ → 1`. The pole coefficient is the gain, numerically, to as many digits as I care to take `γ` close to `1`. Now the constant term. Subtract the known pole off and see what's left, `V_γ(s) − g/(1−γ)`:

```
γ        V_γ(0)−g/(1−γ)   V_γ(1)−g/(1−γ)    difference
0.9         −2.700000        0.300000        3.000000
0.99        −2.700000        0.300000        3.000000
0.999       −2.700000        0.300000        3.000000
0.9999      −2.700000        0.300000        3.000000
```

The leftovers are dead constant in `γ` (no `e_γ` correction even appears here, because this two-state chain's remainder is exactly zero), and their *difference* is `3.000000` at every `γ` — exactly `h(1) − h(0) = 3 − 0`. The absolute levels `−2.7` and `+0.3` differ from my Poisson `h=[0,3]` by a common `−2.7`, which is just the `span{1}` freedom: same bias, different reference pin. So the `O(1)` coefficient of the discounted value really is the bias, up to the additive constant the theory already told me to expect. The Laurent picture isn't an abstraction I'm asserting — I can read the gain and the bias straight off the discounted value as `γ → 1`.

Let me re-check the order bookkeeping once more, because a dropped factor here would be fatal. The pole term contributes `g/(1−γ)` to the value — multiplying `1/(1−γ) → ∞`, so for large `γ`, `g/(1−γ) ≫ h + e_γ`; the value is dominated by the scaled gain. That's the same `V_γ → ∞` I worried about earlier, now decomposed. Two policies with different gains: the `1/(1−γ)` terms differ and, as `γ → 1`, the gain comparison swamps everything — the larger-gain policy eventually wins. Two policies with *equal* gain: the `1/(1−γ)` terms cancel and the comparison drops to the *next* term, the bias `h`.

And that cancellation is the clue to a problem I haven't faced yet. Maximal gain is sometimes *too coarse*. Picture a task where several policies all reach an absorbing goal and then sit there earning `0`: every one of them has long-run average reward `0`, all tied on gain. Gain can't separate them. But I obviously prefer the policy that reaches the goal *fastest*, hoarding the most reward on the way. Where does "fastest" live? In the **bias**. The finite advantage of the transient prefix — the reward you scoop up before settling into the recurrent class — is exactly what `h` measures. So among gain-optimal policies, I should break the tie by choosing the one with the largest `h(s)` at every state. Pictorially this is the `1/(1−γ)` terms canceling and the decision falling to the `O(1)` term: when gains tie, `lim_{γ→1} (V_γ^{π*} − V_γ^π) = h^{π*} − h^π`, so maximizing discounted value in the limit *is* maximizing bias. That refinement — gain-optimal **and** then bias-optimal — is the right notion of best for a continuing task. Howard's policy iteration, which optimizes gain first, lands on a bias-optimal policy when the first gain-optimal policy it finds already has the right recurrent structure; in general bias-optimality is the genuinely sharper criterion.

I can make this ordering systematic, and the Laurent expansion hands me the ladder. Define: `π*` is **`n`-discount-optimal** if `lim_{γ→1} (1−γ)^{-n} (V_γ^{π*}(s) − V_γ^π(s)) ≥ 0` for every state `s` and every competitor `π`. Read off what each `n` means by plugging in the expansion `V_γ = g/(1−γ) + h + e_γ`.

`n = −1`: multiply the difference by `(1−γ)`. Let me write it out so I don't fool myself: `(1−γ)(V_γ^{π*} − V_γ^π) = (g^{π*} − g^π) + (1−γ)(h^{π*} − h^π) + (1−γ)(e^{π*} − e^π)`. As `γ → 1` the last two groups vanish (the `h` terms are finite and multiplied by `1−γ → 0`; the `e` terms vanish anyway). So the condition is `g^{π*} − g^π ≥ 0` for all `π` — `(−1)`-discount-optimal is exactly gain-optimal.

`n = 0`: don't multiply at all. Now I must already be restricting to gain-optimal policies — otherwise the un-multiplied difference contains the `(g^{π*} − g^π)/(1−γ)` pole, which dominates and forces `g^{π*} ≥ g^π` anyway. Among gain-equal policies the pole cancels and `lim_{γ→1} (V_γ^{π*} − V_γ^π) = (h^{π*} − h^π) + lim(e^{π*} − e^π) = h^{π*} − h^π ≥ 0`. So `0`-discount-optimal is gain-optimal *plus* bias-optimal. The two criteria I motivated by hand are the `n = −1` and `n = 0` rungs of one ladder.

Higher `n` keeps going: each rung peels off the next Laurent coefficient `y_n` as the tie-breaker once all lower coefficients are tied. The whole sequence is nested — `m`-discount-optimal implies `n`-discount-optimal for every `n < m` — so the optimal sets shrink as `n` grows: all policies ⊃ gain-optimal ⊃ bias-optimal ⊃ … The limit of this nesting is the sharpest criterion of all: a policy that is `n`-discount-optimal for *every* `n ≥ −1`. Equivalently, a single policy that maximizes the discounted value for *all* `γ` sufficiently close to `1` — one policy that dominates on a whole interval `(γ*, 1)`, so there's no further tie left to break. That is **Blackwell optimality**, and Blackwell's result is that for a finite MDP such a policy exists: there is a `γ*` below `1` such that one stationary policy is discount-optimal for every `γ ∈ (γ*, 1)`, and it is exactly the policy that maximizes the Laurent coefficients lexicographically — gain first, then bias, then the rest. The discounting I started out distrusting is, in the end, what *defines* the finest optimality: not any single `γ`, but the behaviour as `γ → 1^-`.

Now the computation, because I want to actually solve these, not just characterize them. Policy iteration I already have, and I already watched it terminate on the two-state example: evaluate `(g, h)` by the Poisson solve with a reference pin, improve greedily, repeat — finitely terminating. But each evaluation is an `|S|`-dimensional linear solve, expensive when `|S|` is large. Can I iterate the optimality equation directly, the way value iteration iterates the discounted one? Define the average-reward backup `T(V)(s) = max_a [ r(s,a) + Σ_{s'} p(s'|s,a) V(s') ]` and try `V_{k+1} = T(V_k)`. Watch what happens: at the optimum `g* + h* = T(h*)`, so `T` shifts its fixed point *up by `g*` every application* — `T(h*) = h* + g*·1`. The iterates don't converge; they march off to infinity at rate `g*` per step, `V_k ≈ k·g* + h*`. That's the same divergence as the undiscounted sum, reappearing inside the iteration. The absolute values run away.

But the *relative* values — the differences `V_k(s) − V_k(s')` — are exactly what's converging, because the runaway part `k·g*` is a common constant added to every state and cancels in any difference. So I should iterate only the relative values. Subtract off a reference state at each step:

```
V_{k+1}(s) = T(V_k)(s) − T(V_k)(reference).
```

Now `V_{k+1}(reference) = 0` for all `k`, the iterates can't drift, and they should converge to `h` (up to the reference offset). I need to keep one raw quantity before the subtraction: if `U_k = T(V_k)`, then the drift vector `U_k − V_k` approaches `g*·1`, and the offset `U_k(reference) − V_k(reference)` approaches `g*` when the reference is pinned to zero. This is **relative value iteration**. And the stopping rule should ignore the common drift too: not the max-norm of the raw increment, which is stuck near `g*`, but the **span seminorm** `sp(V) = max_s V(s) − min_s V(s)`. Stop when `sp(U_k − V_k) < ε`: the span measures the spread of the *relative* update and is blind to any constant added to all states, which is precisely the part I don't care about and can't stop drifting. Let me make sure this solver agrees with policy iteration on the example, since they take completely different routes to the same pair. Running relative value iteration (with the aperiodicity transform below, `τ=0.5`) on the leaky MDP returns `g=2.7`, `h=[0,3]`, `pi=(1,1)` — the same gain, the same bias, the same policy that policy iteration found. Two independent solvers, one answer; that's the cross-check I wanted.

Two cautions I should bake in. First, periodicity. If the chain has a recurrent class of period `> 1`, the backup `T` doesn't contract in span and the iterates can oscillate forever instead of settling — value iteration needs the chain aperiodic under all policies (or a state reachable from everywhere). The fix is an aperiodicity transformation: blend in a self-loop, `P̃ = (1−τ)I + τP` with `0 < τ < 1`, and scale the rewards to `r̃ = τr`. Then `T̃(h) = h + τg`, the greedy action is unchanged because the transformed maximand is `(1−τ)h(s) + τ[r(s,a)+P_a h]`, and the transformed gain is `g̃ = τg`; the original gain is the measured drift divided by `τ`. Second, asynchrony. If I update states one at a time (Gauss–Seidel / asynchronous), relative value iteration can actually **diverge** — Tsitsiklis's two-state example shows the relative-value map is not monotone and not a max-norm contraction, so the convergence guarantees of asynchronous *discounted* value iteration do not carry over. That's a real trap: the synchronous sweep is safe; the obvious asynchronous shortcut is not, without further care.

Let me pull the pieces together. The pain was a continuing task with no horizon, where discounting injects an arbitrary `γ` that can rank policies wrong (the `γ ≈ 0.8409` crossover) and blows the value up as `γ → 1`. The honest objective is the long-run average `g`. For a unichain policy `g` is a state-independent scalar, which carries no ranking — so a single value function is not enough. Trying `γ = 1` directly hits the singularity of `I − P`, whose null space `span{1}` is the same degeneracy as the state-independent gain. Reintroducing `g` as a per-step toll and asking for the finite relative advantage of each start state produces the bias `h` and the Poisson equation `g + h(s) = r(s) + Σ p h(s')`; solvable precisely because `g = d·r_π`, as I checked. The same equation drops out independently as the `O(1)` coefficient of the Laurent expansion `V_γ = g/(1−γ) + h + O(1−γ)`, whose `O(1/(1−γ))` coefficient is the gain — and I read both coefficients straight off the discounted value numerically as `γ → 1`. Taking the `max` over actions gives the average-reward Bellman optimality equation `g* + h*(s) = max_a [r(s,a) + Σ p(s'|s,a) h*(s')]`, characterizing the optimum by the **pair** `(g*, h*)`, which I watched policy iteration converge to (residual zero) and relative value iteration reproduce. Gain alone is coarse; the Laurent ladder refines it — `(−1)`-discount-optimal = gain-optimal, `0`-discount-optimal = also bias-optimal, and `∞`-discount-optimal = Blackwell-optimal, one policy that wins for all `γ` near `1`. Policy iteration (Poisson-solve + greedy) finds a gain-optimal policy in finitely many steps; the Laurent/Blackwell refinement says how to break equal-gain ties by bias and higher coefficients; relative value iteration with reference subtraction and a span-seminorm stop iterates to `h` and recovers `g*` from the raw drift. The one boundary I hit and won't paper over: the scalar-`g` Poisson formulation needs a single recurrent class — on a multichain policy the evaluation matrix is genuinely singular, because the gain stops being one number. So the scope is unichain (or communicating). That's the landing — the optimality equations plus the gain/bias characterization, with policy iteration and RVI as the concrete solvers.

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

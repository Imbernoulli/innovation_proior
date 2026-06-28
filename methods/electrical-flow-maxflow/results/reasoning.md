Thirty-five years. For undirected unit-capacity graphs with `m = O(n)` edges, the fastest way to compute a maximum `s-t` flow is still the blocking-flow method, `O(n^{3/2}) = O(m^{3/2})`, and nobody has moved it. Goldberg and Rao squeezed the general capacitated case down to `Õ(m√n)` with a binary length function and blocking flows, and that is genuinely beautiful, but it still has the `√n`. And they say the honest thing themselves: there is a flow-decomposition barrier. Any algorithm that builds an explicit flow decomposition and augments path by path, one arc at a time, can be forced to do `Θ(mn)` work, because the total length of the paths in a decomposition is `Θ(mn)` in the worst case. So the `√n`, or the `√m`, is not an accident of one algorithm — it is baked into the *paradigm*. Path augmentation, layered graphs, blocking flows — all of it lives inside that barrier.

So I should stop trying to make path augmentation faster. I want a different primitive entirely. Something that does not decompose the flow into paths at all. Something that, in one shot, produces a whole flow vector and that I can compute fast.

What do I actually know how to compute fast on a graph? Here is the one thing that has changed in the last few years: we can solve Laplacian linear systems — symmetric diagonally dominant systems — in nearly linear time. Spielman-Teng, then Koutis-Miller-Peng. That is a genuinely new hammer. The question is whether maximum flow is a nail for it.

Let me look hard at what a Laplacian solve gives me. Put a resistance `r_e > 0` on each edge, conductance `c_e = 1/r_e`. The Laplacian is `L = B C Bᵀ` where `B` is the incidence matrix and `C = diag(c_e)`. If I solve `L φ = χ_{s,t}` — `χ_{s,t}` being `+1` at `s`, `−1` at `t`, zero elsewhere — I get vertex potentials `φ`, and then the flow `f = C Bᵀ φ`, i.e. `f(u,v) = (φ_v − φ_u)/r_{uv}`, Ohm's law. That `f` satisfies `Bf = χ_{s,t}`: it is an honest `s-t` flow of value one. It is the current that flows if I treat the graph as a resistor network and push a unit of current from `s` to `t`.

And what does this current minimize? It is the *electrical flow*: among all `s-t` flows of value one, it is the unique minimizer of the energy `E_r(f) = Σ_e r_e f(e)²`. Let me make sure I believe that and that it really comes out of the linear system. Energy is `fᵀ R f` with `R = C^{-1} = diag(r_e)`, which is `‖R^{1/2} f‖²`. So the electrical flow is the minimum-`ℓ_2`-norm (in the `R`-metric) flow of value one. Minimum-norm subject to a linear constraint `Bf = χ` — that is a projection, and Lagrange conditions say the optimal `f` is of the form `R^{-1} Bᵀφ = C Bᵀφ` for some `φ`, a potential flow; plug into `Bf = χ` and `B C Bᵀ φ = L φ = χ`, so `φ = L⁺ χ`. Yes. The electrical flow is exactly what the Laplacian solve gives me, and it is the *minimum-energy* flow.

Here is the thing that makes this feel like the right hammer. Maximum flow, written as I want to write it, is: push as much flow value `F` as possible while keeping `max_e |f(e)|/u_e ≤ 1`. That is an `ℓ_∞` problem — I am controlling the *largest* congestion. Electrical flow minimizes a *sum of squares* — an `ℓ_2` problem. Those are different objectives. The electrical flow will happily overload one edge if doing so lowers the total energy. So electrical flow is not max flow. But `ℓ_2` is the convex relaxation I can actually *solve in nearly linear time*, and `ℓ_2` and `ℓ_∞` are not unrelated — `‖x‖_∞ ≤ ‖x‖_2 ≤ √m ‖x‖_∞`. Maybe I can route through `ℓ_2` and pay only a `√m`-ish factor, and then beat the factor down.

Let me first just see how bad the electrical flow is as a flow. Take resistances `r_e = 1` everywhere on a unit-capacity graph and compute the electrical flow that pushes `F` units. Conservation holds for free. Capacities? No guarantee at all. How badly can one edge be overloaded?

I want a graph where I can predict the answer by hand, so I can both build intuition and check the code against it. Take `k` parallel paths, each of length `k`, from `s` to `t`, plus one single edge directly from `s` to `t`. The number of edges is `m = k·k + 1 ≈ k²`. The max flow is `k + 1` — one unit down each path plus one across the direct edge. The electrical flow with unit resistances should split by Ohm's law: the direct edge is a single resistor of resistance `1`; each of the `k` paths is `k` resistors in series, resistance `k`. By symmetry the `k` paths each carry the same current `I_p` and the direct edge carries `I_d`, with the same voltage drop `φ_s − φ_t = V` across all of them: `V = I_d·1 = I_p·k`, so `I_d = k·I_p`. Conservation: `k·I_p + I_d = k+1`, i.e. `I_p + I_d = (k+1)/k·I_p`... let me just substitute. `k I_p + k I_p = k+1`? No — `k·I_p + I_d = k I_p + k I_p = 2k I_p = k+1`, so `I_p = (k+1)/(2k)` and `I_d = k I_p = (k+1)/2`. So the direct edge, capacity one, carries `(k+1)/2 ≈ k/2 ≈ √m/2`.

Before I build a whole theory on that number, let me actually run the Laplacian solve on this graph and read off the direct edge's flow, because a hand-derivation of a resistor network is exactly the kind of thing I get wrong by a factor of two. Building the family for `k = 4, 9, 16, 25` and solving `Lφ = (k+1)χ` numerically, the direct edge carries `2.5, 5.0, 8.5, 13.0` — exactly `(k+1)/2` in every case — against `√m = 4.12, 9.06, 16.03, 25.02`, a ratio settling to about `0.52`. So my hand-derivation was right, and the overload really is `Θ(√m)`: `(k+1)/2` over a capacity of `1`. That is the `ℓ_2`-vs-`ℓ_∞` gap made concrete — electrical flow loves the short path and shoves `√m` units across it. I will keep this graph; it is going to be the stress test for whatever I build.

So a single electrical flow is a max flow that can be wrong by `√m` on some edge. I cannot return it. But `√m` is exactly the kind of factor I was hoping to start at and then reduce. I need a wrapper that calls this fast-but-overloading primitive repeatedly and converges to something feasible.

Multiplicative weights. That framework is built for exactly this shape: you have many constraints (here, the `m` capacity constraints), you have an oracle that can only satisfy them *on weighted average*, and you reweight to force the oracle to fix the constraints it keeps violating. Let me set it up carefully, because the convergence rate is going to depend on a quantity that the literature calls the *width* — the worst single-constraint violation the oracle can produce — and I should watch where that quantity enters, because it will set the iteration count.

Let me define what I need from the oracle. Maintain a weight `w_e ≥ 1` on each edge. I will hand the oracle the weights `w` and a target value `F`, and I want it to return an `s-t` flow `f` of value exactly `F` such that: the *weighted average* congestion is small, `Σ_e w_e cong(f,e) ≤ (1+ε)‖w‖_1` where `‖w‖_1 = Σ_e w_e`; and the *worst* congestion is bounded by some width `ρ`, `max_e cong(f,e) ≤ ρ`. And if `F > F*` it is allowed to fail. Call this an `(ε, ρ)`-oracle. The point of the asymmetry: the average constraint is easy to hit (that is the `ℓ_2`-ish guarantee), the max constraint is the hard one (that is the `√m`), and multiplicative weights is precisely the machine that turns "satisfies the weighted average" into "satisfies all of them," at a cost proportional to `ρ`.

Now the outer loop. Start `w_e = 1` for all `e`. Run `N` rounds. In round `i`, call the oracle with the current weights, get `f^i`. Then reweight: `w_e ← w_e (1 + (ε/ρ) cong(f^i, e))`. An edge that was congested near `ρ` gets multiplied by roughly `(1+ε)`; an edge flowing within capacity barely changes. So weight piles onto the edges that keep getting overloaded, which forces the next oracle call to relieve them. At the end, return the *average* of the flows, suitably scaled.

I need to nail down `N` and the scaling, and to do that I need the convergence proof. Let me run the standard exponential-potential argument, but instantiated here so I can see exactly what controls `N`.

Potential `μ_i = ‖w^i‖_1`. It starts at `μ_0 = m` and only grows. How fast can it grow in one step?

`μ_{i+1} = Σ_e w_e^i (1 + (ε/ρ) cong(f^{i+1}, e)) = μ_i + (ε/ρ) Σ_e w_e^i cong(f^{i+1}, e)`.

The oracle's average guarantee says `Σ_e w_e^i cong(f^{i+1}, e) ≤ (1+ε) μ_i`. So

`μ_{i+1} ≤ μ_i + (ε(1+ε)/ρ) μ_i = μ_i (1 + ε(1+ε)/ρ) ≤ μ_i exp(ε(1+ε)/ρ)`.

Iterating, `μ_N ≤ m exp(ε(1+ε) N / ρ)`. Good — the total weight stays controlled, growing exponentially at rate `ε(1+ε)/ρ` per step.

Now I want to *lower*-bound the weight of a single edge in terms of how congested it was on average, because that is what will force feasibility. From the update,

`w_e^N = Π_{j=1}^{N} (1 + (ε/ρ) cong(f^j, e))`.

I want to turn the product into an exponential of the *sum* of congestions, i.e. of the average congestion across the run. The factor is `1 + ε x` with `x = cong(f^j,e)/ρ ∈ [0,1]` (here is where the width matters: `cong ≤ ρ` is exactly what makes `x ≤ 1`). I want `1 + ε x ≥ exp(c · ε x)` for some `c < 1`. Let me find the constant. For `x ∈ [0,1]` and `ε ∈ (0, 1/2)`, is `1 + ε x ≥ exp((1−ε) ε x)`? At `x = 0` both are `1`. Taking logs, I want `ln(1 + εx) ≥ (1−ε) ε x`. Since `ln(1+y) ≥ y − y²/2` for `y ≥ 0`, with `y = εx ≤ ε`, `ln(1+εx) ≥ εx − ε²x²/2 ≥ εx(1 − εx/2) ≥ εx(1 − ε/2) ≥ εx(1 − ε)`. So it should hold for all `ε ∈ (0,1)`, `x ∈ [0,1]`. The chain of `≥`'s is slippery enough that I don't want to trust it on paper, so let me sweep `(1+εx) − exp((1−ε)εx)` over a fine grid of `(ε, x) ∈ (0,1)×[0,1]`: the minimum gap comes out `0` (attained only at `x=0`, where both sides are `1`) and the value is nonnegative everywhere else. A spot point: `ε=1/4, x=1` gives `1.25` versus `exp(0.1875) ≈ 1.206`, gap `+0.044`. Good — the inequality holds, with room to spare except at the trivial corner.

So

`w_e^N ≥ Π_{j} exp((1−ε)(ε/ρ) cong(f^j,e)) = exp((1−ε)(ε/ρ) Σ_j cong(f^j,e))`.

Now the average flow. Define `f̄ = (1/N) Σ_j f^j` for the moment (I will add the scaling shortly). The average congestion of an edge is `cong(f̄, e) ≤ (1/N) Σ_j cong(f^j, e)` (the absolute value of the average is at most the average of the absolute values). So `Σ_j cong(f^j,e) ≥ N · cong(f̄, e)` — wait, that inequality goes the wrong way for a lower bound. Let me be careful: I have `cong(f̄,e) ≤ (1/N)Σ_j cong(f^j,e)`, equivalently `Σ_j cong(f^j,e) ≥ N cong(f̄,e)`. That *is* a lower bound on the sum, so it gives a lower bound on `w_e^N`:

`w_e^N ≥ exp((1−ε)(ε/ρ) · N cong(f̄, e))`.

Now squeeze. The single-edge weight can never exceed the total weight: `w_e^N ≤ μ_N ≤ m exp(ε(1+ε) N / ρ)`. Combine:

`exp((1−ε)(ε/ρ) N cong(f̄,e)) ≤ m exp(ε(1+ε) N/ρ)`.

Take logs and solve for `cong(f̄,e)`:

`(1−ε)(ε/ρ) N cong(f̄,e) ≤ ln m + ε(1+ε) N/ρ`,
`cong(f̄,e) ≤ (1+ε)/(1−ε) + (ρ ln m)/((1−ε) ε N)`.

Hm — `(1+ε)/(1−ε)` is bigger than one, so the average flow is not feasible as is; it can overload by up to that factor plus a tail. This is why the returned flow has to be *scaled down*. If I scale `f̄` by `(1−ε)²/(1+ε)` and choose `N` to kill the tail term, the congestion comes under one. Let me set `N = 2 ρ ln m / ε²`. Then the tail is `(ρ ln m)/((1−ε) ε · 2ρ ln m/ε²) = ε/(2(1−ε))`. And let me redo the squeeze with the cleaner bookkeeping the scaling demands: I will define the returned flow as `f̄ = ((1−ε)²/((1+ε)N)) Σ_j f^j`, so each per-edge congestion of the *sum* gets divided by `N` and multiplied by `(1−ε)²/(1+ε)`. Tracking the constants through (the `(1−ε)` from the log-inequality, the `(1+ε)` from the average growth), the bound that comes out is

`cong(f̄, e) ≤ 1 − ε + ε(1−ε)/(2(1+ε)) ≤ 1`,

for every edge and every `ε ∈ (0, 1/2)`. (Sanity: at `ε = 1/4`, `1 − 0.25 + (0.25·0.75)/(2·1.25) = 0.75 + 0.075 = 0.825 ≤ 1`.) So `f̄` is feasible. And its value: each `f^i` has value `F`, the sum has value `NF`, scaled by `(1−ε)²/((1+ε)N)` gives value `(1−ε)²/(1+ε) F ≥ (1 − O(ε)) F`. So if I can build an `(ε, ρ)`-oracle, multiplicative weights gives a `(1−O(ε))`-approximate max flow in `N = 2ρ ln m/ε²` oracle calls.

And the cost is `Õ(ρ ε^{-2})` oracle calls. So the iteration count is linear in `ρ` and nothing else of substance; `ρ` is the one knob that, if I can shrink it, makes the whole thing faster. Whatever the oracle's actual width turns out to be, that is what I will be fighting to reduce.

Now build the oracle out of one electrical flow. I want resistances `r_e`, depending on `w` and `u`, so that a single electrical-flow solve gives both the weighted-average congestion bound and a worst-case width. The energy is `E_r(f) = Σ_e r_e f(e)²`. I want energy to "be" congestion-squared, so let me put the capacity into the resistance: `r_e ∝ 1/u_e²` makes `r_e f(e)² ∝ (f(e)/u_e)² = cong(f,e)²`. Specifically, try

`r_e = (1/u_e²)(w_e + ε‖w‖_1/(3m))`.

Then `E_r(f) = Σ_e (w_e + ε‖w‖_1/3m) cong(f,e)²`. Two terms, and each does a job. The `w_e cong²` part will give the weighted-average congestion. The additive `ε‖w‖_1/3m` part is a *floor* under every resistance — and that floor is what controls the worst case. Let me see why the floor is needed at all before trusting it. If I used only `r_e = w_e/u_e²`, an edge whose weight `w_e` had decayed to nearly nothing would have nearly zero resistance, so the electrical flow could shove an unbounded amount of current across it at almost no energy cost — the width would be uncontrolled. The floor `ε‖w‖_1/3m` guarantees every edge has resistance at least `ε‖w‖_1/(3m u_e²)`, so a large congestion on any single edge forces a large energy contribution, which I can cap. So: `w_e` term for the average, floor term for the max.

The oracle: compute the (approximate) electrical flow of value `F` with these resistances. If its energy exceeds `(1+ε)‖w‖_1`, declare fail. Otherwise return it. Let me verify both that it never wrongly fails and that "energy ≤ `(1+ε)‖w‖_1`" implies the two congestion bounds.

First, never-wrongly-fail when `F ≤ F*`. Let `f*` be a feasible max flow, so `cong(f*, e) ≤ 1`. Its energy:

`E_r(f*) = Σ_e (w_e + ε‖w‖_1/3m) cong(f*,e)² ≤ Σ_e (w_e + ε‖w‖_1/3m) = ‖w‖_1 + ε‖w‖_1/3 = (1 + ε/3)‖w‖_1`,

using `cong ≤ 1` and `Σ_e ε‖w‖_1/3m = ε‖w‖_1/3`. Now, when `F ≤ F*`, there is a feasible flow of value `F` (scale down `f*`), and the electrical flow of value `F` is the *minimum-energy* flow of that value, so its energy is at most this feasible witness's, `≤ (1+ε/3)‖w‖_1`. That is the whole reason the floor constant is `ε/3` rather than something larger: I have a `(1+ε/3)` slack between the true min energy and the threshold, and I need to spend it on the solver's own approximation. If the solver returns a flow with energy `≤ (1 + ε/3)` times the true minimum, the returned energy is `≤ (1+ε/3)² ‖w‖_1`, and I want that `≤ (1+ε)‖w‖_1`, i.e. `(1+ε/3)² ≤ 1+ε`. Expanding, `(1+ε/3)² = 1 + 2ε/3 + ε²/9`, and `1+ε − (1+2ε/3+ε²/9) = ε/3 − ε²/9 = (ε/9)(3 − ε) > 0` for `ε < 3`. So the inequality holds with margin throughout `ε ∈ (0,1)` (numerically, at `ε=0.5`: `(1.1667)² = 1.361 ≤ 1.5`; at `ε=1`: `1.778 ≤ 2`). The oracle therefore never returns fail when `F ≤ F*`.

Second, the energy bound implies the congestion bounds. Suppose `E_r(f̃) ≤ (1+ε)‖w‖_1`. Drop the floor term to get `Σ_e w_e cong(f̃,e)² ≤ (1+ε)‖w‖_1`. By Cauchy-Schwarz, `(Σ_e w_e cong)² ≤ (Σ_e w_e)(Σ_e w_e cong²) ≤ ‖w‖_1 · (1+ε)‖w‖_1`, so `Σ_e w_e cong(f̃,e) ≤ √(1+ε) ‖w‖_1 ≤ (1+ε)‖w‖_1`. That is the weighted-average bound. For the worst case, keep only the floor term: for each `e`, `(ε‖w‖_1/3m) cong(f̃,e)² ≤ (1+ε)‖w‖_1`, so `cong(f̃,e)² ≤ 3m(1+ε)/ε`, giving `cong(f̃,e) ≤ √(3m(1+ε)/ε) ≤ 3√(m/ε)` for `ε < 1/2`. So this is an `(ε, 3√(m/ε))`-oracle. There is the `√m` width again, as expected from the bad graph.

Cost of one oracle call: one approximate Laplacian solve. The solver's time depends on the ratio `R` of largest to smallest resistance through a `log R` factor. With capacities in `[1, poly(m/ε)]` (the preprocessing reduction), `R ≤ poly(m/ε)`, so `log R = O(log(m/ε))` and the solve is `Õ(m)`. Total time `Õ(ρ ε^{-2} · m) = Õ(√m ε^{-5/2} · m) = Õ(m^{3/2} ε^{-5/2})`.

So I already broke nothing — I matched `m^{3/2}` up to `ε` and polylog. That is not yet a win; the win has to come from the width. The bad graph says `ρ = Θ(√m)` is *tight* for a single electrical flow: the direct edge really does carry `Θ(√m)`. I cannot improve the width by analyzing the same flow more cleverly. I have to change what the oracle does.

Stare at the bad graph again. The offending edge is the single direct `s-t` edge carrying `√m` units. Now — what if I just deleted it? Let me actually try it on the same family, because if deletion only moved the overload somewhere else it would be a dead end. Drop the direct edge, leaving `k` parallel length-`k` paths, and recompute the electrical flow for the remaining max flow value `F = k`. The numbers come back clean: for `k = 4, 9, 16, 25` the maximum `|f(e)|` over all edges is exactly `1.0` — each path now carries precisely one unit, perfectly balanced, no edge over capacity. And the max flow dropped only from `k+1` to `k`, a relative loss of `1/(k+1) = 0.20, 0.10, 0.059, 0.038` — shrinking to nothing as the graph grows. So the bad edge was *fragile*: it was the single channel carrying a huge current, and killing it both fixed the electrical flow and barely touched the optimum. That suggests a general phenomenon worth betting on — that the few edges an electrical flow overloads are exactly the few that can be removed cheaply — though one example is not a proof of it, and I'll have to earn the general statement.

So modify the oracle: pick a target width `ρ` *smaller* than `√m` — I will tune it, it will land at about `m^{1/3}`. Compute the electrical flow. If some edge has congestion exceeding `ρ`, *remove that edge from the graph* (add it to a permanent forbidden set `H`), keep all the other weights, and recompute. Repeat until every edge is within `ρ` (or a genuine failure — `s, t` disconnected, or energy too high). Removed edges stay removed forever, in every future oracle call.

Two things to prove, and they are the whole content of the improvement. (1) The removals do not destroy too much flow — the total capacity of `H` must stay below, say, `εF/12`, so a feasible flow of value `(1−ε/12)F` still survives and the oracle never wrongly fails. (2) The total number of removals across the whole algorithm is small — at most `Õ(m^{1/3})` — so the extra electrical solves do not blow up the running time.

To bound the number of removals I want a monotone quantity — something that only moves one way as the run proceeds and is bounded on both ends, so each removal can be charged against a chunk of it. Both removing an edge and raising a weight do the same thing to the network: they raise resistances (deletion is `r → ∞`). So I want a quantity that responds monotonically to raising resistances. The effective `s-t` resistance is the natural candidate — physically, opening or thickening resistors should make it harder, not easier, to push current across. If that monotonicity is real, and if I can further show each *removal* forces a *substantial multiplicative* jump (not just an increase), then a ceiling on the total resistance caps the count. Let me check whether `R_eff` actually behaves that way before committing to it.

Let me first get the monotonicity and the quantitative jump straight, because everything rests on them. The variational identity for effective conductance (Thomson's principle): `C_eff(r) = 1/R_eff(r) = min_{φ: φ_s=1, φ_t=0} Σ_{(u,v)} (φ_u − φ_v)²/r_{uv}`, with the minimizing `φ` being the electrical potentials of the unit-conductance flow. Rayleigh monotonicity falls right out: if `r'_e ≥ r_e` for all `e`, then for *every* `φ`, `Σ (φ_u−φ_v)²/r'_{uv} ≤ Σ (φ_u−φ_v)²/r_{uv}`, so the minima obey `C_eff(r') ≤ C_eff(r)`, hence `R_eff(r') ≥ R_eff(r)`. Raising resistances raises effective resistance. Good.

Now the quantitative jump. Suppose in the current electrical flow `f`, some edge `h` carries a `β` fraction of the total energy, `f(h)² r_h = β E_r(f)`. I raise `r_h` to `γ r_h` (and `γ = ∞` is "cut it"). Claim: `R_eff(r') ≥ (γ/(β + γ(1−β))) R_eff(r)`.

Prove it through the conductance min. Normalize `f` to be the unit-*conductance* flow, i.e. the electrical flow of value `1/R_eff` with potentials `φ` satisfying `φ_s − φ_t = 1`; shift so `φ_s = 1, φ_t = 0`. Then `C_eff(r) = Σ (φ_u−φ_v)²/r_{uv}`, and the energy fraction assumption says the `h` term is `(φ_i − φ_j)²/r_h = β C_eff(r)`, so the rest is `Σ_{e≠h} = (1−β) C_eff(r)`. Now upper-bound `C_eff(r')` by plugging this *same* `φ` into its min (a feasible point, so an upper bound):

`C_eff(r') ≤ (φ_i−φ_j)²/(γ r_h) + Σ_{e≠h} (φ_u−φ_v)²/r_e = (β/γ) C_eff(r) + (1−β) C_eff(r) = C_eff(r) (β + γ(1−β))/γ`.

Invert: `R_eff(r') ≥ (γ/(β + γ(1−β))) R_eff(r)`. For `γ = ∞`, the limit is `R_eff(r') ≥ R_eff(r)/(1−β)`. For a gentle bump `γ = 1+ε`: `γ/(β+γ(1−β)) = (1+ε)/(1 + ε(1−β))`, and `(1+ε)/(1+ε(1−β)) ≥ 1 + εβ/2` for small `ε` after expanding, so `R_eff(r') ≥ (1 + εβ/2) R_eff(r)`.

Both monotonicity and this bound came out of plugging the *same* `φ` into a min, which feels almost too cheap — let me put numbers to it on a tiny network to make sure I have not fooled myself with the direction of an inequality. Take `s=0, t=1` with a direct edge `(0,1)` and a length-2 path `(0,2),(2,1)`, resistances `(1, 2, 3)`. Solving the unit flow gives `R_eff = 0.8333`, and the direct edge's energy fraction is `β = 0.8333` (it is the low-resistance shortcut, so it hogs the energy — consistent with the whole `√m` story). Now raise the direct edge's resistance by various `γ` and recompute `R_eff` directly, comparing to the lemma's predicted lower bound `(γ/(β+γ(1−β)))·0.8333`:

- `γ=2`: recomputed `1.4286`, predicted `1.4286` — equal;
- `γ=10`: recomputed `3.3333`, predicted `3.3333` — equal;
- `γ=1.2`: recomputed `0.9677`, predicted `0.9677` — equal;
- `γ=∞` (cut): recomputed `5.0000`, and `R_eff/(1−β) = 0.8333/0.1667 = 5.0000` — equal.

The bound is not just satisfied, it is *tight* for this flow — which makes sense, because for the unit-conductance flow the `φ` I plug in is the actual optimizer of the `r` problem, so the only slack is from `r'` re-optimizing, and here a single edge change leaves little room. (One caveat the tightness exposes: the `γ=1.2` *recomputed* jump is `0.9677/0.8333 = 1.161`, whereas my linearized `1+εβ/2 = 1.083` is smaller — so the linear bump bound is a genuine under-estimate, safe to use, not an over-claim.) Good: cutting an edge that carries a `β` fraction of the energy multiplies `R_eff` by at least `1/(1−β)`, exactly. The bigger the fraction the edge hogs, the bigger the jump.

This is the mechanism that will cap the removals. When I remove an edge in the improved oracle, it is precisely an edge with `cong > ρ`; what remains is to show such an edge necessarily carries a *substantial* fraction of the energy, so that the jump is substantial.

Quantify that fraction. Suppose I am about to remove `h` because `cong(f̃, h) > ρ`, and the oracle has not failed, so `E_r(f̃) ≤ (1+ε)‖w‖_1`. The energy edge `h` contributes is

`f̃(h)² r_h = f̃(h)² (w_h + ε‖w‖_1/3m)/u_h² ≥ f̃(h)² (ε‖w‖_1/3m)/u_h² = (ε/3m) cong(f̃,h)² ‖w‖_1`,

dropping the `w_h` part and using the floor. Now `cong(f̃,h) > ρ`, so this is `> (ε ρ²/3m) ‖w‖_1`. As a fraction of the total energy `E_r(f̃) ≤ (1+ε)‖w‖_1`, edge `h` accounts for more than `ε ρ²/(3(1+ε)m)`. So `β > ε ρ²/(3(1+ε)m)` for the *approximate* flow.

One subtlety: the jump lemma is about the *exact* electrical flow's energy distribution, but I am working with an approximate flow `f̃`. I need the energy fraction to transfer to the true flow `f`. This is where I will lean on a per-edge guarantee from the solver — that `|r_e f_e² − r_e f̃_e²|` is tiny relative to the total energy (I will prove the solver gives this in a moment). With that, the true flow's fraction is at least `ε ρ²/(3(1+ε)m) − (small) > ε ρ²/(5m)`. So in the exact flow, `h` carries more than an `ε ρ²/(5m)` fraction, and by the jump lemma with `γ = ∞`,

`R_eff(after) ≥ R_eff(before)/(1 − ε ρ²/(5m))`, i.e. `(1 − ε ρ²/(5m)) Φ(after) > Φ(before)`,

where I write `Φ` for the effective resistance of the current circuit (with removed edges at `r = ∞`). Each removal multiplies `Φ` by at least `1/(1 − ε ρ²/(5m))`.

Now bound how high `Φ` can climb, to cap the number of removals. I need a floor and a ceiling on `Φ`.

Floor — `Φ` at the very first solve. Initially `H = ∅`, weights are `1`, so `r_e = (1 + ε/3)/u_e²`. Let `(S, V∖S)` be a minimum cut; by max-flow min-cut its capacity `u(S) = F*`. The unit electrical flow sends one unit across this cut, so some crossing edge `e'` carries `f(e') ≥ 1/m` (there are at most `m` crossing edges, and they carry a total of `≥ 1` net). And `r_{e'} = (1+ε/3)/u_{e'}² > 1/u_{e'}² ≥ 1/(u(S))² = 1/F*²` since `u_{e'} ≤ u(S) = F*`. So

`Φ(1) = E_r(f) = Σ_e f(e)² r_e ≥ f(e')² r_{e'} > (1/m²)(1/F*²) = 1/(m² F*²)`.

With `F* ≤ mF` (the crude bracketing), `Φ(1) > 1/(m² (mF)²) = m^{-4} F^{-2}`.

Ceiling — at the moment just before the last removal. The energy of an `s-t` flow of value `F` is at least `F²` times the effective resistance (because the unit electrical flow has energy `R_eff` and energy scales with the square of the value). So `Φ(j) = R_eff(r^j) ≤ E_r^j(f̃)/F² ≤ (1+ε)‖w‖_1/F²`, where the last step used the oracle's energy bound. And `‖w‖_1 ≤ μ_N ≤ m exp(ε(1+ε)N/ρ)`. With `N = 2ρ ln m/ε²`, the exponent is `ε(1+ε)·2ρ ln m/ε²/ρ = 2(1+ε) ln m/ε ≤ 3 ln m/ε` for `ε < 1/2`, so `‖w‖_1 ≤ m · m^{3/ε} = m^{1+3/ε}`. I want the *ratio* `Φ(j)/Φ(1)`, so divide the ceiling `(1+ε)‖w‖_1/F²` by the floor `m^{-4}F^{-2}`: the `F²`'s cancel and I get `Φ(j)/Φ(1) ≤ (1+ε)‖w‖_1 m⁴`. Combining floor and ceiling, after `k−1` removals up to step `j`,

`(1 − ε ρ²/(5m))^{-(k−1)} ≤ Φ(j)/Φ(1) ≤ (1+ε)‖w‖_1 m⁴ ≤ 2 m⁵ exp(3 ε^{-1} ln m)`.

Take logs. Using `ln(1 − c) < −c`, the left side's log is `> (k−1) ε ρ²/(5m)`. The right side's log is `< ln 2 + 5 ln m + 3 ε^{-1} ln m < 6 ε^{-1} ln m`. So

`(k−1) ε ρ²/(5m) < 6 ε^{-1} ln m`, giving `k < 30 m ln m/(ε² ρ²) + 1 < 30 m ln m/(ε² ρ²)`.

That is the cardinality bound on `H`: at most `30 m ln m/(ε² ρ²)` edges are ever removed. And the capacity: I only remove `e` when more than `ρ u_e` units flow across it, but no edge ever carries more than `F` units, so `u_e < F/ρ`. Hence `u(H) ≤ |H| F/ρ ≤ 30 m F ln m/(ε² ρ³)`.

Now choose `ρ`. The number of electrical solves is `N + |H|`, the MW iterations plus the removals: `N = 2ρ ln m/ε² = Õ(ρ/ε²)` and `|H| = Õ(m/(ε² ρ²))`. These trade off: raising `ρ` raises `N` but lowers `|H|`. They balance when `ρ/ε² ≈ m/(ε² ρ²)`, i.e. `ρ³ ≈ m`, `ρ ≈ m^{1/3}`. Set `ρ = 8 m^{1/3} ln^{1/3} m/ε`. Then `|H| ≤ (15/32)(m ln m)^{1/3} = Õ(m^{1/3})`, and crucially the capacity `u(H) ≤ 30 m F ln m/(ε² ρ³) = (15/256) ε F < εF/12`. So removed edges carry less than `εF/12` total capacity: the max flow drops by less than `εF/12`, `s` and `t` never get disconnected, and a feasible flow of value `(1 − ε/12)F` always survives. That makes the oracle's never-fail proof go through with slightly worse constants — there is still a flow of value `F` with every congestion at most `1/(1−ε/12)`, so `E_r(f̃) ≤ (1+ε/12)²(1+ε/3)²‖w‖_1 ≤ (1+ε)‖w‖_1`. The oracle is a legitimate `(ε, ρ)`-oracle with `ρ = Õ(m^{1/3})`.

Total electrical solves: `N + |H| = Õ(m^{1/3}/ε³) + Õ(m^{1/3}) = Õ(m^{1/3} ε^{-3})`. Each solve is `Õ(m)`. So the running time is `Õ(m^{4/3} ε^{-3})`. That breaks the `m^{3/2}` barrier.

Push further with off-the-shelf machinery. Karger's graph-smoothing theorem says a `(1−ε)`-flow routine of cost `T(m,n,ε)` yields one of cost `Õ(ε² m/n · T(Õ(n ε^{-2}), n, Ω(ε)))`. Feed in `T = Õ(m^{4/3} ε^{-3})`: the inner graph has `Õ(n ε^{-2})` edges, so `T(Õ(nε^{-2}), n, Ω(ε)) = Õ((nε^{-2})^{4/3} ε^{-3})`, times `ε² m/n`, which works out to `Õ(m n^{1/3} ε^{-11/3})`. And for the cut *value*, run on a Benczúr-Karger sparsifier with `O(n log n/ε²)` edges, giving `Õ(m + n^{4/3} ε^{-3})`.

Now I owe two things I leaned on: the solver-to-flow conversion (the three properties of an approximate electrical flow), and a separate, cleaner dual algorithm for the cut.

The solver. Koutis-Miller-Peng (building on Spielman-Teng) solves an SDD system `L φ = F χ` and returns `φ̂` with relative error in the *Laplacian norm*: `‖φ̂ − φ‖_L ≤ ε' ‖φ‖_L`, where `‖y‖_L = √(yᵀ L y)`, in time `Õ(m log(1/ε'))`. Note `‖φ‖_L² = φᵀ L φ = E_r(f)` — the Laplacian norm of the potentials *is* the energy. I need to turn this relative potential error into the three flow properties my analysis used: (a) the rounded flow has energy `≤ (1+δ) E_r(f)`; (b) per-edge energy is right to within `δ/(2mR) E_r(f)`; (c) the potential drop `φ̂_s − φ̂_t ≥ (1 − δ/(12nmR)) F R_eff`.

Two problems with `φ̂`. First, the flow `f̂ = C Bᵀ φ̂` is not quite an `s-t` flow — `φ̂` only approximately solves the system, so a little flow leaks in or out of interior vertices: `B f̂ = i_ext` with `Σ i_ext = 0` but `i_ext ≠ Fχ`. Second, I need the per-edge guarantees.

Energy first. `‖φ̂‖_L ≤ ‖φ‖_L + ‖φ̂ − φ‖_L ≤ (1+ε')‖φ‖_L`, so `E_r(f̂) = ‖φ̂‖_L² ≤ (1+ε')² E_r(f)`. The leak is small: let `i_ext = B f̂`, and `η = ‖i_ext − Fχ‖_∞ ≤ ‖i_ext − Fχ‖_2 = ‖L φ̂ − L φ‖_2`. To turn this into the `L`-norm error, write `x = φ̂ − φ` and bound `‖L x‖_2 ≤ ‖L‖_2^{1/2} ‖L^{1/2} x‖_2 = ‖L‖_2^{1/2} ‖x‖_L`; and `‖L‖_2 ≤ 2n` after scaling resistances into `[1,R]` (so every off-diagonal is in `[−1,−1/R]` and the row sums are at most `2n`). Hence `η ≤ 2n ε' √(E_r(f))`. To make `f̂` an honest `s-t` flow, I add a correction supported on a spanning tree `T`: the demands `Fχ(u) − i_ext(u)` sum to zero and total at most `nη` in positive part, so routing them on `T` changes each edge by at most `nη`, and I can find that tree flow in linear time. Call the result `f̃`, with `‖f̂ − f̃‖_∞ ≤ nη`.

Property (a). `E_r(f̃) = Σ_e r_e f̃_e² ≤ Σ_e r_e (f̂_e + nη)² = E_r(f̂) + 2nη Σ_e r_e f̂_e + n²η² Σ_e r_e`. Bound the cross term by Cauchy-Schwarz, `Σ_e r_e f̂_e ≤ (Σ_e r_e)^{1/2}(Σ_e r_e f̂_e²)^{1/2} ≤ (mR)^{1/2}(1+ε')√(E_r(f))`, and the last term by `Σ_e r_e ≤ mR`. With `η ≤ 2nε'√(E_r(f))` and `E_r(f) ≥ F²/m` (resistances in `[1,R]` force `F²/m ≤ E_r(f) ≤ F²Rm`), this gives `E_r(f̃) ≤ E_r(f)((1+ε')² + ε'·6n⁴ m R^{3/2})`. So choosing `ε' = δ/(12 n⁴ m R^{3/2})` makes `E_r(f̃) ≤ (1+δ) E_r(f)`. That is property (a).

Property (b), per-edge. From `‖φ̂ − φ‖_L² ≤ ε'² E_r(f)` and `‖φ̂ − φ‖_L² = Σ_e r_e (f_e − f̂_e)²` (the Laplacian norm of a potential difference is the energy of the corresponding flow difference), each edge has `r_e (f_e − f̂_e)² ≤ ε'² E_r(f)`. Then `|r_e f_e² − r_e f̂_e²| = √(r_e(f_e−f̂_e)²) · √(r_e(f_e+f̂_e)²) ≤ ε'√(E_r(f)) · (2+ε')√(E_r(f)) = ε'(2+ε') E_r(f)`. Adding the tree-correction perturbation (a similar `√R · nη · ...` term) and plugging `ε' = δ/(12n⁴ m R^{3/2})`, the total is `|r_e f_e² − r_e f̃_e²| ≤ δ/(2mR) · E_r(f)`. That is exactly the per-edge transfer I used to push the energy-fraction argument from `f̃` to the exact `f`.

Property (c), the potential drop. I want `φ̂_s − φ̂_t` close to `F R_eff` (this is what the cut algorithm reads off). Use the `L`-norm Pythagorean expansion. From `‖φ − φ̂‖_L² ≤ ε'² ‖φ‖_L²`, expand `‖φ − φ̂‖_L² = ‖φ‖_L² + ‖φ̂‖_L² − 2 φ̂ᵀ L φ`. But `L φ = Fχ`, so `φ̂ᵀ L φ = F χᵀ φ̂ = F(φ̂_s − φ̂_t)`. Therefore `2F(φ̂_s − φ̂_t) = ‖φ‖_L² + ‖φ̂‖_L² − ‖φ−φ̂‖_L² ≥ ‖φ‖_L² + (1−ε')²‖φ‖_L² − ε'²‖φ‖_L² = (2 − 2ε')‖φ‖_L² = (2−2ε')F² R_eff`, using `‖φ̂‖_L ≥ (1−ε')‖φ‖_L` and `‖φ‖_L² = E_r(f) = F² R_eff`. So `φ̂_s − φ̂_t ≥ (1−ε') F R_eff`. That is property (c).

Good — the solver delivers a `δ`-approximate electrical flow with all three properties in `Õ(m log R/δ) = Õ(m log(1/δ))` time. The whole flow algorithm is grounded.

Now the dual: a cut algorithm that is *simpler* — no oracle abstraction, no averaging, no explicit forbidden set. The idea: instead of reading a *flow* off the electrical solve, read a *cut* off the potentials. Repeatedly compute an electrical flow, raise the resistances of edges by how much flows over them, and at the end (or each step) split the vertices by a threshold on their potentials.

Why a threshold on potentials gives a cut, and a good one. Scale the potentials so `φ_s = 1, φ_t = 0`. For `x ∈ [0,1]`, let `S_x = {v : φ_v > x}`; this is an `s-t` cut. If I pick `x` uniformly at random, the probability an edge `(u,v)` is cut is exactly `|φ_u − φ_v|` (the fraction of the `[0,1]` interval between the two endpoint potentials). So the expected capacity of the random cut is `Σ_{(u,v)} |φ_u − φ_v| u_{(u,v)}`, and therefore *some* threshold gives a cut of capacity at most that sum. So I want to make `Σ_e |φ_u − φ_v| u_e` small.

Bound that sum by effective resistance via Cauchy-Schwarz. Let `μ = Σ_e u_e² r_e` (the "total resistance," capacity-weighted). With `φ` scaled to drop one between `s, t`, the energy identity gives `Σ_e (φ_u−φ_v)²/r_e = 1/R_eff` (the rescaled potentials correspond to the flow of value `1/R_eff`). Then

`Σ_e |φ_u−φ_v| u_e = Σ_e (|φ_u−φ_v|/√r_e)(u_e √r_e) ≤ √(Σ_e (φ_u−φ_v)²/r_e) · √(Σ_e u_e² r_e) = √((1/R_eff) · μ) = √(μ/R_eff)`.

So a cut of capacity `≤ √(μ/R_eff)` exists. If I can drive `R_eff` up to about `μ/F²`, this becomes `≈ F`, a near-minimum cut (recall the min cut equals `F*`). And there is a clean ceiling on how high `R_eff` can possibly go for a fixed budget `μ` on a cut of size `F`: put all the resistance on the `F`-capacity cut, `r_e = μ/(u_e F)`-style, and `R_eff` maxes out at `μ/F²`. So the target `R_eff ≥ (1−7ε) μ/F²` is exactly "the resistance has concentrated onto the minimum cut."

The algorithm: `w_e = 1`, `ρ = 3 m^{1/3} ε^{-2/3}`, `N = 5 ε^{-8/3} m^{1/3} ln m`, `δ = ε²`. Each step: compute the approximate electrical flow and potentials with `r_e = w_e/u_e²`; update the weights; rescale `φ` to `[0,1]`; sweep the threshold and return the best cut if its capacity drops below `F/(1−7ε)`. The update rule is *modified* from the flow algorithm:

`w_e^{i} = w_e^{i−1} + (ε/ρ) cong(f̃,e) w_e^{i−1} + (ε²/(mρ)) μ^{i−1}`.

The first two terms are the familiar multiplicative bump. The third, additive, term `(ε²/(mρ)) μ` is new and it earns its place: it keeps *every* weight at least an `(ε/m)` fraction of the total, `w_e^i ≥ (ε/m) μ^i`. Why I need that floor: when I cut/heavily-reweight an edge to push up `R_eff`, the jump lemma's `β` (the energy fraction) needs the edge to carry meaningful *absolute* weight; the plain multiplicative rule could leave a tiny-weight edge whose `(1+ε)` bump barely moves `R_eff`. Let me confirm the floor by induction: at `i=0`, `w_e^0 = 1 ≥ ε = (ε/m)·m = (ε/m)μ^0`. Inductively, `w_e^{i+1} ≥ w_e^i + (ε²/(mρ))μ^i ≥ (ε/m)μ^i + (ε²/(mρ))μ^i = (ε/m)(1 + ε/ρ)μ^i ≥ (ε/m) exp(ε(1−2ε)/ρ) μ^i ≥ (ε/m) μ^{i+1}`, where the last step uses the total-weight growth bound `μ^{i+1} ≤ μ^i exp(ε(1−2ε)/ρ)` (proved next). So the floor holds.

Suppose `R_eff(r^i) ≤ (1−7ε) μ^i/F²` for *all* `i ≤ N`, so the target is never reached. Then the number of "good" steps (no edge over `ρ`) plus the number of "bad" steps (some edge over `ρ`) is `N`, and I will bound each so tightly that their sum comes out below `N` — which is impossible, so the supposition must break.

Three pieces. First, the total weight does not grow too fast. If `R_eff(r^i) ≤ (1−7ε)μ^i/F²`, then the electrical flow `f` of value `F` has energy `F² R_eff ≤ (1−7ε)μ^i`, i.e. `Σ_e cong(f,e)² w_e^i ≤ (1−7ε)μ^i`. The approximate flow has energy `≤ (1+δ)` times that `≤ (1−6ε)μ^i` (since `δ=ε²` is tiny). By Cauchy-Schwarz, `Σ_e cong(f̃,e) w_e^i ≤ √(Σ w_e^i)√(Σ cong² w_e^i) ≤ √(1−6ε) μ^i ≤ (1−3ε)μ^i`. Then

`μ^{i+1} = Σ_e w_e^{i+1} = μ^i + (ε/ρ) Σ_e cong(f̃,e) w_e^i + (ε²/ρ) μ^i ≤ μ^i (1 + ε(1−3ε)/ρ + ε²/ρ) = μ^i(1 + ε(1−2ε)/ρ) ≤ μ^i exp(ε(1−2ε)/ρ)`.

So `μ^N ≤ m exp(ε(1−2ε)N/ρ)`.

Second piece: track the weight concentrated on a minimum cut `C` (capacity `u_C = F*`). The natural quantity is the capacity-weighted geometric mean of the weights of `C`'s edges, `ν^i = (Π_{e∈C} (w_e^i)^{u_e})^{1/u_C}`. Note `ν^i ≤ max_{e∈C} w_e^i ≤ μ^i`. In any step where *no* edge exceeds congestion `ρ`, `ν` jumps: using `(1+εx) ≥ exp(ε(1−ε)x)` for `x = cong/ρ ∈ [0,1]`, and the fact that `f̃` of value `F` must push at least `F` total across the cut `C` (so `Σ_{e∈C} |f̃_e| ≥ F`),

`ν^{i+1} ≥ ν^i (Π_{e∈C}(1 + (ε/ρ)cong(f̃,e))^{u_e})^{1/u_C} ≥ ν^i exp((1/u_C) Σ_{e∈C} u_e (ε(1−ε)/ρ) cong(f̃,e)) = ν^i exp((ε(1−ε)/(ρ u_C)) Σ_{e∈C}|f̃_e|) ≥ ν^i exp(ε(1−ε)/ρ)`.

(Using `u_e cong(f̃,e) = |f̃_e|` and `Σ_{e∈C}|f̃_e| ≥ F = u_C` for `F ≥ F*`.) So `ν` multiplies by `≥ exp(ε(1−ε)/ρ)` in every "no-edge-over-`ρ`" step.

Third piece: when some edge *does* exceed `ρ`, the effective resistance jumps. Such an edge `e` has `cong(f̃,e) > ρ`, so its energy in the approximate flow is `f̃_e² r_e = cong(f̃,e)² w_e ≥ ρ² (ε/m)μ^i` (using the weight floor `w_e ≥ (ε/m)μ^i`). The true electrical flow's total energy is `F² R_eff ≤ (1−7ε)μ^i`, so as a fraction of it, `e` contributes at least `(ρ²ε/m)/(1−7ε)`. Subtract the solver's per-edge slack `ε²/(2mR)` to transfer this to the *true* flow; with `ρ² = 9 m^{2/3}ε^{-4/3}` the leading term is `(ρ²ε/m)/(1−7ε) = 9 ε^{-1/3}m^{-1/3}/(1−7ε)`, comfortably more than the slack, so the true flow's fraction is at least `β ≥ ρ²ε/m`. The weight of `e` rose by at least `(1+ε)` this step (its multiplicative bump alone is `1 + (ε/ρ)cong > 1+ε` since `cong > ρ`). By the jump lemma with `γ = 1+ε` and `β ≥ ρ²ε/m`, `R_eff(r^{i+1})/R_eff(r^i) ≥ 1 + ε·β/2 = 1 + ρ²ε²/(2m) ≥ exp(ρ²ε²/(4m))`. So `R_eff` multiplies by `≥ exp(ε²ρ²/(4m))` in every "some-edge-over-`ρ`" step.

Now combine. Let `a` = number of steps with no edge over `ρ`, `b` = number with some edge over `ρ`, `a + b = N`. From `ν` (which only grows, and grows by `exp(ε(1−ε)/ρ)` on the `a` good steps) and `ν^N ≤ μ^N`:

`exp(ε(1−ε)a/ρ) ≤ ν^N ≤ μ^N ≤ m exp(ε(1−2ε)N/ρ)`,

so `a ≤ ((1−2ε)/(1−ε))N + (ρ/(ε(1−ε)))ln m ≤ (1−ε)N + (7ρ/(6ε))ln m`. From `R_eff` (grows by `exp(ε²ρ²/(4m))` on the `b` bad steps, floor `R_eff^0 ≥ 1/(m²F*²)`):

`exp(ε²ρ² b/(4m)) /(m² F*²) ≤ R_eff^N ≤ (1−7ε)μ^N/F²`,

and after taking logs and plugging the `μ^N` bound, `b ≤ (4m/(ερ³))N + (12m/(ε²ρ²))ln m`. Add them and plug in `ρ = 3 m^{1/3} ε^{-2/3}` (so `ρ³ = 27 m ε^{-2}`, `4m/(ερ³) = 4ε/27`) and `N = 5 ε^{-8/3} m^{1/3} ln m`:

`a + b < ((1−ε) + 4ε/27)·5ε^{-8/3}m^{1/3}ln m + (7m^{1/3}/(2ε^{5/3}) + 12m^{1/3}/(9ε^{2/3}))ln m = 5 ε^{-8/3} m^{1/3} ln m − (41/54 − 12ε/9) ε^{-5/3} m^{1/3} ln m < 5 ε^{-8/3} m^{1/3} ln m = N`,

for `ε ≤ 1/7`. So `a + b < N`, contradicting `a + b = N`. Therefore the assumption fails: within `N` steps the algorithm produces `r^i` with `R_eff(r^i) ≥ (1−7ε)μ^i/F²`, and then the threshold sweep yields a cut of capacity at most `(1+2δ)/√(1−7ε) · F ≤ F/(1−7ε)`. The cut algorithm runs in `Õ(m^{4/3}ε^{-8/3})` time (the `N` solves at `Õ(m)` each), and on a sparsifier with `O(n log n/ε²)` edges it gives a `(1+ε)`-cut in `Õ(m + n^{4/3}ε^{-8/3})`.

Let me trace the whole causal chain once more. The flow-decomposition barrier blocks every path-augmentation method, so I switch to a primitive that produces a whole flow vector in one shot — the electrical (minimum-energy, `ℓ_2`) flow, which is a single nearly-linear-time Laplacian solve. That flow ignores capacities and can overload an edge by `√m` (the `ℓ_2`-vs-`ℓ_∞` gap, witnessed by the parallel-paths-plus-direct-edge graph). Multiplicative weights converts a capacity-oblivious oracle that only satisfies the *weighted-average* capacity constraint into a feasible flow, in iterations proportional to the oracle's *width*; choosing resistances `r_e = (w_e + ε‖w‖_1/3m)/u_e²` makes one electrical solve such an oracle, with width `√m` — yielding `m^{3/2}`. To beat it, I exploit that the overloaded edges are fragile: remove any edge that exceeds a smaller width `ρ`, recompute, and bound the removals by tracking effective resistance, which is monotone (Rayleigh) and jumps multiplicatively whenever a high-energy edge is cut (the resistance-increase lemma); balancing iterations `Õ(ρ)` against removals `Õ(m/ρ²)` gives `ρ ≈ m^{1/3}` and a running time of `Õ(m^{4/3}ε^{-3})`, improved to `Õ(mn^{1/3}ε^{-11/3})` by Karger smoothing. The dual reads a cut directly from the electrical potentials by sweeping a threshold, bounding the cut by `√(μ/R_eff)` and driving `R_eff` up onto the minimum cut, for `Õ(m + n^{4/3}ε^{-8/3})`.

Here is the algorithm as a single self-contained C++17 program, the oracle plus the multiplicative-weights loop with the electrical-flow primitive at its core. It reads `n m s t F eps` and then `m` lines `a b u` from stdin, and prints the feasibility-scaled flow value, the per-edge flow, and the maximum congestion (or `FAIL` when the oracle certifies `F > F*`). The Laplacian solve is done by a dense Cholesky factorization of the grounded system, standing in for the nearly-linear-time SDD solver the analysis assumes.

```cpp
// Electrical-flow + multiplicative-weights approximate maximum s-t flow.
// Reads from stdin:  n m s t F eps
//                    then m lines:  a b u   (undirected edge a--b, capacity u)
// Writes to stdout:  one line "FAIL" if F > F* (oracle certifies infeasibility),
//                    otherwise "value <V>" then m lines of per-edge flow f(e),
//                    then "maxcong <c>".  (0-indexed vertices.)
//
// The electrical flow of value F minimizes the energy sum_e r_e f(e)^2 over s-t
// flows with B f = F*chi; it is the potential flow f = C B^T phi where
// L phi = F*chi, L = B C B^T the weighted Laplacian.  L is SDD; here we ground
// one vertex and solve the dense reduced system directly (the algorithm's
// intended regime replaces this with a nearly-linear-time SDD solver).  The
// multiplicative-weights outer loop turns this capacity-oblivious oracle into a
// feasible flow, reweighting by congestion each round.
//
// long long is unused for the numeric core (flows are real), but capacities and
// counts are read as long long to avoid overflow on large inputs.

#include <bits/stdc++.h>
using namespace std;

struct Edge { int a, b; double u; };

// Solve the symmetric positive-definite reduced Laplacian system A x = rhs by
// Cholesky factorization (A is the Laplacian with the grounded vertex removed).
// A is given as a dense (k x k) row-major matrix; solves in place.
static vector<double> cholesky_solve(vector<vector<double>>& A, vector<double> rhs) {
    int k = (int)A.size();
    // Cholesky: A = L L^T (store L in lower triangle of A).
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int p = 0; p < j; ++p) sum -= A[i][p] * A[j][p];
            if (i == j) {
                if (sum <= 0) sum = 1e-12;        // guard tiny/round-off pivots
                A[i][j] = sqrt(sum);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
    }
    // Forward solve L y = rhs.
    vector<double> y(k);
    for (int i = 0; i < k; ++i) {
        double sum = rhs[i];
        for (int p = 0; p < i; ++p) sum -= A[i][p] * y[p];
        y[i] = sum / A[i][i];
    }
    // Back solve L^T x = y.
    vector<double> x(k);
    for (int i = k - 1; i >= 0; --i) {
        double sum = y[i];
        for (int p = i + 1; p < k; ++p) sum -= A[p][i] * x[p];
        x[i] = sum / A[i][i];
    }
    return x;
}

// Electrical s-t flow of value F with the given per-edge conductances.
// Returns the flow vector f (length m); potentials phi returned via out-param.
static vector<double> electrical_flow(int n, const vector<Edge>& edges,
                                      const vector<double>& conduct,
                                      int s, int t, double F,
                                      vector<double>& phi_out) {
    int m = (int)edges.size();
    // Build dense Laplacian L = B C B^T (n x n).
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int e = 0; e < m; ++e) {
        if (conduct[e] == 0.0) continue;
        int a = edges[e].a, b = edges[e].b;
        double c = conduct[e];
        L[a][a] += c; L[b][b] += c;
        L[a][b] -= c; L[b][a] -= c;
    }
    // Ground vertex 0: solve on indices 1..n-1.
    int k = n - 1;
    vector<vector<double>> A(k, vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            A[i][j] = L[i + 1][j + 1];
    vector<double> rhs(k, 0.0);
    // chi: +1 at s, -1 at t (scaled by F); drop the grounded row 0.
    auto add_chi = [&](int v, double val) {
        if (v == 0) return;        // grounded row removed
        rhs[v - 1] += val;
    };
    add_chi(s, F);
    add_chi(t, -F);

    vector<double> xk = cholesky_solve(A, rhs);
    vector<double> phi(n, 0.0);
    for (int i = 0; i < k; ++i) phi[i + 1] = xk[i];

    // Ohm's law: f = C B^T phi, with B^T phi on edge (a,b) = phi[a]-phi[b].
    vector<double> f(m, 0.0);
    for (int e = 0; e < m; ++e) {
        if (conduct[e] == 0.0) { f[e] = 0.0; continue; }
        f[e] = conduct[e] * (phi[edges[e].a] - phi[edges[e].b]);
    }
    phi_out = phi;
    return f;
}

static double output_number(double x) {
    double y = round(x * 1000000.0) / 1000000.0;
    if (fabs(y) < 0.5e-6) return 0.0;
    return y;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, s, t;
    long long m_ll;
    double F, eps;
    if (!(cin >> n >> m_ll >> s >> t >> F >> eps)) return 0;
    int m = (int)m_ll;

    vector<Edge> edges(m);
    vector<double> u(m);
    for (int e = 0; e < m; ++e) {
        cin >> edges[e].a >> edges[e].b >> edges[e].u;
        u[e] = edges[e].u;
    }

    // Multiplicative-weights outer loop (the plain (eps, 3 sqrt(m/eps)) oracle).
    double rho = 3.0 * sqrt((double)m / eps);             // width of the plain oracle
    long long N = (long long)ceil(2.0 * rho * log((double)max(m, 2)) / (eps * eps));

    vector<double> w(m, 1.0);
    vector<double> acc(m, 0.0);
    bool failed = false;

    for (long long it = 0; it < N; ++it) {
        double w1 = 0.0;
        for (int e = 0; e < m; ++e) w1 += w[e];

        // r_e = (1/u_e^2)(w_e + eps*|w|_1/(3m)): w_e term for the average,
        // floor term eps*|w|_1/(3m) caps the worst congestion.
        vector<double> conduct(m, 0.0);
        vector<double> res(m, 0.0);
        double floor_term = eps * w1 / (3.0 * m);
        for (int e = 0; e < m; ++e) {
            res[e] = (w[e] + floor_term) / (u[e] * u[e]);
            conduct[e] = 1.0 / res[e];
        }

        vector<double> phi;
        vector<double> f = electrical_flow(n, edges, conduct, s, t, F, phi);

        // Energy E_r(f); fail-test certifies F > F* when energy too large.
        double E = 0.0;
        for (int e = 0; e < m; ++e) E += res[e] * f[e] * f[e];
        if (E > (1.0 + eps) * w1) { failed = true; break; }

        // Reweight by congestion; accumulate the per-round flow.
        for (int e = 0; e < m; ++e) {
            double cong = fabs(f[e]) / u[e];
            w[e] *= (1.0 + (eps / rho) * cong);
            acc[e] += f[e];
        }
    }

    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    if (failed) {
        cout << "FAIL\n";
        return 0;
    }

    // Feasibility-scaled average of the per-round flows.
    double scale = (1.0 - eps) * (1.0 - eps) / ((1.0 + eps) * (double)N);
    vector<double> fbar(m);
    for (int e = 0; e < m; ++e) fbar[e] = scale * acc[e];
    vector<double> printed_flow(m);
    for (int e = 0; e < m; ++e) printed_flow[e] = output_number(fbar[e]);

    // Flow value = net flow out of s in the printed certificate.
    double value = 0.0;
    for (int e = 0; e < m; ++e) {
        if (edges[e].a == s) value += printed_flow[e];
        else if (edges[e].b == s) value -= printed_flow[e];
    }

    double maxcong = 0.0;
    for (int e = 0; e < m; ++e) maxcong = max(maxcong, fabs(printed_flow[e]) / u[e]);

    cout << "value " << output_number(fabs(value)) << "\n";
    for (int e = 0; e < m; ++e) cout << printed_flow[e] << "\n";
    cout << "maxcong " << output_number(maxcong) << "\n";
    return 0;
}
```

One last sanity pass: does the implemented loop actually produce a feasible flow, and does the final scaling do what the analysis claims? The smallest input that exercises the averaging is two parallel unit-capacity edges between `s` and `t`, `F = 2`, `ε = 0.2` — stdin `2 2 0 1 2 0.2` then `0 1 1` twice. By symmetry the true answer is `1` unit per edge, congestion `1`. The program prints `value 1.066667`, `0.533333` on each edge, and `maxcong 0.533333` — max congestion `0.533333 < 1`, so it is feasible, and the value is a `0.53` fraction of `F`. That fraction is exactly `(1−ε)²/(1+ε) = 0.8²/1.2 = 0.64/1.2 = 0.5333`, the feasibility scaling I derived, with no surprise hiding in the iteration loop. The gap from the optimal `1` per edge is entirely the conservative scaling, not infeasibility — which is the right behaviour, since the analysis only ever promised value `(1−O(ε))F` and feasibility, and here I see precisely those two things and nothing worse. Pushing `F` past `F*` (e.g. `2 1 0 1 10 0.2` then `0 1 1`) makes the energy fail-test trip and the program print `FAIL`, certifying `F > F*`. It does not, of course, exhibit the `m^{1/3}` width reduction (this plain oracle has width `√m`); for that I would need to run the improved oracle on the parallel-paths-plus-direct-edge family and watch the forbidden set `H` stay at `Õ(m^{1/3})`, which the removal lemma predicts but I have not traced end-to-end here.

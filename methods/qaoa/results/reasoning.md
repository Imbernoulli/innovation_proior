Let me start from what actually hurts. I have a combinatorial optimization instance: `n` bits, `m` clauses, each clause `C_α` a local constraint on a few of the bits, and an objective `C(z) = Σ_α C_α(z)` that just counts how many clauses the string `z` satisfies. MaxSat wants the maximizing string; I'll settle for approximate — a `z` whose `C(z)` is close to the max. This is NP-hard in general, so I'm not going to find the exact optimum in polynomial time on any computer I know how to build. What I want is something more modest and more honest: a quantum procedure with a knob I can turn to spend more quantum effort and provably get a better answer, that compiles to gates no more complicated than the clauses themselves, and whose run parameters I can figure out cheaply offline.

The one quantum optimization idea that's really on the table is adiabatic evolution, so let me think hard about why it works and exactly where it breaks, because the break is going to tell me what to build. The premise is clean. Promote each bit `z_i` to a qubit. The objective `C` is diagonal in the computational basis: `C|z⟩ = C(z)|z⟩`. So the best string is literally the top eigenstate of `C` (or the ground state of `−C`, same thing) — optimization has become "find an extremal eigenstate of a diagonal operator." If I could prepare that eigenstate and measure, I'd read off the answer. I can't prepare it directly — that's the whole difficulty — but adiabatic evolution gives a path to it. Take a beginning Hamiltonian whose ground state is trivial. The standard one is the transverse field `H_B = Σ_i ½(1 − σˣ_i)`. Its ground state is the uniform superposition `|s⟩ = 2^{−n/2} Σ_z |z⟩ = |+⟩_1…|+⟩_n`, because `|+⟩` is the `+1` eigenstate of `σˣ` and these terms don't fight each other. A layer of Hadamards on `|0…0⟩` and I have it. Now interpolate: `H(t) = (1−t/T)H_B + (t/T)H_P`, drag `H_B` slowly into the problem Hamiltonian `H_P`. Start in `H_B`'s ground state and evolve under Schrödinger's equation `i d|ψ⟩/dt = H(t)|ψ⟩` for total time `T`.

Why does this land on the answer? The adiabatic theorem. Write `H̃(s) = (1−s)H_B + s H_P` with `s = t/T`, with instantaneous eigenvalues `E_0(s) ≤ E_1(s) ≤ …` and eigenstates `|ℓ;s⟩`. If the gap `E_1(s) − E_0(s)` stays strictly positive all the way along, then `lim_{T→∞} |⟨0;s=1 | ψ(T)⟩| = 1` — start in the ground state, end in the ground state, which is the answer. And the theorem is quantitative about how slow "slow" has to be: it suffices that

```
T ≫ ℰ / g_min²,   g_min = min_s (E_1(s) − E_0(s)),   ℰ = max_s |⟨1;s| dH̃/ds |0;s⟩|.
```

`ℰ` is roughly the size of one eigenvalue, nothing pathological. So the runtime is set, up to that factor, by `1/g_min²` — the inverse square of the smallest gap I pass through. That's the load-bearing fact, and it's where the pain lives. For hard instances `g_min` can be exponentially small in `n`, so the guaranteed-success `T` is exponential. And here's the part that really kills it for any near-term device: this `T` is the duration of a *single continuous coherent analog evolution*. I'd have to keep `n` qubits coherent, undergoing one long smooth Hamiltonian evolution, for a time that's both huge and instance-dependent. Real hardware decoheres; it has a gate budget and a coherence window, and it wants a *shallow, discrete* circuit, not one monolithic deep analog run. So "just take `T` large" is physically out of reach.

It's worse than out of reach, it's not even reliable. The success probability of adiabatic evolution isn't monotonic in `T`. For a fixed instance it can climb with runtime and then drop sharply, and the eventual large-`T` recovery sits at times I can neither simulate nor run — Crosson, Farhi, Lin, Lin and Shor exhibit exactly this on a 20-qubit Max2Sat instance, the success probability rising then falling off a cliff. And there are objectives where it's stuck for any subexponential time: take an objective that depends only on the Hamming weight, symmetric in the bits; the evolution stays trapped in a false minimum at weight `w = n` while the real optimum sits at `w = 0` (that's the symmetric example Farhi, Goldstone and Gutmann analyzed against simulated annealing). So I cannot lean on "run it longer."

Let me hold onto what's *good* here, though, because I don't want to throw out the encoding. The encoding is right: optimum = extremal eigenstate of a diagonal `C`, reachable in principle by interpolating from an easy `H_B`. What's wrong is the *implementation* — a long continuous coherent evolution whose length I can't control. So the question becomes: can I approximate that continuous evolution with a *short, discrete, gate-model* circuit, and turn the uncontrollable analog runtime into a controllable discrete depth?

Trotterization is exactly the tool for turning a continuous evolution under a sum of terms into alternating discrete steps. The product formula: for non-commuting `A` and `B`,

```
e^{−i(A+B)t} = (e^{−iA t/N} e^{−iB t/N})^N + O(t²/N).
```

My evolving Hamiltonian is, up to the time-dependent coefficients, made of two pieces: the cost operator `C` and the mixer `B = Σ_j σˣ_j` (the `σˣ` part of `H_B`; the constant and the ½ just shift and scale, I can fold them in). So a Trotterized approximation to the adiabatic path is an alternation of `e^{−iγC}` and `e^{−iβB}` for a sequence of little time slices, where the `γ`'s and `β`'s are the slice durations and their running sum is the total adiabatic time. Let me look at each factor.

The cost factor `e^{−iγC}`. Because `C = Σ_α C_α` is diagonal, all the `C_α` commute, so

```
U(C, γ) = e^{−iγC} = ∏_α e^{−iγ C_α},
```

and this is *exact* — no Trotter error inside the cost layer. Even better, each factor `e^{−iγ C_α}` acts only on the bits clause `α` touches, so its locality is the clause's locality: a 2-bit clause gives a 2-qubit gate, a 3-bit clause a 3-qubit gate. That's precisely the "gates no harder than the objective" property I wanted, and it falls out for free. And since `C` has integer eigenvalues, `e^{−iγC}` is periodic in `γ` with period `2π`, so I can keep `γ ∈ [0, 2π]`.

The mixer factor `e^{−iβB} = ∏_j e^{−iβ σˣ_j}` is a product of single-qubit `σˣ`-rotations, one `RX(2β)` per qubit, depth one. `σˣ` has period `2π` in the rotation angle, so `β ∈ [0, π]` covers it. Why is `B = Σσˣ` the right driver, and not something else? Two reasons, and they're the reasons it was the beginning Hamiltonian in the first place. Its ground state is the trivial `|+⟩^n` I start from. And it *moves amplitude between basis states*: a single-qubit `σˣ` flips one bit, so `B` connects every computational basis string to its single-bit-flip neighbors — the off-diagonal entries are non-negative, the operator is irreducible on the hypercube. By Perron–Frobenius that non-negative off-diagonal structure forces a non-degenerate extremal eigenvalue with a gap below it, which is exactly the gap-positivity the adiabatic theorem demands for the extremal-state path. Without something like `B` mixing the basis states, the cost layer alone would just rephase a diagonal state and never explore — `e^{−iγC}` is diagonal, it can't move probability between strings. So I need both: `C` to imprint the objective as phases, `B` to convert those phases into amplitude that flows toward good strings.

So the Trotterized state, for `p` slices, is

```
|γ, β⟩ = U(B, β_p) U(C, γ_p) ⋯ U(B, β_1) U(C, γ_1) |s⟩,
```

`p` cost layers and `p` mixer layers, applied to `|s⟩ = |+⟩^n`. Without exploiting any structure this is a circuit of depth at most `mp + p`: each of the `p` cost layers is at most `m` clause-gates, each mixer layer is depth one.

Now I'm staring at the requirement that this faithfully approximate the adiabatic path, and it's pulling me in two contradictory directions. For Trotter to be accurate I want each `γ` and `β` *small* (small time slices, small `O(t²/N)` error). For the adiabatic theorem to succeed I want the total run time — the sum of the angles — *large*. Small steps that sum to something large means *many* steps: large `p`. So a faithful adiabatic approximation forces `p` to be big, maybe exponentially big, and I'm right back to a deep circuit. If I insist on tracking the adiabatic schedule, Trotterization bought me a discrete circuit but not a shallow one.

So stop insisting on it. Why am I locking the angles to the adiabatic schedule at all? The schedule prescribes specific small, monotone `γ`'s and `β`'s. But the only thing I actually care about is the final expectation of `C` in whatever state I produce. Let me cut the angles loose from the schedule and treat `γ_1…γ_p, β_1…β_p` — all `2p` of them — as *free parameters* I'm allowed to choose however I like. Define

```
F_p(γ, β) = ⟨γ, β| C |γ, β⟩,   M_p = max_{γ,β} F_p(γ, β).
```

This is the variational principle put to work. `⟨ψ|C|ψ⟩ ≤ C_max` for any state, so `M_p` can never exceed the true optimum — every value of `F_p` I achieve is a genuine lower bound on what I can claim, and pushing `F_p` up is real progress, not a heuristic I have to apologize for. And freeing the angles can only *help* relative to the adiabatic schedule, because the schedule's particular angle sequence is just one point in the `2p`-dimensional box I'm now optimizing over. A short circuit with cleverly chosen angles can do far better than a short circuit forced to take faithful little Trotter steps.

Two structural facts make this feel right. First, `M_p ≥ M_{p−1}`: a depth-`(p−1)` circuit is the special case of a depth-`p` circuit with the extra layer set to do nothing, so the depth-`p` maximization is over a strictly larger set — more layers never hurt. The knob I wanted is `p`, and turning it up provably can't lower the achievable objective. Second — and this is what reconnects me to the adiabatic story and proves the knob actually reaches the answer — `lim_{p→∞} M_p = max_z C(z)`. Here's why that limit holds: among all the angle choices available at large `p` are exactly the ones that *do* trace a faithful Trotterization of an adiabatic path — small angles summing to a long run time. For that subfamily, Trotterization plus the adiabatic theorem (with the Perron–Frobenius gap guaranteeing success along the extremal-state path) drives the state to the optimal eigenstate, so `F_p` at those angles approaches `max_z C(z)`. Since `M_p` is the *max* over all angles, it's at least that, and by the variational bound it's at most `max C`. So `M_p → max C`. The faithful-adiabatic regime I was worried about isn't gone — it's sitting inside my search space as the worst-case fallback, while the optimizer is free to find something far shallower that works just as well or better.

That last point deserves a sanity check, because I might be fooling myself that this is just the adiabatic algorithm in a discrete costume. It isn't, and I can see why on a tiny example. Take the ring — a connected 2-regular MaxCut graph, the "ring of disagrees." At `p = 1`, with the best single `(γ, β)`, the state gives a `3/4` approximation ratio. But that state has *exponentially small overlap with the optimal strings*. The adiabatic algorithm aims to build large overlap with the optimum; this is doing something different — producing a state whose *expectation* of `C` is high even though it's nowhere near a delta function on the best string. Measuring it still hands me good strings often enough, because the expectation is what governs the measured mean. So this is genuinely not "approximate the ground state," it's "make `⟨C⟩` large," and those come apart.

Now, the algorithm needs me to actually *choose* good angles, and the abstract picture doesn't tell me how. Let me make this concrete on MaxCut, and let me exploit locality to get the angles cheaply. For MaxCut, each edge `⟨jk⟩` contributes a clause that's satisfied when its endpoints differ, so

```
C = Σ_{⟨jk⟩} C_{⟨jk⟩},   C_{⟨jk⟩} = ½(1 − σᶻ_j σᶻ_k),
```

which is `1` when the two bits disagree and `0` when they agree — exactly the cut indicator. Now

```
F_p(γ, β) = Σ_{⟨jk⟩} ⟨s| U†(C,γ_1)⋯U†(B,β_p) C_{⟨jk⟩} U(B,β_p)⋯U(C,γ_1) |s⟩.
```

Look at one edge's operator `U†⋯ C_{⟨jk⟩} ⋯U`. I claim it only involves qubits within graph-distance `p` of the edge `⟨jk⟩`. Take `p = 1`. The mixer factors `e^{−iβ σˣ_ℓ}` for qubits `ℓ` not in `{j,k}` commute through `C_{⟨jk⟩}` (different qubits) and meet their daggers — they cancel. What survives is `e^{iβ(σˣ_j+σˣ_k)} C_{⟨jk⟩} e^{−iβ(σˣ_j+σˣ_k)}`, touching only `j` and `k`. Then the cost factors `e^{−iγ C_α}` for clauses `α` not involving `j` or `k` likewise commute through and cancel; only clauses sharing a vertex with the edge survive — that's the edge and its neighboring edges, qubits one step away. Iterate: at general `p`, only edges and qubits within `p` steps of `⟨jk⟩` can matter. So each edge's contribution depends only on its local neighborhood subgraph.

That's the lever for cheap angle-finding. Two edges whose neighborhood subgraphs are isomorphic contribute *the same function* of `(γ, β)`. So group edges by subgraph *type* `g`:

```
F_p(γ, β) = Σ_g w_g · f_g(γ, β),
```

where `f_g` is the contribution of one edge of type `g` and `w_g` counts how many edges have that type. The `f_g` are functions on a Hilbert space whose size is set by the number of qubits in the subgraph, which for maximum degree `v` and a tree neighborhood is

```
q_tree = 2[((v−1)^{p+1} − 1)/((v−1) − 1)]
```

(or `2p + 2` if `v = 2`) — independent of `n`. For fixed `p` and bounded degree there are finitely many subgraph types, each `f_g` lives in a Hilbert space of size at most `2^{q_tree}` that doesn't grow with `n`, and the *only* `n`-dependence in `F_p` is through the integer weights `w_g`, which I read straight off the graph by counting subgraphs. So I can evaluate `F_p(γ, β)` — and maximize it over the `2p` angles — on a classical computer whose resources don't grow with `n`. Find the best `(γ, β)` offline, then run the quantum computer once at those angles, measure, get a string. (For large `p` the classical bookkeeping can blow up — the space of subgraph types can grow badly — but it's `n`-independent, which is the point for fixed `p`.)

Let me actually do `p = 1` on 3-regular graphs and see what ratio drops out, because if this can't beat random guessing it's worthless. Each edge `⟨jk⟩` in a cubic graph has a `p=1` neighborhood that is one of three small subgraphs: a 4-vertex piece I'll call `g4` (the "crossed square"), a 5-vertex piece `g5`, or a 6-vertex tree `g6`. Let me compute the per-edge expectation for the generic triangle-free case, where each endpoint has its full degree 3, i.e. two *other* neighbors besides the partner across the edge.

I'll evaluate `⟨s|e^{iγC}e^{iβB} C_{⟨jk⟩} e^{−iβB}e^{−iγC}|s⟩` for one edge. Write `C_{⟨jk⟩} = ½(1 − Z_j Z_k)`; the constant `½` contributes `½` to the expectation, so I need `−½⟨Z_j Z_k⟩` in the evolved frame. Conjugating `Z_j Z_k` by the mixer `e^{−iβB}` rotates each `Z` in the `Y`–`Z` plane: `e^{iβσˣ} Z e^{−iβσˣ} = cos(2β) Z + sin(2β) Y` (since `σˣ` generates rotation by `2β` about the `x`-axis). So `Z_j Z_k` becomes `(cos2β Z_j + sin2β Y_j)(cos2β Z_k + sin2β Y_k)`. Now conjugate by the cost layer `e^{−iγC}`, which is diagonal and only acts nontrivially through the edges touching `j` and `k`. Working this out term by term, the cross-terms involving the partner edge and the neighboring edges combine into products of cosines of `γ` over the neighbor count. For a triangle-free edge with `d_j = d_k = 2` other neighbors each, the surviving expectation is

```
⟨C_{⟨jk⟩}⟩ = ½ + ¼ sin(4β) sin(γ)(cos^{d_j} γ + cos^{d_k} γ),   d_j = d_k = 2.
```

Let me read this. The `sin(4β)` factor is the mixer turning phase into population (it's the `cos2β·sin2β` cross term doubled); the `sin γ` is the cost layer's first imprint of the objective; the `cos^d γ` factors are the "leakage" through each of the `d` neighboring edges, which damp the signal because every extra neighboring edge rotates a little phase away. So more neighbors hurt at fixed angle — interference from the surrounding graph. Maximize over `(γ, β)`:

I push `sin(4β)` to its max of `1` at `β = π/8`, and numerically optimize `γ` in `¼·sin(γ)(2cos²γ)`. The maximum of the whole per-edge expectation comes out to `0.6924…` (around `γ ≈ 0.616`, `β ≈ 0.393` once I include the small corrections from the actual subgraph types). For a triangle-free 3-regular graph all edges share this value, so `F_1 / (number of edges) = 0.6924…`.

To turn a per-edge expectation into a guaranteed approximation *ratio* I need the denominator, the best possible cut, and that's not just a function of the subgraph counts. But I can bound it. Suppose the graph has `T` isolated triangles and `S` crossed squares. Every triangle forces at least one uncut edge (you can't 2-color a triangle with all edges cut), and every crossed square forces at least one uncut edge, and no triangle and square can share a vertex, so `3T + 4S ≤ n`. Hence the optimal cut is at most `(3n/2 − S − T)` out of the `3n/2` edges. Meanwhile the QAOA value at `p=1` is `M_1(n,S,T) = max_{γ,β} F_1`, with

```
F_1 = S·f_{g4} + (4S + 3T)·f_{g5} + (3n/2 − 5S − 3T)·f_{g6},
```

counting `S` edges of type `g4` (one per crossed square), `4S + 3T` of type `g5` (four per crossed square plus all three edges of each isolated triangle), and the remaining `3n/2 − 5S − 3T` of type `g6`. The ratio is at least

```
M_1(n,S,T) / (3n/2 − S − T).
```

Scale out `n`: `M_1/n` depends only on `s = S/n` and `t = T/n`, with `s,t ≥ 0` and `4s + 3t ≤ 1`, so the guaranteed ratio is `M_1(1,s,t)/(3/2 − s − t)`. Sweeping `s, t` numerically, the minimum is at `s = t = 0` — the triangle-free, square-free case — and the value there is `0.6924`. So on *any* 3-regular graph the `p=1` algorithm produces a cut at least `0.6924` times optimal. It's a clean provable quantum approximation guarantee (it doesn't beat the best classical cubic-MaxCut algorithms, but it's a real bound, and it's the floor, not the ceiling — `p` is a knob).

Let me also do the ring (2-regular) for all `p`, because it shows the knob converging. A connected 2-regular graph is a ring; for `p < n/2` the `p`-neighborhood of every edge is the same single type — a line segment of `2p + 2` qubits with the edge in the middle — so there's one `f_g` and weight `n`. Numerically maximizing it for `p = 1,…,6` gives `M_p/n = 3/4, 5/6, 7/8, 9/10, 11/12, 13/14`, i.e. `M_p = n(2p+1)/(2p+2)`. So the ratio is `(2p+1)/(2p+2) → 1` as `p` grows, *independent of `n`*, and I can break the cost sum into even and odd edges to get circuit depth `3p`, also independent of `n`. The knob delivers: arbitrarily good approximation at constant depth.

Before I trust the measured mean, I should check the spread, or the measured strings might scatter wildly around `F_p`. The variance is `⟨C²⟩ − ⟨C⟩²`. Expanding `C² = Σ_{⟨jk⟩,⟨j'k'⟩} C_{⟨jk⟩} C_{⟨j'k'⟩}`, the connected part of a pair of edges vanishes whenever their `p`-neighborhoods share no qubit — which happens unless there's a path of length `≤ 2p+1` between them. The number of edges within that range of a fixed edge is at most `2[((v−1)^{2p+2} − 1)/((v−1)−1)]` (the tree-size formula with `p → 2p+1`), an `n`-independent constant for fixed `v, p`. Each summand is bounded by `1`, so

```
⟨C²⟩ − ⟨C⟩² ≤ 2[((v−1)^{2p+2} − 1)/((v−1)−1)] · m.
```

Variance `O(m)`, so standard deviation `O(√m)`. The distribution of `C(z)` is concentrated near `F_p`: a sample mean of order `m²` values lands within `1` of `F_p` with probability `1 − 1/m`, and there's only a tiny chance of strings far above `F_p`, so I'm not secretly relying on rare lucky tails — the typical measured string is genuinely near `F_p`.

That's the whole method on MaxCut. Now let me stress-test it on a problem where I can push the analysis further and see what the `p=1` algorithm buys over random guessing as a function of degree — bounded-occurrence Max E3LIN2. Each constraint is a mod-2 linear equation on exactly three bits, `x_a + x_b + x_c = 0` or `1`, each bit in at most `D + 1` equations. Random guessing satisfies half the equations; the question is how far above `½` I can provably go. Writing the equation operator as `½(1 ± Z_a Z_b Z_c)` and dropping the additive `½`,

```
C = ½ Σ_{a<b<c} d_{abc} Z_a Z_b Z_c,   d_{abc} ∈ {0, ±1}.
```

The expected number satisfied is `m/2` plus `⟨−γ, β| C |−γ, β⟩` (I use `−γ` to keep signs tidy; this expectation is an odd function of `γ`, so at `γ=0` it's zero — pure guessing — and I need nonzero angles to beat it). I'll fix `β = π/4` purely because it collapses the algebra: a `π/4` rotation about `x` sends `Z` to `Y` exactly (`e^{iβσˣ}Z e^{−iβσˣ} = cos2β·Z + sin2β·Y`, and at `β=π/4` the `cos2β` term vanishes, leaving `Y`). This isn't claimed optimal — it's a provable choice that makes the integral doable.

Take one clause's term, say bits 1,2,3: `½ d_{123} ⟨s|e^{−iγC} e^{iβB} Z_1 Z_2 Z_3 e^{−iβB} e^{iγC}|s⟩`. All mixer factors except `X_1+X_2+X_3` commute through the three `Z`'s, and at `β=π/4` the `Z`'s turn into `Y`'s:

```
½ d_{123} ⟨s| e^{−iγC} Y_1 Y_2 Y_3 e^{iγC} |s⟩.
```

Now split the central clause out of `C`: `C = C̄ + ½ d_{123} Z_1 Z_2 Z_3`. Conjugating `Y_1 Y_2 Y_3` by the central clause's own contribution `e^{−iγ·½ d_{123} Z_1Z_2Z_3}` rotates it. Since `Z_a Z_b Z_c` anticommutes appropriately with `Y_1Y_2Y_3` on the shared qubits, the conjugation gives

```
½ d_{123} ⟨s| e^{−iγC̄} ( cos(γ d_{123}) Y_1 Y_2 Y_3 + sin(γ d_{123}) X_1 X_2 X_3 ) e^{iγC̄} |s⟩.
```

Handle the `X X X` piece first: `⟨s| e^{−iγC̄} X_1 X_2 X_3 e^{iγC̄} |s⟩`. Insert complete sets of states for qubits 1,2,3 on both sides; `X` is off-diagonal so it flips each of those three bits, leaving

```
Σ_{z_1,z_2,z_3} ⟨s| e^{−iγC̄} |z_1 z_2 z_3⟩ ⟨−z_1, −z_2, −z_3| e^{iγC̄} |s⟩.
```

Now look at `C̄` — all clauses touching bits 1,2,3 except the central one. Group its terms by which of bits 1,2,3 they carry:

```
C̄ = Z_1 C_1 + Z_2 C_2 + Z_3 C_3 + Z_1 Z_2 C_{12} + Z_1 Z_3 C_{13} + Z_2 Z_3 C_{23},
```

where `C_1 = ½ Σ_{a<b} d_{1ab} Z_a Z_b` runs over the *other* two bits of each clause containing bit 1 (and similarly `C_2, C_3`), and the `C_{12}`-type pieces carry a single extra `Z`. On the left of the matrix element bits 1,2,3 take values `z_i`; on the right they took `−z_i` after the `X`-flips. In the terms with a single `Z_i`, flipping `z_i → −z_i` flips the sign; the doubly-indexed `C_{ij}` terms get two sign flips and cancel between left and right. So only the `Z_i C_i` parts survive the difference, and the bracket collapses to

```
⅛ Σ_{z_1,z_2,z_3} ⟨s̄| e^{−2iγ(z_1 C_1 + z_2 C_2 + z_3 C_3)} |s̄⟩,
```

with `|s̄⟩ = ∏_{a∈Q} |+⟩_a` over the at-most-`6D` other bits appearing in `C_1,C_2,C_3` (each of bits 1,2,3 is in at most `D` other clauses, each pulling in two more bits). Summing `z_1,z_2,z_3 ∈ {±1}` gives four cosines:

```
¼ ⟨s̄| [ cos(2γ(C_1+C_2+C_3)) + cos(2γ(C_1−C_2−C_3)) + cos(2γ(−C_1+C_2−C_3)) + cos(2γ(−C_1−C_2+C_3)) ] |s̄⟩.
```

Evaluating `⟨s̄|·|s̄⟩` is an average over the `±1` values of the `z_a` (`a ∈ Q`), so writing `c_i(z) = Σ_{a<b} d_{iab} z_a z_b` this is `¼ E_z[ cos(γ(c_1+c_2+c_3)) + cos(γ(c_1−c_2−c_3)) + … ]`. Putting back the `sin(γ d_{123})` prefactor, the full `XXX` contribution is `⅛ d_{123} sin(γ d_{123}) E_z[Σ cos(γ(±c_1±c_2±c_3))]`. The `YYY` term goes through identically with `sin` in place of `cos`, giving `⅛ d_{123} cos(γ d_{123}) E_z[Σ sin(…)]`. Adding them (and using `sin(A)cos(B)+cos(A)sin(B)=sin(A+B)`), the whole single-clause expectation at `β=π/4` is

```
⅛ d_{123} E_z[ sin(γ(d_{123}+c_1+c_2+c_3)) + sin(γ(d_{123}+c_1−c_2−c_3)) + sin(γ(d_{123}−c_1+c_2−c_3)) + sin(γ(d_{123}−c_1−c_2+c_3)) ].
```

Now Taylor expand in `γ` and pull out the linear term. The coefficient of `γ` is `⅛ d_{123} · 4 · d_{123} = ½ d_{123}² = ½`, independent of the equation type — every clause contributes the *same* `½ γ` at leading order, which is the signal I'm trying to keep. Write it as

```
½ d_{123}² γ + P^k_{123}(γ) + R^k_{123}(γ),
```

`P^k` the cubic-and-up terms through order `k`, `R^k` the tail beyond `k`, with

```
|R^k_{123}(γ)| ≤ ⅛ · |γ|^{k+2}/(k+2)! · E_z[ |d_{123}+c_1+c_2+c_3|^{k+2} + … ].
```

I need to bound that high moment of `d ± c_1 ± c_2 ± c_3`, which is a degree-2 polynomial in the `±1` variables `z_a`. There's a hypercontractivity bound for exactly this — for any degree-2 polynomial `c` over the cube (Theorem 5 of Dinur, Friedgut, Kindler, O'Donnell on Fourier tails),

```
E_z[|c|^{k+2}] ≤ (k+1)^{k+2} (E_z[c²])^{(k+2)/2}.
```

And `E_z[c_i²] = Σ_{a<b} d_{iab}² ≤ D` by the bounded-occurrence assumption, while `E_z[c_i] = 0`, so `E_z[(d ± c_1 ± c_2 ± c_3)²] = 1 + E_z[(c_1 ± c_2 ± c_3)²] ≤ 1 + 9D` (Cauchy–Schwarz on the three terms). Plugging in,

```
E_z[|d ± c_1 ± c_2 ± c_3|^{k+2}] ≤ (k+1)^{k+2} (1 + 9D)^{(k+2)/2},
```

and so, using Stirling and `e < 3`,

```
|R^k_{123}(γ)| ≤ ½ (k+1)^{k+2}/(k+2)! · (1+9D)^{(k+2)/2} |γ|^{k+2} ≤ ½ (e(1+9D)^{1/2}|γ|)^{k+2} ≤ (9 D^{1/2} |γ|)^{k+2}.
```

Sum over all `m` clauses. The target `⟨−γ,π/4|C|−γ,π/4⟩` is `Σ_{a<b<c}[½ γ + P^k_{abc} + R^k_{abc}] = (m/2)γ + P^k(γ) + Σ R^k_{abc}`. By the triangle inequality its absolute value is at least `|(m/2)γ + P^k(γ)| − Σ|R^k_{abc}| ≥ |(m/2)γ + P^k(γ)| − m(9D^{1/2}|γ|)^{k+2}`. To keep the negative tail small I restrict `|γ| ≤ 1/(10 D^{1/2})`, which makes `(9D^{1/2}|γ|)^{k+2} ≤ (9/10)^{k+2}`. To lower-bound the *positive* part I need that the polynomial `(m/2)γ + P^k(γ)` — leading coefficient `m/2`, then cubic-and-up — can't be small everywhere on `[−1/(10D^{1/2}), 1/(10D^{1/2})]`. That's a Chebyshev-type fact (Corollary 2.7 of the same Dinur et al paper): for odd `k`, `max_{r=0..k} |x_r + a_2 x_r² + … + a_k x_r^k| ≥ 1/k` at the nodes `x_r = cos(πr/k)`, for *any* coefficients. Setting `γ_r = cos(πr/k)/(10D^{1/2})`,

```
max_r |(m/2)γ_r + P^k(γ_r)| ≥ m/(20 D^{1/2} k).
```

Combining,

```
max_r { |(m/2)γ_r + P^k(γ_r)| − m(9D^{1/2}|γ_r|)^{k+2} } ≥ m/(20 D^{1/2} k) − m(9/10)^{k+2}.
```

Choose `k = 5 ln D`: the negative `(9/10)^{k+2}` term decays like a power of `D` fast enough that the first term dominates for large `D`, and the right side exceeds `m/(101 D^{1/2} ln D)`. Since the expectation is odd in `γ`, one of the `γ_r` makes it positive, so there's a `γ` in the allowed range — found by checking only `5 ln D` values — with

```
⟨γ, π/4| C |γ, π/4⟩ ≥ m/(101 D^{1/2} ln D),
```

i.e. the `p=1` algorithm produces a string satisfying at least `(½ + 1/(101 D^{1/2} ln D)) m` equations. Beating random guessing by an explicit, certifiable margin, with only `O(ln D)` quantum runs to pin down `γ`.

I can say something sharper for the *typical* instance — fix the set of triples and choose each equation's `0/1` (i.e. each `d_{abc}`'s sign) uniformly at random. Average the single-clause expectation over the `d`'s. The `YYY` piece carried `cos(γ d_{123})` times terms not involving `d_{123}`, so its `d_{123}`-average is zero; the `XXX` piece carried `sin(γ d_{123})`, and with `d_{123} = ±1` equally likely, `½ d_{123} sin(γ d_{123}) = ½ sin γ`. The four-cosine bracket factorizes under the `d`-average: `E_d[cos(γ c_1)] = ∏_{a<b} cos(γ d_{1ab} z_a z_b) = ∏ cos γ = cos^{D_1} γ` where `D_1 ≤ D` is the number of terms in `c_1` (the `z_a z_b = ±1` and `d = ±1` just toggle the sign inside an even function). So the `d`-averaged single-clause contribution is `½ sin γ · cos^{D_1+D_2+D_3} γ`, and summing,

```
(m/2) sin γ · cos^{3D} γ ≤ E_d[⟨−γ,π/4|C|−γ,π/4⟩] ≤ (m/2) sin γ.
```

Maximize the lower bound. Set `γ = g/D^{1/2}`; for large `D`, `cos^{3D}(g/√D) → e^{−3g²/2}` and `sin γ ≈ g/√D`, so the lower bound is `(m/2)(g/√D) e^{−3g²/2}`, maximized at `g = 1/√3`, giving `m/(2√(3e) D^{1/2})`. So with `γ = 1/(√3 D^{1/2})` the typical instance gets `(½ + 1/(2√(3e) D^{1/2})) m` equations. And the variance is controlled: the squared expectation is a double sum over clauses, and a clause pair contributes only if they share one of the random `d`'s, which happens for at most `m(6D+3)(D+1)` pairs (each clause is "linked" through its `≤ 6D+3` bits, each in `≤ D+1` clauses), each contributing `≤ ¼`, so the variance is `≤ ¼ m(6D+3)(D+1) = O(m)` at fixed `D`. Mean `O(m)`, standard deviation `O(√m)`, so the typical instance really does hit the `½ + 1/(2√(3e) D^{1/2})` rate with high probability. The `D^{1/2}` scaling is the interesting part: it matches the inapproximability threshold below which doing better would imply P=NP.

So the recipe is settled. Encode the objective as a diagonal `C`; build a `p`-layer alternating circuit `U(B,β_p)U(C,γ_p)…U(B,β_1)U(C,γ_1)` on `|+⟩^n`, where `U(C,γ)=∏_α e^{−iγC_α}` imprints the objective as phases (locality = clause locality) and `U(B,β)=∏_j e^{−iβσˣ_j}` mixes amplitude; treat the `2p` angles as free variational parameters; maximize `F_p(γ,β)=⟨γ,β|C|γ,β⟩` — classically for fixed `p` and bounded degree, or by a classical optimizer querying the quantum computer for `F_p`; then run at the best angles, measure in the computational basis, and keep the best string. Turning `p` up provably can't lower the achievable objective and recovers the optimum as `p→∞`.

Let me land it as real code on MaxCut. Build the cost Hamiltonian as a sum of `ZZ` edge terms and the mixer as a sum of single-qubit `X`'s; each variational layer is `e^{−iγ H_C}` followed by `e^{−iβ H_B}`; the objective is the (negated) expectation of `H_C`; optimize the `2p` angles by gradient descent; then sample.

```python
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

# --- the instance ---
n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
wires = range(n_wires)

# --- cost layer: U(C, gamma) = e^{-i gamma C}, C = sum_edges 1/2(1 - Z_j Z_k) ---
# the constant 1/2 is a global phase; e^{-i gamma/2 (-Z_jZ_k)} on each edge is CNOT-RZ-CNOT
def U_C(gamma):
    for j, k in graph:
        qml.CNOT(wires=[j, k])
        qml.RZ(gamma, wires=k)         # phases strings by their cut value
        qml.CNOT(wires=[j, k])

# --- mixer layer: U(B, beta) = prod_j e^{-i beta X_j}, one RX per qubit ---
def U_B(beta):
    for w in wires:
        qml.RX(2 * beta, wires=w)       # the transverse-field driver that moves amplitude

dev = qml.device("lightning.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    for w in wires:
        qml.Hadamard(wires=w)           # |s> = |+>^n, ground state of the mixer
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)                       # p alternating layers
        U_B(beta)
    if return_samples:
        return qml.sample()              # read a candidate string in the computational basis
    # F_p = <C>; here <sum_edges Z_j Z_k>, turned into the cut count below
    C = qml.sum(*(qml.Z(j) @ qml.Z(k) for j, k in graph))
    return qml.expval(C)

def objective(params):
    gammas, betas = params[0], params[1]
    # cut value = 1/2 (|E| - <sum Z_jZ_k>); maximize it => minimize the negative
    return -0.5 * (len(graph) - circuit(gammas, betas))

def qaoa_maxcut(p=1, steps=30):
    params = 0.01 * np.random.rand(2, p, requires_grad=True)   # the 2p free angles
    opt = qml.GradientDescentOptimizer(stepsize=0.5)           # classical optimizer of F_p
    for _ in range(steps):
        params = opt.step(objective, params)
    # run once at the best angles, sample many shots, keep the most frequent string
    return params
```

To recap the chain: optimization is finding the extremal eigenstate of a diagonal `C`; adiabatic evolution from the easy `H_B` ground state reaches it but needs runtime `∝ 1/g_min²` — exponential and one long coherent analog run, too deep and not even monotone in `T`; Trotterizing the path into alternating `e^{−iγC}` and `e^{−iβB}` discretizes it but a *faithful* adiabatic approximation still forces large `p`; so cut the angles free of the schedule and make all `2p` of them variational, bounded honestly by `⟨C⟩ ≤ C_max` — more layers never hurt (`M_p ≥ M_{p−1}`) and `p→∞` recovers the adiabatic optimum, while small `p` with optimized angles already gives a provable `0.6924` for cubic MaxCut and a `½ + 1/(2√(3e)D^{1/2})` typical edge over guessing on E3LIN2; locality makes the angle-finding `n`-independent for fixed `p`; and it all compiles to one Hadamard layer plus `p` rounds of cost-phase-then-mix.

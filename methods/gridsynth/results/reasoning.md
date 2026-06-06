Let me start from the thing that actually hurts. On a fault-tolerant machine, the Clifford gates — `H`, `S`, the scalar `ω = e^{iπ/4}` — are cheap. The `T`-gate, `T = diag(1, e^{iπ/4})`, is the expensive one: it needs magic-state distillation, and the cost of a single-qubit computation is, to first approximation, just the number of `T`-gates, the T-count. And almost every algorithm I care about eventually wants a `z`-rotation `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})` for some angle θ that is not a multiple of π/4. So I'm stuck with: given θ and a precision ε, produce a Clifford+T circuit `U` with `‖Rz(θ) − U‖ ≤ ε`, and I want it to use as few `T`s as humanly possible. Not "asymptotically few." Few.

What do I have to work with? Solovay–Kitaev. It works — it's universal, it'll get me to any ε. The trouble is how. It nets `SU(2)` at some coarse ε₀, then recursively refines using group commutators, shrinking error geometrically each level. The length comes out `O(log^c(1/ε))` with `c = log5/log(3/2) ≈ 3.97`. So the exponent is already worse than 3, and the hidden constant is large, and — this is the part that bugs me — it treats Clifford+T as just *some* universal gate set. It never uses the fact that Clifford+T has razor-sharp algebraic structure. It's a geometric algorithm wearing a blindfold.

At the other extreme: exhaustive search. Enumerate circuits by T-count, test each. That gives the genuinely optimal answer. But the number of circuits of T-count `m` grows exponentially in `m`, so search dies around ε ≈ 10⁻⁴, or 10⁻¹⁷ if I'm clever with meet-in-the-middle. Useless at high precision.

So I have "fast but wasteful" and "optimal but exponential," and a big gap between them. Where would structure come from? A 2×2 unitary can be written *exactly* as a Clifford+T product **iff** all its entries lie in `D[ω] = Z[1/√2, i]`. That's an iff. The reachable operators aren't some fuzzy net — they are *exactly* the unitaries over `D[ω]`. So my approximation problem is "find a `D[ω]`-unitary near `Rz(θ)`," which is discrete. And there's even an exact-synthesis routine that, given such a unitary, spits out a *minimal*-T-count circuit for it. So I don't have to worry about how to build the circuit; I only have to find the right `D[ω]`-unitary, and the T-count is whatever that unitary's structure dictates.

Let me make that form concrete. A Clifford+T-representable unitary can be written
`U = [[u, −t†ω^ℓ], [t, u†ω^ℓ]]`, `u, t ∈ D[ω]`, `ℓ ∈ Z`, with `u†u + t†t = 1`.
The `ω^ℓ` is annoying. But I can argue it away when ε is small. Suppose `ε < |1 − e^{iπ/8}|` and `U` is within ε of `Rz(θ)`. Then `‖I − U Rz(θ)⁻¹‖ ≤ ε < |1 − e^{iπ/8}|`, and the left side equals `max(|1−e^{iφ₁}|, |1−e^{iφ₂}|)` over the eigenphases of `U Rz(θ)⁻¹`. So each `|φⱼ| < π/8`, hence `|φ₁+φ₂| < π/4`, hence `|1 − e^{i(φ₁+φ₂)}| < |1 − e^{iπ/4}| = |1 − ω|`. But `e^{i(φ₁+φ₂)} = det(U Rz(θ)⁻¹) = ω^ℓ`, so `|1 − ω^ℓ| < |1 − ω|` forces `ω^ℓ = 1`. Good — for small ε I can take ℓ = 0:
`U = [[u, −t†], [t, u†]]`, `u†u + t†t = 1`. (And if ε is large there's a trivial Clifford solution `u = ω^{−j}`, `t = 0`, T-count 0, so nothing is lost.)

Now what's the T-count of this `U`? The exact-synthesis routine reduces the denominator exponent of the matrix step by step, each step multiplying by `H` and a power of `T` from the left. Let `k` be the least denominator exponent of `u` — the smallest `k` with `√2^k u ∈ Z[ω]`. First note `u` and `t` *share* the same least denominator exponent: in `Z[ω]`, `√2 | s` iff `2 | s†s`, so `√2^k u ∈ Z[ω]` iff `2^k u†u ∈ Z[ω]` iff `2^k(1 − t†t) ∈ Z[ω]` iff `√2^k t ∈ Z[ω]`. So `t` has denominator exponent `k` too. Tracing the residues through exact synthesis, the T-count of this special `U` comes out as either `2k` or `2k − 2`. And when it's `2k`, I can fix it: conjugate, `U' = T U T†`. Since `T` commutes with `Rz(θ)`, `‖Rz(θ) − U'‖ = ‖T Rz(θ) T† − T U T†‖ = ‖Rz(θ) − U‖`, so `U'` is an equally-good approximation, and it has T-count `2k − 2`. So without loss of generality the cost is `2k − 2` (or 0 when k = 0).

That collapses the whole problem to one number: **minimize the least denominator exponent `k` of `u`.** Everything from here is about finding the smallest-`k` admissible `u`.

Let me turn the closeness constraint into a constraint purely on `u`. Set `z = e^{-iθ/2}`. Using `u†u + t†t = 1` and `z†z = 1`:
`‖Rz(θ) − U‖² = ‖u − z‖² + ‖t‖² = (u−z)†(u−z) + t†t = u†u + t†t − z†u − u†z + z†z = 2 − 2 Re(z†u)`.
So `‖Rz(θ) − U‖ ≤ ε` ⟺ `Re(z†u) ≥ 1 − ε²/2`. Identify `z = (x,y)` and `u = (a,b)` as real vectors; `Re(z†u)` is exactly the inner product `⟨z⃗, u⃗⟩`. The constraint is
`⟨z⃗, u⃗⟩ ≥ 1 − ε²/2`.
Geometrically that's a half-plane cutting the unit disk: keep the part of the disk on the far side of a chord at distance `ε²/2` below the boundary point in direction `z⃗`. Call that sliver the **ε-region** `R_ε`. It's thin — width `ε²/2` at the widest, you can inscribe a disk of radius `ε²/4` in it, area `Θ(ε³)`.

So I want the smallest-`k` point `u ∈ D[ω]` with `u ∈ R_ε`. But that's not the whole story, because I also need `t` to *exist*. When does the Diophantine equation `t†t = 1 − u†u` have a solution? There's a necessary condition I can read off cheaply, and it's geometric. Unitarity says `u†u = 1 − t†t ≤ 1`, so `u` is in the unit disk — fine, `R_ε` is already inside the disk. But apply `(−)•`, the automorphism `√2 ↦ −√2`: `(u•)†(u•) = 1 − (t•)†(t•) ≤ 1`, so `u•` is *also* in the unit disk. That second condition is *not* implied by the first; it's a genuinely separate constraint, on the √2-conjugate of `u`.

Stop and look at what I have. I'm hunting for `u ∈ D[ω]` such that
`u ∈ A` (the ε-region) and `u• ∈ B` (the unit disk).
A point of `D[ω]` is `(some scaling of) Z[ω]`, a 2D lattice-ish set. I want lattice points whose value lands in one convex set and whose √2-conjugate lands in another. This is a clean abstract problem in its own right: **the two-dimensional grid problem** — find `u ∈ Z[ω]` (or `D[ω]` at scale `1/√2^k`) with `u ∈ A`, `u• ∈ B`. If I can solve *that* efficiently, enumerating solutions in order of increasing `k`, then for each candidate `u` I just try to solve `t†t = 1 − u†u`, and the first one that works gives the optimal circuit. Let me build the grid solver, then come back and handle the Diophantine equation.

Begin in one dimension, where the conjugates live in `Z[√2]`. Given intervals `A = [x₀,x₁]`, `B = [y₀,y₁]`, find `α = a + b√2 ∈ Z[√2]` with `α ∈ A`, `α• = a − b√2 ∈ B`. How many solutions, and can I list them cheaply? The discreteness fact `|α−β||α•−β•| ≥ 1` for distinct grid points means if the two intervals have widths `δ, Δ` with `δΔ < 1` there's at most one solution; and `δΔ ≥ (1+√2)²` forces at least one. So I can rescale: the grid problem for `(A, B)` is equivalent to the one for `(λ⁻¹A, −λB)` where `λ = 1+√2`, because `α ∈ A, α• ∈ B` iff `λ⁻¹α ∈ λ⁻¹A` and `(λ⁻¹α)• = −λ α• ∈ −λB` (using `(λ⁻¹)• = −λ`). Rescaling like this I can assume `λ⁻¹ ≤ δ < 1`. Then for any fixed integer `b`, the constraint `x₀ ≤ a + b√2 ≤ x₁` with `x₁−x₀ < 1` pins down at most one integer `a`. And `b = (α − α•)/√(2)³`... more precisely `b = (α − α•)/(2√2)`, so any solution has `b ∈ [(x₀−y₁)/(2√2), (x₁−y₀)/(2√2)]`. Enumerate the integers `b` in that range, solve for the unique `a`, test. The number of `b` to check is `O(Δδ)`, and the number of solutions is `Ω(Δδ)` — so it's a constant number of operations per solution. The 1D grid problem is *efficient*.

Two dimensions. `Z[ω]` decomposes nicely: every `u ∈ Z[ω]` is `α + βi` or `α + βi + ω` with `α, β ∈ Z[√2]` (because `ω = (1+i)/√2`, and the parity of the integer coefficients decides which coset). If both `A` and `B` are **upright rectangles** — products of intervals, axis-aligned — the 2D grid problem just splits into independent 1D problems on the real and imaginary parts. In the `α+βi` case, `u• = α• + β•i`, so `u ∈ A_x×A_y` and `u• ∈ B_x×B_y` is exactly `(α ∈ A_x, α• ∈ B_x)` and `(β ∈ A_y, β• ∈ B_y)`; the `+ω` coset shifts the intervals by `±1/√2`. So: upright rectangles are easy.

What if `A` and `B` are convex but only *close* to upright? Define the **uprightness** `up(A) = area(A)/area(BBox(A))`, the fraction of its bounding box the set fills (between 0 and 1). If `A, B` are `M`-upright, I can enumerate the grid points of the bounding boxes (easy, they're upright rectangles) and just filter for the ones actually in `A, B`. By convexity, a fraction `≈ M²` of bounding-box grid points survive, so the cost is `O(1/M²)` per solution — a constant when `M` is bounded below.

But what if the set is *skinny and tilted*, uprightness near zero? Then the bounding box is enormous compared to the set and I'd enumerate a flood of useless candidates. I need to fix the tilt. The move: apply a linear map that turns the skinny tilted set into an upright one — but it has to respect the lattice, i.e. map `Z[ω]` into `Z[ω]`. Call such a real-linear `G` with `G(Z[ω]) ⊆ Z[ω]` a **grid operator**, and **special** if `det G = ±1`. Why do these help? Because they preserve solutions: `u` solves the grid problem for `(A, B)` iff `Gu` solves it for `(G(A), G•(B))`, where `G•` applies `(−)•` to each entry. The proof is one line — `u ∈ A, u• ∈ B` iff `Gu ∈ G(A)` and `(Gu)• = G• u• ∈ G•(B)` — using the homomorphism property `G• u• = (Gu)•`. So if I can find a special `G` that makes `G(A)` and `G•(B)` simultaneously upright, I've reduced the hard tilted problem to the easy upright one, and `G⁻¹` (also a special grid operator) carries the solutions back.

The crux is therefore: **given convex `A, B`, find a grid operator that makes both upright.** And here I should be honest about what "both" costs me — `G` makes `A` upright but it's `G•`, the conjugate operator, that acts on `B`, and `G•` is generally a *different* (often expansive) map. I can't just rotate `A` to be axis-aligned and ignore what that does to `B`. I have to upright them *together*. Let me restrict to the case that matters and is tractable: `A, B` are **ellipses**. (Any bounded convex set inscribes in an ellipse of area at most `4π/(3√3) ≈ 2.418×` its own — sharp for the equilateral triangle — so handling ellipses handles everything up to a constant factor.)

An ellipse `{u : (u−p)†D(u−p) ≤ 1}` with positive-definite `D = [[a,b],[b,d]]` has area `π/√det D` and bounding-box area `4√(ad)/det D`, so `up(E) = (π/4)√(det D / ad)`. Normalize to `det D = 1`; then `D = [[eλ^{−z}, b],[b, eλ^z]]` with `e² = b² + 1`, and `up(E) = π/(4e²) = π/(4√(b²+1))`. So uprightness is high exactly when `b` (the off-diagonal, the "tilt") is small. For a *pair* of ellipses define a **state** `(D, Δ)` — two SPD det-1 matrices, `D` with off-diagonal `b` and "stretch" `z`, `Δ` with off-diagonal `β` and stretch `ζ`. Define the **skew** `b² + β²` and the **bias** `ζ − z`. Both ellipses are upright iff the skew is small.

So I want an iterative procedure: while the skew is large, apply one special grid operator that provably *decreases* it, until it's below a threshold (15 turns out to be a convenient cutoff, corresponding to uprightness comfortably bounded below). Why can I always decrease it? The action of a special `G` on the state is `(D, Δ) ↦ (G†DG, G•†ΔG•)`. The bias measures the relative stretch of the two ellipses; first use "shift" operations (conjugates of `σ`-like maps, which scale the diagonals by powers of `λ`) to bring the bias into a small window like `[−1, 1]`, so the two ellipses are stretched comparably. Then, in the small-bias regime, a finite menu of fixed special grid operators each handles a region of the `(z, ζ)` and sign-of-`b` configuration:
- a Hadamard-like rotation `R = (1/√2)[[1,−1],[1,1]]` when both stretches are near zero — it kills the tilt outright by rotating;
- shears `Aⁿ = [[1,−2],[0,1]]ⁿ` and `Bⁿ = [[1,√2],[0,1]]ⁿ` when one direction is much longer, picking `n` to undo most of the shear;
- a `K = (1/√2)[[−λ⁻¹,−1],[λ,1]]`-type operator in the strongly-asymmetric corner;
- and `X` (swap), `Z` (sign-flip) symmetries to fold the case analysis in half.
Each region's operator drops the skew by a factor of at most ≈ 0.9, and one of the regions always applies, so iterating reduces the skew geometrically. Since the starting skew is `O(1/M²)` from uprightness `M`, the loop runs `O(log(1/M))` times. That's the engine: a special grid operator making any pair of ellipses `(1/6)`-upright (or so) in `O(log(1/M))` operations.

Now I can solve the general 2D grid problem: enclose `A, B` in ellipses, run the skew-reduction to get `G`, enumerate solutions of the upright problem for `G(A), G•(B)`, pull them back through `G⁻¹`, filter against the original `A, B`. Total work `O(log(1/M))` plus a constant per solution. That `O(log(1/M))`-overall versus the naïve `O(1/M²)`-per-candidate is the whole payoff of grid operators.

I need a *scaled* version, because my `u` lives in `D[ω] = ⋃_k (1/√2^k) Z[ω]`, not `Z[ω]`. For fixed `k`: `u = (1/√2^k) v` solves the scaled problem for `(A,B,k)` iff `v ∈ Z[ω]` solves the unscaled problem for `(√2^k A, (−√2)^k B)` — note the `(−√2)^k` on the `B` side, since `((1/√2^k)v)• = ((−1)^k/√2^k) v•`. And uprightness is scale-invariant, so the grid operator `G` only has to be computed *once*, not per `k`; for each `k` I rescale and re-run the (cheap) upright enumeration. To enumerate in order of increasing `k`, just run `k = 0, 1, 2, …`. (A point first appears at exponent `k > k−1` iff, writing `u·√2^k = aω³+bω²+cω+d`, `a−c` or `b−d` is odd — so I can skip duplicates.)

One more thing I want before moving on — a guarantee that the supply of candidates *grows*, because I'll be discarding some. Two lower bounds. First: if `A` contains a disk of radius `r`, `B` a disk of radius `R`, and `rR ≥ (1+√2)²/2^k`, then the scaled problem at exponent `k` has at least **two** solutions (inscribe squares in the disks and apply the 1D bound twice). Second, and this is the multiplier: if there are two distinct solutions `u ≠ v` at exponent `k`, then for every ℓ ≥ 0 there are at least `2^ℓ + 1` solutions at exponent `k + 2ℓ`. Why — take `φ = j/2^ℓ` for `j = 0,…,2^ℓ` and form `u_j = φu + (1−φ)v`. Each `u_j` has denominator exponent `k + 2ℓ`. It's a convex combination of `u` and `v`, so `u_j ∈ A` by convexity; and since `φ• = φ` (it's rational), `u_j• = φu• + (1−φ)v•` is a convex combination of `u•, v•`, so `u_j• ∈ B`. Both sets convex ⟹ all `2^ℓ + 1` of them are solutions. So once I have two, the count explodes exponentially in ℓ. Good.

Back to the Diophantine equation: for a candidate `u`, solve `t†t = ξ`, `ξ = 1 − u†u ∈ D[√2]`, `t ∈ D[ω]`. When is this solvable, and how? Necessary: `t†t = ξ ≥ 0` and `(t•)†(t•) = ξ• ≥ 0` — `ξ` must be **doubly positive**. The geometry already guarantees this for my candidates (both `u` and `u•` in the unit disk give `0 ≤ ξ, ξ• ≤ 1`). The rest is classical algebraic number theory in the Euclidean domains `Z`, `Z[√2]`, `Z[ω]`.

Work it in stages. First reduce `D[√2]` to `Z[√2]`: with `δ = 1 + ω` satisfying `δ†δ = λ√2 ∼ √2`, the element `ξ` is `†`-decomposable iff `√2 ξ` is — so multiply away the dyadic denominator until `ξ' = √2^ℓ ξ ∈ Z[√2]` with `ξ'•ξ' = n ∈ Z`, solve there, and divide back by `δ^ℓ`. Now in `Z[√2]`: I want `t†t ∼ ξ'` (up to a unit; doubly-positive units are squares, so `∼` recovers exact equality). Factor `ξ'` into primes of `Z[√2]`. A prime `ξ'` divides a unique rational prime `p`, and the question of whether `ξ'` is `†`-decomposable depends only on `p mod 8`:
- `p = 2`: `ξ' ∼ √2`, and `δ†δ = λ√2 ∼ √2`, so yes.
- `p ≡ 1 (mod 4)`: there's `u` with `u² ≡ −1 (mod p)`, so `ξ' | p | u²+1 = (u+i)(u−i)`; take `t = gcd(ξ', u+i)`, and a three-case argument (`t†t ∼ 1, ξ', or ξ'²`) forces `t†t ∼ ξ'`. Yes.
- `p ≡ 3 (mod 8)`: similarly `−2` is a QR, `u² ≡ −2`, `ξ' | u²+2 = (u+i√2)(u−i√2)`, `t = gcd(ξ', u+i√2)`. Yes.
- `p ≡ 7 (mod 8)`: **no** (to odd power). If `t†t ∼ ξ'`, then `(t•t)(t•t)† ∼ ξ'•ξ' ∼ p`, and `t•t ∈ Z[i]`, say `a+bi`, so `p ∼ a²+b²`; but `a², b² ∈ {0,1,4} (mod 8)`, which can never be `7`. Contradiction.
And a prime to an *even* power is always a square, hence decomposable. So: `t†t = ξ` is solvable iff `ξ` is doubly positive and every prime `p | n` with `p ≡ 7 (mod 8)` occurs to even multiplicity. The whole construction is constructive given the **prime factorization of `n`** (the inner steps — `√(−1) mod p`, `√(±2) mod p` — are probabilistic-polynomial). Factoring `n` is the one genuinely hard subproblem.

Now assemble the algorithm. Enumerate `u ∈ D[ω]` solving the grid problem `u ∈ R_ε`, `u• ∈ disk`, in order of increasing `k`. For each, set `ξ = 1 − u†u`, write `ξ•ξ = n/2^ℓ`, factor `n`, solve `t†t = ξ`; the first success gives `U = [[u,−t†],[t,u†]]`, then `U' = T U T†`, synthesize whichever has smaller T-count, done. That's optimal: candidates come in increasing `k`, the Diophantine solver succeeds exactly when a `t` exists, so the first success has the smallest possible `k`, and `T`-count `2k−2` is then minimal over all approximations — *absolutely* minimal, not asymptotically, because any competing approximation reduces (by the ℓ=0 lemma) to the same form with denominator exponent `≥ k`.

The wrinkle is factoring. With a factoring oracle (a quantum computer running Shor), the factoring step always succeeds and the optimality above is unconditional. Without one, I can't factor a generic large `n` in reasonable time — so I cap the factoring effort and, conservatively, only *accept* a candidate when `n` happens to be prime (then a single primality-flavored test handles it). How many candidates must I discard before hitting a prime `n`?

The numbers `n` aren't arbitrary: each one satisfies `n ≥ 0` and either `n = 0` or `n ≡ 1 (mod 8)` (a parity computation on the residues). And a prime `n ≡ 1 (mod 8)` is *always* solvable — by the `p ≡ 1 (mod 4)` case above. So "n is prime" already implies "the Diophantine equation succeeds." Now count. For a candidate of exponent `k_j`, `n_j ≤ 4^{k_j}` (since `0 ≤ ξ, ξ• ≤ 1`), so by the prime number theorem the chance `n_j` is prime is at least `≈ 2/ln(4^{k_j}) = 1/(k_j ln2)`. The exponential-growth lemma bounds `k_j ≤ k₂ + 2(1 + log₂ j)` (since once two solutions exist at `k₂`, there are `≥ 2^ℓ+1` by `k₂+2ℓ`). And the two-solutions bound with `r = ε²/4`, `R = 1` gives `k₂ = O(log(1/ε))`. Plugging in, the expected index of the first prime `n` is
`E(j₀) = Σ_{j≥0} P(j₀ > j) ≤ 1 + Σ_{j≥1} (1 − 1/((k₂+2)ln2 + 2 ln j))^j = O(k₂) = O(log(1/ε))`,
using the summation estimate `Σ (1 − 1/(a + b ln x))^x = O(a)`. So I try `O(log(1/ε))` candidates in expectation. And the T-count I land on: with `k ≤ k_{j₀} ≤ k'' + 2(1 + log₂ j₀)` where `k''` is the second-to-optimal exponent, and `2k−2 ≤ m ≤ 2k`, taking expectations (and `E(log j₀) ≤ log E(j₀)` by concavity),
`E(m) ≤ m'' + 6 + 4 log₂ E(j₀) = m'' + O(log log(1/ε))`.
So in the absence of a factoring oracle I lose only an additive `O(log log(1/ε))` over the second-to-optimal solution — near-optimal. (And as a bonus: the first candidate whose solver doesn't fail gives a *firm* per-instance lower bound `2k_j − 2` on the T-count, not just the average-case info-theoretic bound.)

What's the actual T-count number? Worst case my algorithm enumerates *all* grid solutions and can solve the Diophantine equation whenever the older subset-enumeration method could, so it's never worse than `K + 4 log₂(1/ε)` (`K ≈ 10`) — and that worst case is hit only where it's genuinely optimal. When is that? Look at the ε-region's straight boundary: its slope is `r = −1/tan(θ/2)`, independent of ε. Two grid points span a line of slope in `Q(√2)`. If `r ∈ Q(√2)`, the ε-region runs *parallel* to a discrete family of grid lines, so the number of solutions at exponent `k` is governed by the region's **width** `ε²/2` versus the grid-line spacing `1/2^k`: I get a solution when `1/2^k ≲ ε²`, i.e. `k ≳ K + 2 log(1/ε)`, T-count `2k ≈ K + 4 log(1/ε)`. But if `r ∉ Q(√2)`, the grid lines are dense and it's the **area** `Θ(ε³)` versus the grid-point density `4^k` that matters: a point lands in the region when `1/4^k ≲ ε³`, i.e. `2k ≳ K + 3 log(1/ε)`, the typical T-count `K + 3 log₂(1/ε)` — matching the information-theoretic lower bound. So generic angles get the optimal `3 log₂`; only the measure-zero `tan(θ/2) ∈ Q(√2)` angles pay `4 log₂`.

Complexity overall: `M = up(R_ε) = Ω(ε⁴)`, so `log(1/M) = O(log(1/ε))` for the grid setup; `j₀ = O(log(1/ε))` candidates; each Diophantine solve is `polylog(n)` with `n ≤ 4^{k_{j₀}}`, `k_{j₀} = O(log(1/ε))`; exact synthesis `O(k_{j₀})`. Total expected arithmetic `O(polylog(1/ε))`, each operation at precision `O(log(1/ε))` — expected time `O(polylog(1/ε))`, oracle or not.

I also want the "up to a phase" version, since global phase is unobservable. Claim: it suffices to consider only `λ ∈ {1, e^{iπ/8}}`. Because Clifford+T operators have discrete determinant (`det U = ω^k`), so for any optimal phase I can snap it to `e^{inπ/8}` (using: for unitary `W` with `det W = 1`, `tr W ≥ 0`, `‖I − W‖ ≤ ‖I − λW‖` for every unit λ — diagonalize and compare arcs), then absorb `ω^k = e^{ikπ/4}` into `U` itself at no T-cost, leaving the phase in `{1, e^{iπ/8}}`. Run the on-the-nose algorithm for `λ = 1` (interleaving even T-counts) and a twin for `λ = e^{iπ/8}` (where `U = [[u, −t†ω⁻¹],[t, u†ω⁻¹]]`, giving odd T-counts), enumerate both in increasing T-count, take the smaller. The `ω⁻¹` twin is exactly the same machinery with the constraint sets scaled by `√2±1`-flavored factors.

Let me write the code, mirroring the structure I've derived: ring arithmetic with both conjugations; the two convex sets `EpsilonRegion` and `UnitDisk` (each as an ellipse plus a membership/line-intersection method); the skew-reduction `to_upright`; the scaled 2D grid solver `solve_2d_grid`; the Diophantine solver; the exact synthesis; and the main increasing-`k` loop.

```python
import mpmath
from rings import ZRootTwo, ZOmega, DRootTwo, DOmega   # +,*, conj (†), conj_sq2 (•), denomexp k
from exact_synthesis import decompose_domega_unitary   # KMM: minimal-T Clifford+T from a D[ω]-unitary
from diophantine import diophantine_dyadic             # solve t†t = ξ given a factoring of n, else NO_SOLUTION
from to_upright import to_upright_set_pair             # skew-reduction -> grid operator G + upright bboxes
from grid_2d import solve_TDGP                          # scaled 2D grid problem at fixed k


class EpsilonRegion:
    # the slice of the unit disk with ⟨z⃗, u⃗⟩ ≥ 1 − ε²/2,  z = e^{-iθ/2}
    def __init__(self, theta, epsilon):
        self.theta, self.epsilon = theta, epsilon
        self.zx, self.zy = mpmath.cos(-theta / 2), mpmath.sin(-theta / 2)
        self.d = mpmath.sqrt(1 - epsilon**2 / 4)        # chord depth, in cos terms
        # enclosing ellipse: rotate to z⃗, stretch (long along the chord ~1/ε, short ~1/ε²)
        R    = mpmath.matrix([[self.zx, -self.zy], [self.zy, self.zx]])
        Sdiag = mpmath.matrix([[64 / epsilon**4, 0], [0, 4 / epsilon**2]])
        self.ellipse = (R, Sdiag, R.T, self.d)          # D = R · Sdiag · Rᵀ, centered at d·z⃗

    def inside(self, u):                                # u ∈ D[ω]: in unit disk AND past the chord
        cos_sim = self.zx * u.real + self.zy * u.imag
        return DRootTwo.from_domega(u.conj * u) <= 1 and cos_sim >= self.d

    def intersect(self, p, q):                          # {t : p + t q ∈ R_ε} as an interval
        # quadratic for the disk boundary, linear for the chord; intersect the two
        ...


class UnitDisk:
    def inside(self, u):
        return DRootTwo.from_domega(u.conj * u) <= 1
    def intersect(self, p, q):
        ...                                             # quadratic |p + t q|² ≤ 1


def gridsynth(theta, epsilon):
    A = EpsilonRegion(theta, epsilon)                   # u ∈ A   (closeness)
    B = UnitDisk()                                      # u• ∈ B  (so the Diophantine eq is solvable)

    # one skew-reduction: find special grid operator G s.t. G(A), G•(B) are upright (compute once)
    G, ellA, ellB, bboxA, bboxB = to_upright_set_pair(A, B)

    k = 0
    while True:                                         # increasing least denominator exponent
        for u in solve_TDGP(A, B, G, ellA, ellB, bboxA, bboxB, k):   # u ∈ D[ω], u ∈ A, u• ∈ B
            if (u * u.conj).denom_is_trivial:           # skip u already appearing at exponent < k
                continue
            xi = 1 - DRootTwo.from_domega(u.conj * u)   # ξ = 1 − u†u ∈ D[√2]
            t = diophantine_dyadic(xi)                  # solve t†t = ξ  (NO_SOLUTION if n unfactorable/insoluble)
            if t is not None:
                U = DOmegaUnitary(u, t)                 # [[u, −t†], [t, u†]]
                return decompose_domega_unitary(U)      # minimal-T Clifford+T circuit (T-count 2k−2)
        k += 1
```

```python
# the scaled 2D grid solver: enumerate u ∈ (1/√2^k) Z[ω] with u ∈ A, u• ∈ B
def solve_TDGP(A, B, G, ellA, ellB, bboxA, bboxB, k):
    Ginv = G.inv
    # in the UPRIGHT frame, the bounding boxes are axis-aligned rectangles:
    # enumerate the y-coordinates β (a scaled 1D grid problem on the bbox's y-intervals),
    for beta in solve_scaled_1d(bboxA.Iy, bboxB.Iy, k + 1):
        # for each row, the x-extent is the line A ∩ (row) intersected with B ∩ (conjugate row),
        z0 = Ginv * point(alpha0, beta, k + 1)          # a corner of this row, pulled back to the A,B frame
        v  = Ginv * step(k)                              # the unit step along x, pulled back
        tA = A.intersect(z0, v)                          # interval of t with z0 + t v ∈ A
        tB = B.intersect(z0.conj_sq2, v.conj_sq2)        # interval of t with (z0 + t v)• ∈ B
        if tA is None or tB is None:
            continue
        # x-coordinates α are a 1D grid problem on (tA, tB) with the parity fixed by β,
        for alpha in solve_scaled_1d_with_parity(tA, tB, parity_of(beta), k):
            u = point(alpha, beta, k)
            if A.inside(u) and B.inside(u.conj_sq2):     # final exact filter
                yield u
```

```python
# the skew-reduction loop: repeatedly apply one special grid operator that drops the skew,
# until both ellipses are upright.  Returns the composite grid operator.
def to_upright_ellipse_pair(ellA, ellB):
    state = normalize_pair(ellA, ellB)                  # (D, Δ): SPD, det 1, off-diagonals b, β
    Gl = Gr = GridOp.identity()
    while True:
        b, beta = state.D.b, state.B.b
        if beta < 0:                       state, G = apply(state, OP_Z)       # fold sign of β
        elif state.A.bias * state.B.bias < 1: state, G = apply(state, OP_X)    # swap to equalize biases
        elif state.bias far from 1:        state = shift(state, n); Gl, Gr = absorb(...)  # σ: bias -> ~1
        elif state.skew <= 15:             return Gl * Gr                       # upright enough — done
        elif both biases near 1:           state, G = apply(state, OP_R)       # rotate out the tilt
        elif b >= 0 and one bias small:    state, G = apply(state, OP_K)       # asymmetric corner
        elif b >= 0:                       state, G = apply(state, OP_A ** n)   # shear [[1,−2],[0,1]]ⁿ
        else:                              state, G = apply(state, OP_B ** n)   # shear [[1,√2],[0,1]]ⁿ
        Gr = G * Gr                                                            # accumulate
```

```python
# the Diophantine solver t†t = ξ, ξ ∈ D[√2], via factoring n = (ξ•ξ)·2^ℓ
def diophantine_dyadic(xi):
    if xi < 0 or xi.conj_sq2 < 0:                        # doubly positive is necessary
        return None
    # clear the dyadic denominator using δ = 1+ω (δ†δ = λ√2 ∼ √2), reduce to ξ' ∈ Z[√2]
    xi_int = clear_denominator(xi)                       # ξ' = √2^ℓ ξ, with ξ'•ξ' = n ∈ Z
    t = adj_decompose(xi_int)                            # factor n; per prime p: solvable unless p≡7 mod 8 to odd power
    if t is None:                                        # (uses √-1 mod p, √±2 mod p; Pollard-rho for n)
        return None
    return fix_up_units(t, xi)                           # multiply by δ^{-ℓ} and a Z[√2]-square to match ξ exactly
```

The causal chain, start to finish: Solovay–Kitaev is universal but blind to Clifford+T's algebra, so it overshoots the `3 log₂` lower bound; the exact-representability iff (`entries ∈ D[ω]`) plus minimal-T exact synthesis turn the problem into "find the smallest-denominator-exponent `u ∈ D[ω]` near `e^{-iθ/2}`," because T-count `= 2k−2`; rewriting closeness as `⟨z⃗,u⃗⟩ ≥ 1−ε²/2` and unitarity-via-`(−)•` as `u• ∈ disk` makes that a 2D grid problem; grid operators plus skew-reduction solve any such problem in `O(log(1/M))` and enumerate solutions in increasing `k`; for each candidate the relative-norm equation `t†t = 1−u†u` is solvable by classical number theory once `n` is factored; with a factoring oracle the first solvable candidate is absolutely T-optimal, and without one, accepting only prime `n` (always solvable, since `n ≡ 1 mod 8`) costs an expected `O(log(1/ε))` extra candidates and an additive `O(log log(1/ε))` in T-count — landing on the information-theoretic `3 log₂(1/ε)` for generic angles, in `O(polylog(1/ε))` time.

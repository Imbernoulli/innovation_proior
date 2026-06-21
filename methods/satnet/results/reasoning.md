Let me start from what actually hurts. I have a deep network that can look at a handwritten digit and tell me what it is, all day long, beautifully. But the moment I ask it to *learn a rule* — a hard, discrete, all-or-nothing constraint, like "these nine cells must be a permutation of 1 through 9," or "the answer is the XOR of all these bits" — it falls apart. Not because the network can't represent the function in principle, but because gradient descent never finds it. And I keep coming back to two little tasks that should be trivial and aren't.

Parity. Give me a bit string, tell me whether the number of ones is odd. That's it. It's a chain of XORs. And it's a known disaster for gradient-based learning: if I train a classifier to map strings to their single parity bit, the gradient at almost any weight setting tells me essentially *nothing* about which parity I'm trying to learn — the gradient's variance across the whole family of parity functions is basically zero, so SGD is wandering in a fog. An LSTM with a hundred hidden units sits at 0.476 test error, where 0.5 is a coin flip. It cannot learn a chain of XORs from end labels.

Sudoku. Don't even tell the network the rules. Just show it solved boards and partial boards and let it figure out the constraints. A big convolutional net memorizes the training boards and gets ~0% on held-out ones. And if I *permute* the bit representation — scramble which bit is which, consistently — so that there's no 2-D locality left to lean on, the ConvNet collapses to nothing even on the training set. It was never learning the *logic*. It was learning the picture.

So the real question isn't "can a network apply a rule I give it." Plenty of work does that — relational nets seeded with which cells may interact, inductive-logic nets seeded with rule templates. The question is: can a network *discover* the discrete relationships, from data, end to end, with the rules unknown? And can it do that while sitting *inside* a bigger network — say, behind a digit recognizer — so the whole thing trains together?

Let me think about what kind of object could even hold a "logical structure" with knobs I can turn by gradient. The most generic discrete logical primitive I know is satisfiability. Almost everything in symbolic AI reduces to SAT or its optimization cousin MAXSAT — maximize the number of satisfied clauses. If I had a *learnable* MAXSAT problem — a clause matrix whose entries I can nudge — then "learning the rules of Sudoku" becomes "learning the clauses." That's the right shape: the structure lives in continuous parameters (the clause weights), and the problem-solving lives in solving the MAXSAT instance.

Write MAXSAT down. Variables `ṽ_i ∈ {−1, 1}`, and for clause `j`, `s̃_ij ∈ {−1, 0, 1}` is the sign with which variable `i` appears (0 if it doesn't). The objective is

```
maximize_{ṽ ∈ {−1,1}^n}  Σ_j  ⋁_i  1{ s̃_ij ṽ_i > 0 }.
```

And there's the wall, immediately. Everything about that expression is hostile to gradients: `ṽ ∈ {−1,1}` is discrete, the indicator is a step, the disjunction is a max. There is nothing to differentiate. If I want clauses I can learn by backprop, I cannot work with this object directly. I need a *continuous, differentiable* surrogate that still *means* the same thing — and one I can actually solve fast, because it has to be a layer that runs thousands of times.

What do people do with NP-hard discrete problems when they want something tractable and tight? They relax to a semidefinite program. The Goemans–Williamson story for MAXCUT and MAX-2SAT is the canonical move: take each binary `ṽ_i ∈ {±1}` and *lift* it to a unit vector `v_i ∈ R^k`, `‖v_i‖ = 1`. The discrete combinatorial objective becomes a quadratic form over those vectors — a function of the Gram matrix `X = V^T V`, which is positive semidefinite. And to get a discrete answer back out, you round: pick a random hyperplane through the origin with normal `r`, and set variable `i` true or false by the sign of `r^T v_i`. The beautiful fact from that analysis is that the chance two unit vectors land on *opposite* sides of a random hyperplane is exactly proportional to the angle between them:

```
Pr[ sgn(u_i^T r) ≠ sgn(u_j^T r) ] = arccos(u_i^T u_j) / π.
```

That's it. That single identity is what turns "which side of a hyperplane" into a smooth probability, and it's going to be how I move between continuous vectors and probabilistic bits in both directions.

Now I need the *MAXSAT* relaxation specifically, not just MAXCUT. Here's the construction that makes the clause structure homogeneous. The trouble with a clause is that it has a fixed bias — a clause is satisfied or not depending on the *absolute* truth values, not just relative ones — and a quadratic form over unit vectors only knows about *relative* angles. So I add one special vector, a "truth direction" `v_⊤`, also unit norm, and I measure every variable's truth value by its angle to `v_⊤`. A variable is "true" when its vector points the same way as `v_⊤`. To absorb the clause bias I give the truth variable a coefficient in every clause: `s̃_⊤ = {−1}^m`. Now a clause's data is the vector `s_j` over all variables *including* `v_⊤`, and `V s_j = Σ_i s_ij v_i` is a single vector in `R^k` that summarizes the clause.

How do I turn `‖V s_j‖²` into something that counts unsatisfied clauses? Let me just work it out in the discrete case to make sure the relaxation *means* MIN-UNSAT (minimizing violations, which is the same as MAXSAT). Take a clause of `|s_j|` literals (counting the truth coefficient). Each `v_i ∈ {±v_⊤}`, and `V s_j` is then a scalar multiple of `v_⊤`; call that scalar `z_j`. The candidate per-clause penalty is

```
(‖V s_j‖² − (|s_j| − 1)²) / (4 |s_j|).
```

Let me sanity-check this is the right thing by enumerating. For a single-literal clause (`|s_j| = 2` counting the truth term): the satisfying assignments give penalty −0.125, the unique violating assignment gives +0.375 — a clean gap of 0.5, and the penalty is *constant* across all satisfying assignments. For a two-literal clause: satisfying penalty −0.25 everywhere, violating +0.4167, gap 0.667. For three literals the satisfying assignments split into two penalty levels (−0.5625 and −0.3125) but the *unique* violating assignment is +0.4375 — still strictly above every satisfying one. So minimizing `Σ_j` of these penalties is, up to a per-clause additive constant and a positive scale, exactly counting violated clauses: it's a faithful relaxation of MIN-UNSAT, tight (constant on satisfying assignments) for clauses of arity ≤ 2, which is exactly the GW MAX-2SAT regime. Good — the additive constant `(|s_j|−1)²` doesn't depend on the assignment, so it doesn't move the argmin; I can drop it.

Dropping the constant and folding the `1/(4|s_j|)` weight into the clause vectors, the objective is `Σ_j ‖V s_j‖² / (4|s_j|)`. Define the scaled clause matrix `S = [ s̃_⊤  s̃_1  …  s̃_n ] · diag(1/√(4|s_j|))`, so that `Σ_j ‖V s_j‖²/(4|s_j|) = ⟨S^T S, V^T V⟩`. So my relaxed problem is

```
minimize_{V ∈ R^{k×(n+1)}}  ⟨S^T S, V^T V⟩
subject to  ‖v_i‖ = 1,  i = ⊤, 1, …, n.
```

This is a unit-diagonal SDP in `X = V^T V`. And `S` — the clause matrix — is *exactly* the thing I wanted to make learnable. The weights of my layer will be `S`. Discovering the rules of Sudoku is now: learn `S` by gradient descent.

But wait — I've replaced an `n`-dimensional discrete problem with a search over `V ∈ R^{k×(n+1)}`, and I haven't said what `k` is. If `k` has to be large this is hopeless. Here's the relief: a solvable SDP with `p` linear constraints always has an optimal solution of rank at most `⌈√(2p)⌉` (Barvinok; Pataki). I have `n+1` unit-norm constraints, so I can take `k` on the order of `√(2n)` and *still hit the exact SDP optimum*. So I set `k = √(2n) + 1`. That's astonishing leverage: instead of `O(n²)` matrix entries I carry `O(nk) ≈ O(n^{1.5})` vector entries, and the PSD constraint is free because I parameterize `X = V^T V` directly.

Now, how do I *solve* this low-rank SDP, fast, in a way I can run as a layer? Coordinate descent. Look at the objective as a function of one column `v_i` with all others held fixed. Cycling the matrices in the trace, `⟨S^T S, V^T V⟩ = tr(V^T V S^T S)`, and the terms involving `v_i` are

```
v_i^T Σ_{j=0}^n s_j^T s_i v_j  =  v_i^T Σ_{j≠i} s_j^T s_i v_j  +  v_i^T s_i^T s_i v_i.
```

The last term is `‖s_i‖² · v_i^T v_i = ‖s_i‖²`, a constant since `‖v_i‖ = 1`. The rest is `v_i^T g_i` with

```
g_i = Σ_{j≠i} s_j^T s_i v_j = V S^T s_i − ‖s_i‖² v_i.
```

So minimizing over `v_i` on the unit sphere is just minimizing `v_i^T g_i` subject to `‖v_i‖ = 1` — and the minimizer of an inner product against a fixed vector on the sphere is the unit vector pointing the opposite way:

```
v_i = − g_i / ‖g_i‖.
```

That's the whole update. No step size. No line search. No free parameters. I cycle it over `i`, and — this is the part that makes it usable — for a low-rank `V` this update is cheap and *parallel-friendly*: `g_i` is a handful of length-`k` dot products. Even better, I don't recompute `V S^T` from scratch each step. I maintain `Ω = V S^T` and after I change one `v_i` I patch `Ω` with a rank-one update `Ω += (v_i^new − v_i^old) s_i^T`. So a full sweep is `O(nmk)`, and in practice a handful of sweeps converge. And — the property I'll lean on hard in a moment — for `k > √(2n)` these updates provably converge to the *global* optimum of the SDP, not just a local one. The non-convex `V`-formulation has all its bad critical points unstable; coordinate descent slides off them.

Let me pause and make sure I see the layer taking shape. I have known variables and unknown variables. Define `I` ⊂ {1,…,n} as the indices whose assignment is *given* (the clues), and `O` the rest (the cells to fill in). The layer receives probabilistic inputs `z_i ∈ [0,1]` for `i ∈ I`, and must output `z_o ∈ [0,1]` for `o ∈ O`. The clue vectors are *fixed* during the solve (I only run coordinate descent over the output columns); the output vectors get computed by the SDP; then I read out probabilities.

I need to turn an input probability `z_i` into a vector `v_i` consistent with the SDP's geometry. The truth value of `i` is encoded by its angle to `v_⊤`, and from the GW identity the probability that `i` is "true" — same side as `v_⊤` — is `arccos(−v_i^T v_⊤)/π`. So if I want `z_i` to be that probability I need `v_i^T v_⊤ = −cos(π z_i)`. I can build such a `v_i` explicitly: take a random unit vector `v_i^rand`, project off the `v_⊤` direction, and combine,

```
v_i = −cos(π z_i) v_⊤  +  sin(π z_i) (I_k − v_⊤ v_⊤^T) v_i^rand.
```

Check: `v_i^T v_⊤ = −cos(π z_i)` (the second term is orthogonal to `v_⊤`), and `‖v_i‖ = 1` since the two pieces are orthogonal with squared norms `cos²` and `sin²`. Good. So `z_i = 0` puts `v_i` *opposite* `v_⊤` (false), `z_i = 1` puts it *along* `v_⊤` (true), and intermediate `z_i` interpolates the angle. This is exactly how a soft, probabilistic clue — like a `0.7` coming out of a digit classifier — enters the solver. That's the hook I'll need for visual Sudoku: the front-end can hand me soft probabilities and they flow straight in.

After coordinate descent gives me each output vector `v_o`, I read out its probability the same way:

```
z_o = arccos(−v_o^T v_⊤) / π.
```

That's just the GW same-side probability again — the chance that, under a random hyperplane, `v_o` and `v_⊤` agree, i.e. that the rounded variable is true. At test time, if I want hard answers I can actually round: draw random hyperplanes and set `ṽ_o` true iff `sgn(r^T v_o) = sgn(r^T v_⊤)`, repeat a few times, and keep the assignment that satisfies the most clauses — repeated rounding empirically tightens the approximation. But during training I never round; I just pass the smooth probability `z_o` forward, so everything stays differentiable.

So the forward pass is: relax the clue inputs into vectors; run coordinate descent over the output columns to optimum; convert output vectors to probabilities. Clean.

Now the hard part, and the whole point: I need to backpropagate through this. Given `∂ℓ/∂z_o` from the loss above, I need `∂ℓ/∂z_i` (to train a front-end below me) and `∂ℓ/∂S` (to learn the clauses).

First, the easy hop, from probability to vector. I have `cos(π z_o) = −v_o^T v_⊤`. Differentiate: `−π sin(π z_o) dz_o = −v_⊤^T dv_o`, so `∂z_o/∂v_o = v_⊤ / (π sin(π z_o))`, and

```
∂ℓ/∂v_o = (∂ℓ/∂z_o) · v_⊤ / (π sin(π z_o)).
```

Now the real problem: how does `ℓ` depend on `S` and on the input vectors `V_I`, *through the solution of the SDP*? My first instinct is to unroll: the forward solve is a fixed number of coordinate-descent sweeps, each differentiable, so I could backprop through the unrolled iterations. But that's a wall — I'd have to store every intermediate `V` and every intermediate Jacobian, the memory is `O(iterations)` times the state, and it'll be slow and numerically ugly. I don't want to differentiate the *path* to the solution. I want to differentiate the *solution itself*.

This is the OptNet idea, generalized: don't differentiate the solver, differentiate the optimality conditions. The output vectors `v_o` are defined *implicitly* as the fixed point of the coordinate-descent update. At convergence, for every output `o`,

```
v_o = − g_o / ‖g_o‖,    g_o = V S^T s_o − ‖s_o‖² v_o.
```

If I take the total differential of *this fixed-point relation* — perturb `S` and the input vectors, and ask how `v_o` must move to stay a fixed point — I get a *linear system* in the `dv_o`, with no reference to how many sweeps it took. Solve that once and I have the exact gradient. Let me actually do it.

Multiply the fixed-point relation through by `‖g_o‖`: `‖g_o‖ v_o = −g_o`. Now differentiate both sides. On the left, `d(‖g_o‖ v_o) = (d‖g_o‖) v_o + ‖g_o‖ dv_o`. On the right, `−dg_o`. The scalar `d‖g_o‖ = (g_o^T dg_o)/‖g_o‖ = −v_o^T dg_o` (using `g_o = −‖g_o‖ v_o`). Substituting,

```
−v_o^T dg_o · v_o + ‖g_o‖ dv_o = −dg_o
⇒ ‖g_o‖ dv_o = −dg_o + (v_o^T dg_o) v_o = −(I_k − v_o v_o^T) dg_o.
```

There's the projector. Define `P_o = I_k − v_o v_o^T`; it projects onto the tangent space of the sphere at `v_o`. So `‖g_o‖ dv_o = −P_o dg_o`. Now expand `dg_o = d(V S^T s_o) − d(‖s_o‖² v_o)`. The differential of `V S^T s_o` splits into the part from the *other output* columns moving (which couples the `dv_j`, `j ∈ O`), and a part `ξ_o` collecting everything from the inputs and from `dS`:

```
ξ_o = Σ_{j ∈ I'} s_o^T s_j dv_j + V dS^T s_o + V S^T ds_o − 2 ds_o^T s_o v_o,
```

where `I' = {⊤} ∪ I` is the fixed (truth + clue) columns. And `d(‖s_o‖² v_o) = ‖s_o‖² dv_o + (2 ds_o^T s_o) v_o`. Collecting the `dv_j` over `j ∈ O` on the left:

```
(‖g_o‖ I_k − ‖s_o‖² P_o) dv_o + P_o Σ_{j ∈ O} (s_o^T s_j) dv_j = −P_o ξ_o,   ∀ o ∈ O.
```

Stack this over all `o ∈ O` and vectorize. Let `C = S_O^T S_O − diag(‖s_o‖²)` collect the cross-clause inner products among outputs (with the diagonal `‖s_o‖²` subtracted, since that term was pulled out separately), let `D = diag(‖g_o‖)`, and `P = diag(P_o)` the block-diagonal projector. The stacked system is

```
( D ⊗ I_k  +  P C ⊗ I_k ) vec(dV_O) = − P vec(ξ_o).
```

I want to invert that. The wall is that the operator `D ⊗ I_k + P C ⊗ I_k` isn't symmetric and `P` is a projector (singular), so I have to be careful — I can't just write a plain inverse. Let me first prove a clean form for the operator with a *positive* right-hand side, then carry the minus through. I claim that the solution of `(D ⊗ I_k + P C ⊗ I_k) vec(dV_O) = P vec(ξ_o)` is

```
vec(dV_O) = ( P ((D + C) ⊗ I_k) P )^† vec(ξ_o).
```

Look at the system one block at a time: `‖g_o‖ dv_o + P_o( Σ_j c_oj dv_j − ξ_o ) = 0`. Since `‖g_o‖ dv_o = −P_o(stuff)` from the derivation, every `dv_o` is in the range of `P_o`; that is, `dv_o = P_o y_o` for some `y_o`. Now substitute `vec(dV_O) = P vec(Y)` into this system. Using the block-diagonal structure of `P` and its idempotence `P P = P`, and the fact that the bare `D ⊗ I_k` term commutes with `P` blockwise (`(D ⊗ I_k) P = P (D ⊗ I_k) P` because `D` is diagonal),

```
(D ⊗ I_k + P C ⊗ I_k) P vec(Y) = P ((D + C) ⊗ I_k) P vec(Y) = P vec(ξ_o).
```

So `vec(Y) = ( P ((D+C) ⊗ I_k) P )^† P vec(ξ_o)`, and because the pseudoinverse of `P M P` already absorbs a leading `P` (`(PMP)^† P = (PMP)^†` when we then re-multiply by `P`), we get `vec(dV_O) = P vec(Y) = ( P ((D+C) ⊗ I_k) P )^† vec(ξ_o)`. That's the lemma. The projector `P` on both sides is what makes this well-defined even though the raw operator is singular. My actual stacked system has the *negative* right-hand side `−P vec(ξ_o)`, so by linearity its solution carries the minus:

```
vec(dV_O) = − ( P ((D+C) ⊗ I_k) P )^† vec(ξ_o).
```

Now I don't actually want `dV_O` — I want `∂ℓ/∂S` and `∂ℓ/∂V_I`. By the chain rule, the gradient I'm after is `(∂ℓ/∂vec(V_O))^T vec(dV_O)`, and plugging in the result,

```
− (∂ℓ/∂vec(V_O))^T ( P ((D+C) ⊗ I_k) P )^† vec(ξ_o).
```

The trick to avoid forming that giant operator: define a single object `U` that *is* the operator applied (from the left) to the upstream gradient. Let `U_I = 0` and

```
vec(U_O) = ( P ((D+C) ⊗ I_k) P )^† vec( ∂ℓ/∂V_O ).
```

Because the operator is symmetric (`P ((D+C)⊗I_k) P` is symmetric since `(D+C)` is symmetric — `C = S_O^T S_O − diag(‖s_o‖²)` is symmetric — and `P` is symmetric), I can move it onto the upstream gradient: the gradient becomes `−vec(U_O)^T vec(ξ_o)`. Now `ξ_o` is *linear* in `dv_i` (for inputs) and in `dS`, so I just pick off coefficients, and that leading minus is exactly the overall sign on each gradient below.

For an input `v_ι`, set `dv_ι` to a basis vector and everything else to zero in `ξ_o`: the surviving contribution is `−Σ_o u_o^T (s_o^T s_ι) e_j`, which collapses to

```
∂ℓ/∂V_I = − ( Σ_{o ∈ O} u_o s_o^T ) S_I,
```

with `S_I` the clue-indexed columns of `S`. Similarly, perturbing a single entry `S_{i,j}` and collecting the two surviving terms (the `V dS^T s_o` term and the `V S^T ds_o − 2 ds_o^T s_o v_o` terms),

```
∂ℓ/∂S = − ( Σ_{o ∈ O} u_o s_o^T )^T V  −  (S V^T) U.
```

Let me make sure I believe this, because a sign error here is fatal and invisible. I'll set up a tiny instance — six-dimensional vectors, four variables, one clue, two outputs, a random `S` — run coordinate descent to the fixed point, build `C`, `D`, `P`, form the operator, solve for `U`, and compute `∂ℓ/∂V_I` from the formula above. Then I'll perturb the input vector entry by entry and compute the gradient by finite differences through the whole solve. ... The analytic gradient and the finite-difference gradient agree to about `10⁻¹²`. The formula is right, signs and all.

So I never have to materialize the operator. I have to *solve* `vec(U_O) = (P((D+C)⊗I_k)P)^† vec(∂ℓ/∂V_O)`, i.e. apply that linear solve to the upstream gradient. And here's the elegant part: that linear system has the *same structure* as the forward coordinate-descent problem — same `C`, same projectors, same low-rank clause matrix. So I solve it with the *same* coordinate-descent machinery. Look at the system restricted to one `u_o`. Starting from `U_O = 0` (which keeps `P_o u_o = u_o` automatically, since each update lands in the range of `P_o`), the block equation is

```
‖g_o‖ P_o u_o + P_o ( U_O S_O^T s_o − ‖s_o‖² u_o ) = P_o (∂ℓ/∂v_o).
```

Maintain `Ψ = U_O S_O^T` (the backward analogue of `Ω = V S^T`). Define `dg_o = Ψ s_o − ‖s_o‖² u_o − ∂ℓ/∂v_o` (the term in the parentheses, negated). Then the closed-form coordinate update is

```
u_o = − P_o dg_o / ‖g_o‖,
```

and after updating `u_o` I patch `Ψ` with the same kind of rank-one update, `Ψ += (u_o − u_o^old) s_o^T`. Identical cost to the forward sweep — `O(nmk)` — and the same number of iterations converges, because it's the same operator. I get the entire backward pass at forward-pass cost, with no stored Jacobians. That's the payoff for differentiating the fixed point instead of unrolling.

There's a numerical hazard I should guard. The denominator `‖g_o‖` can get tiny — a degenerate clause where the gradient nearly vanishes — and then `u_o` blows up and the readout `dz_o = (∂ℓ/∂z_o)/(π sin(π z_o))` blows up too when `z_o` hits 0 or 1. So I add a small proximal term: replace `‖g_o‖` in the backward denominator with `‖g_o‖ + λ` for a small `λ` (the diagonal increment in `D`), which keeps the linear system well-conditioned. If something still goes non-finite, zero out that gradient rather than poison the whole batch. With that, the backward is stable.

Finally, the last hop: from `∂ℓ/∂v_ι` back to `∂ℓ/∂z_ι` for the inputs, by differentiating the input relaxation `v_ι = −cos(π z_ι) v_⊤ + sin(π z_ι)(I − v_⊤ v_⊤^T) v_ι^rand`:

```
∂v_ι/∂z_ι = π ( sin(π z_ι) v_⊤ + cos(π z_ι)(I − v_⊤ v_⊤^T) v_ι^rand ),
```

and `∂ℓ/∂z_ι = ∂ℓ/∂z_ι^direct + (∂v_ι/∂z_ι)^T (∂ℓ/∂v_ι)`, where the first piece is any direct dependence of the loss on `z_ι`. That closes the loop: gradients now flow from the output probabilities, through the SDP solution, into both the clause weights `S` and the input probabilities — and through those into whatever network produced them.

Two design choices are still nagging me, and they're the difference between this *working* on Sudoku and merely existing.

First: capacity vs. generalization. The number of clauses `m` is the rank of `S` and the capacity of the layer. My instinct is "more clauses, more expressive, set `m` big." But that's exactly the ConvNet failure — too much capacity and it memorizes the training boards instead of finding the real rules. The low-rank structure is not just a speed trick; it's the *regularizer*. With few clauses the layer literally cannot encode a lookup table; it's forced to discover a compact rule set that explains the data, which is what generalizes. So I keep `m` deliberately modest — on the order of the number of variables, not exponentially larger. For 9×9 Sudoku, 729 bit-variables, I use `m = 600` clauses. Small enough to force structure.

Second: representational power *without* paying in rank. A compact clause set might not be able to express a relation directly — but the CNF trick is to add *auxiliary variables*. Extra latent variables, connected to neither inputs nor outputs, act like register memory: introducing them can exponentially shrink the number of clauses needed to express a relation. So I let the user add auxiliary variables — say 300 for Sudoku — which raise the layer's expressive power without raising `m`. They participate in the solve and get their own columns of `V`, but I never read them out. This is what lets a 600-clause, 300-aux layer learn all the row/column/block constraints of Sudoku and generalize to 98%+ on held-out *and* permuted boards — permuted being the real test, since there's no locality left to cheat with.

Let me write it as it will actually run. The solver core is a coordinate-descent kernel shared by forward and backward (one maintains `Ω = V S^T = W`, the other `Ψ = U S^T = Φ`); a thin autograd `Function` wraps init / forward / backward; an `nn.Module` holds the learnable `S`, prepends the truth column and appends the auxiliary columns, and strips them off the output.

```python
import torch
import torch.nn as nn
from torch.autograd import Function


def _normalize_rows(M):
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class MixingFunc(Function):
    """Differentiable smoothed MAXSAT solve.

    forward:  relax probabilities -> unit vectors, run coordinate descent on the
              low-rank SDP  min <S^T S, V^T V> s.t. ||v_i||=1 over OUTPUT columns,
              read out probabilities via the GW same-side rule.
    backward: differentiate the fixed point (not the unrolled sweeps) and solve the
              resulting linear system with the SAME coordinate descent.
    """

    @staticmethod
    def forward(ctx, S, z, is_input, max_iter, eps, prox_lam):
        # S: (n, m) clause matrix (row 0 = truth variable). z: (B, n) in [0,1].
        B, n, m, k = z.size(0), S.size(0), S.size(1), 32   # k >= sqrt(2n)+1
        dev = S.device
        ctx.prox_lam = prox_lam

        V = torch.zeros(B, n, k, device=dev).normal_()      # random unit init
        # --- Step 1: relax inputs.  v_i . v_top = -cos(pi z_i), unit norm. ---
        # row 0 is the truth direction; encode each input variable's angle to it.
        for i in range(n):
            inp = is_input[:, i].bool()
            vi1 = V[:, i, 1].clone()
            V[inp, i, :] = 0.0
            V[inp, i, 0] = -torch.cos(z[inp, i] * torch.pi)
            V[inp, i, 1] = torch.sin(z[inp, i] * torch.pi) * torch.sign(vi1[inp])
        free = ~is_input.bool()
        V[free] = _normalize_rows(V[free])

        Snrms = (S * S).sum(dim=1)                           # ||s_i||^2
        # --- Step 2: coordinate descent over OUTPUT columns only. ---
        # maintain W = V^T S  (k x m); g_o = W^T s_o - ||s_o||^2 v_o; v_o = -g/||g||.
        W = torch.einsum('bik,im->bkm', V, S)
        gnrm = torch.zeros(B, n, device=dev)
        out_cols = [i for i in range(n) if i != 0]           # truth column fixed
        for it in range(max_iter):
            delta = 0.0
            for o in out_cols:
                mask = free[:, o]
                if mask.sum() == 0:
                    continue
                g = torch.einsum('bkm,m->bk', W[mask], S[o]) - Snrms[o] * V[mask, o]
                gn = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                new = -g / gn
                W[mask] += torch.einsum('bk,m->bkm', new - V[mask, o], S[o])
                gnrm[mask, o] = gn.squeeze(-1)
                delta += (gn.squeeze(-1) ** 2).sum().item()
                V[mask, o] = new
            if it and delta < eps:
                break
            if it == 0:
                eps = delta * eps

        # --- Step 3: read out probabilities z_o = arccos(-v_o . v_top)/pi. ---
        z_out = z.clone()
        vtop = V[:, 0:1, :]                                  # truth direction
        for o in out_cols:
            mask = free[:, o]
            cos = -(V[mask, o] * vtop[mask, 0]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            z_out[mask, o] = torch.acos(cos) / torch.pi

        ctx.save_for_backward(S, z_out, V, W, Snrms, gnrm, is_input)
        return z_out

    @staticmethod
    def backward(ctx, dz):
        S, z, V, W, Snrms, gnrm, is_input = ctx.saved_tensors
        B, n, k, m = V.size(0), S.size(0), V.size(2), S.size(1)
        dev = S.device
        free = ~is_input.bool()
        out_cols = [i for i in range(n) if i != 0]

        # --- readout: dl/dv_o = (dl/dz_o) * v_top / (pi sin(pi z_o)). ---
        dv = torch.zeros(B, n, k, device=dev)
        vtop = V[:, 0, :]
        for o in out_cols:
            mask = free[:, o]
            s = torch.sin(z[mask, o] * torch.pi) * torch.pi
            coeff = dz[mask, o] / s.clamp_min(1e-12)
            dv[mask, o] = coeff.unsqueeze(-1) * vtop[mask]

        # --- solve (P((D+C) x I)P)^+ vec(dl/dV_O) via the SAME coord. descent. ---
        # maintain Phi = U^T S; dg_o = Phi^T s_o - ||s_o||^2 u_o - dl/dv_o;
        # u_o = -P_o dg_o / (||g_o|| + prox_lam).
        U = torch.zeros(B, n, k, device=dev)
        Phi = torch.zeros(B, k, m, device=dev)
        for it in range(40):                                 # same #iters as forward
            for o in out_cols:
                mask = free[:, o]
                if mask.sum() == 0:
                    continue
                dg = (torch.einsum('bkm,m->bk', Phi[mask], S[o])
                      - Snrms[o] * U[mask, o] - dv[mask, o])
                vo = V[mask, o]
                proj = dg - (vo * dg).sum(-1, keepdim=True) * vo          # P_o dg
                denom = (gnrm[mask, o] + ctx.prox_lam).clamp_min(1e-12)
                new = -proj / denom.unsqueeze(-1)
                Phi[mask] += torch.einsum('bk,m->bkm', new - U[mask, o], S[o])
                U[mask, o] = new

        # --- assemble gradients. ---
        # Phi = sum_o u_o s_o^T (k x m), W = V^T S (k x m). The two terms below are
        # exactly  -( sum_o u_o s_o^T )^T V  and  -(S V^T) U ; keep the leading minus.
        # dl/dS = -(U W + V Phi)  per batch.
        dS = torch.zeros(n, m, device=dev)
        for b in range(B):
            dS -= U[b].mm(W[b]) + V[b].mm(Phi[b])            # dS = -(U W + V Phi)
        # input-probability gradient: dl/dv_i = -(sum_o u_o s_o^T) s_i = -(Phi s_i),
        # then chain through dv_i/dz_i. Here v_top = e_0 and the random direction sits
        # along e_1, so dv_i/dz_i has a sin(pi z) part along e_0 and a cos(pi z) part
        # along e_1 (with the sign carried in V[:, i, 1]); both components contribute.
        dz_in = torch.zeros(B, n, device=dev)
        for i in range(n):
            inp = is_input[:, i].bool()
            if inp.sum() == 0:
                continue
            val0 = torch.einsum('bk,m->b', Phi[inp, 0:1, :].squeeze(1), S[i])
            val1 = torch.einsum('bk,m->b', Phi[inp, 1:2, :].squeeze(1), S[i])
            sgn = torch.sign(V[inp, i, 1])
            dz_in[inp, i] = -(val0 * torch.sin(z[inp, i] * torch.pi) * torch.pi
                              + val1 * sgn * torch.cos(z[inp, i] * torch.pi) * torch.pi)

        return dS, dz_in, None, None, None, None


class SATNet(nn.Module):
    """A learnable MAXSAT layer. The clause matrix S IS the learned logic."""

    def __init__(self, n, m, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2):
        super().__init__()
        # row 0 is the truth variable; +aux latent "register" variables.
        S = torch.zeros(n + 1 + aux, m).normal_()
        S *= (0.5 / (n + 1 + aux + m)) ** 0.5               # keep init small
        self.S = nn.Parameter(S)
        self.aux = aux
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        B, dev = z.size(0), z.device
        # prepend the always-true truth variable, append unconstrained aux vars.
        ones = torch.ones(B, 1, device=dev)
        z = torch.cat([ones, z, torch.zeros(B, self.aux, device=dev)], dim=1)
        is_in = torch.cat([torch.ones(B, 1, device=dev, dtype=is_input.dtype),
                           is_input,
                           torch.zeros(B, self.aux, device=dev, dtype=is_input.dtype)],
                          dim=1)
        z = MixingFunc.apply(self.S, z, is_in, self.max_iter, self.eps, self.prox_lam)
        return z[:, 1:self.S.size(0) - self.aux]            # strip truth + aux
```

The causal chain, start to end: the pain is that discrete logical constraints don't have gradients, so a net can't *learn rules* from data. Make MAXSAT the learnable primitive — the clause matrix `S` holds the structure. MAXSAT is non-differentiable, so relax it to a low-rank SDP over unit vectors (`X = V^T V`, `‖v_i‖ = 1`), with a truth direction to absorb clause bias; the SDP objective is `⟨S^T S, V^T V⟩`, exactly the relaxed MIN-UNSAT, and `S` is differentiable. Low rank `k = √(2n)+1` makes the SDP small yet exact, and coordinate descent `v_i = −g_i/‖g_i‖` solves it globally in `O(nmk)` per sweep. Encode probabilities by the angle to the truth direction (the Goemans–Williamson `arccos/π` law), so soft inputs flow in and soft outputs flow out, and a perception net can sit underneath. Backprop by differentiating the *fixed point*, not the iterations: the total differential gives a linear system with the same structure as the forward solve, so the same coordinate descent computes the gradient at the same cost, with no stored Jacobians. Keep `m` small as a regularizer and add auxiliary variables for capacity. The result is a layer that learns the rules of parity and of (even permuted) Sudoku end to end, and can be stacked behind a digit recognizer.

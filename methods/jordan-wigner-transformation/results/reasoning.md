Let me set the problem cleanly first, because the temptation is to start writing matrices and I want to know what I'm actually after. We have quantized the radiation field and we have quantized the Bose matter gas. In both cases the trick is the same and it is beautiful: you take the wave amplitude on ordinary three-dimensional space, you promote it to a *q*-number, and then the discreteness of the particles and their statistics are no longer separate postulates — they are consequences of the way the amplitude multiplies. For light: an oscillator field, creation and annihilation operators, the number operator with eigenvalues `0,1,2,…`, photons fall out. For the Einstein gas: amplitudes `b_r`, conjugate phases `Θ_r`, the number `N_r = b_r^† b_r`, the commutator `b_r b_r^† − b_r^† b_r = 1`, and the eigenvalues come out `0,1,2,3,…` again, and the density fluctuation comes out proportional to `n_r(1+n_r)` exactly as Einstein found. So the bosonic case is done and I understand it from the inside.

Now I want the same thing for a gas that obeys Pauli's exclusion principle. And the very first thing I notice is that the bosonic machine, copied verbatim, gives the wrong answer. The reason is structural, not a matter of tuning. `[b,b^†]=1` forces the spectrum of `N=b^†b` to be the non-negative integers — that is a theorem about the harmonic-oscillator algebra, there is no knob in it. But for electrons I need the occupation of every mode to be only `0` or `1`. And there is a second thing, deeper than the number-capping: Pauli's principle in its original form is that the many-electron wave function is *antisymmetric* — swap two electrons and you pick up a minus sign. The bosonic amplitudes on different modes *commute*, `b_α b_β − b_β b_α = 0`, and commuting things give a *symmetric* state space. So I have two failures from the same source: the spectrum is wrong and the exchange sign is wrong. What I want is a single algebra of amplitudes `a_r`, `a_r^†` from which *both* the cap at 1 and the antisymmetric exchange drop out automatically — the way particle-existence drops out of the boson algebra. If I can do that, then the existence of corpuscular electrons and the validity of the Pauli principle become *consequences* of the multiplication rules of the de Broglie wave amplitudes, not things I assume. That is the prize.

Let me also pin Pauli's fluctuation result, because it's a clean target to hit at the end. The mean square density fluctuation for the Fermi gas is proportional to `n_r(1 − n_r)`. Same algebraic shape as the boson's `n_r(1 + n_r)` but with a flipped sign in the parenthesis, and that `(1 − n_r)` is the fingerprint of single occupancy — it vanishes the instant the mode is full, which is exactly "you can't add a second one." So whatever I build, the matrix element factors should turn the boson `(1 + N)` into a Fermi `(1 − N)`.

Start with one mode, since the difficulty surely isn't there. A mode that can hold `0` or `1` particle is a two-state system. I want operators with `N` having eigenvalues `{0,1}`. The smallest realization is two-by-two:
```
b = (0 1 ; 0 0),   b^† = (0 0 ; 1 0).
```
Then `b^† b = (0 0 ; 0 1)`, which is `N` with eigenvalues `0` and `1` — good, capped automatically. And `b b^† = (1 0 ; 0 0) = 1 − N`. So already, for one mode,
```
b^† b = N,     b b^† = 1 − N,     b^2 = 0,     (b^†)^2 = 0.
```
`b^2 = 0` is lovely: it *is* "you cannot put two particles in one mode," built straight into the algebra. And `b b^† = 1 − N` instead of `1 + N` — there is my flipped sign, the `(1 − N)` factor, sitting in the single-mode algebra already. So the one-mode problem solves itself. Notice what these matrices are: they're the spin-1/2 ladder operators, `b = σ^-` the lowering operator, `b^† = σ^+` the raising one (taking the occupied mode `N=1` to be spin up, so adding/removing a particle lowers/raises `S^z`), and the parity `1 − 2N = diag(1, −1) = −σ^z` (it is `+1` on the empty state, `−1` on the full one). A single Fermi mode and a single spin are literally the same two-level object. I'll keep that in my back pocket; it might be the whole story in disguise.

Let me write the one-mode relations in the form the boson theory used, with the conjugate phase, just to keep the analogy tight. I want `b` to look like `e^{(2πi/h)Θ}·(\text{something in }N)`. Working it out for the two-state matrices: with `Θ = (h/4)(0 1 ; 1 0)` one gets `e^{(2πi/h)Θ} = i·(h/4)^{-1}Θ`-type relation, and indeed
```
b^† b = N,   b b^† = 1 − N,   N^2 = N,   (b^†)^2 = (b)^2 = 0,   e^{(2πi/h)Θ} = i(4/h)Θ.
```
`N^2 = N` is just "eigenvalues are 0 or 1" restated. And I can package the three two-by-two objects with the quaternion units `k_1, k_2, k_3` (the Pauli matrices in disguise): `k_1 k_2 = −k_2 k_1 = k_3`, `k_i^2 = −1`, with `b = −(ik_3 + k_2)/2`, `b^† = −(ik_3 − k_2)/2`, `N = −(ik_1 − 1)/2`. Fine — the single mode is the quaternion/Pauli algebra and there is genuinely nothing more to discover there.

So the entire difficulty is combining `K` modes. The obvious move — and the one I have to try first because it's what worked for bosons — is to make the `K`-mode amplitude by tensoring: `b_r = 1 ⊗ ⋯ ⊗ b ⊗ ⋯ ⊗ 1`, the single-mode `b` sitting in slot `r`. Let me check what statistics that gives. Two different modes `r ≠ s`: `b_r` acts on slot `r`, `b_s` on slot `s`, disjoint slots, so they *commute*: `b_r b_s = b_s b_r`. Hmm. That's the boson exchange, the symmetric one. I built the per-mode cap correctly (each `b_r^2 = 0`) but the cross-mode statistics came out symmetric. So the naive tensor product gives me a state space that is capped at one-per-mode but *symmetric under exchange* — which is not Fermi statistics, it's something incoherent. I've hit the wall I expected: the cap is local and easy, the exchange sign is global and is where the real content lives.

Where does the minus sign even live? It is not in any single mode — within one mode there's nothing to exchange. It lives in the *many-electron wave function*, which is antisymmetric: the Heisenberg-Dirac determinant
```
Ψ = (1/√N!) Σ_perm ε_n  Φ(β^(1), q^(n_1)) Φ(β^(2), q^(n_2)) ⋯ ,    ε_n = ±1 for even/odd perm.
```
This vanishes whenever two of the `β`'s coincide — that *is* the Pauli principle, "no two particles in the same state." Good, but I want to read off how the sign enters when I *add or remove* one particle, because that's what `a_r^†` and `a_r` do. And here is the subtle thing about a determinant: it has a sign ambiguity. To even *define* it single-valuedly I have to fix, once and for all, an ordering of the single-particle states — call it `β'_1 < β'_2 < ⋯ < β'_K`, where `<` is just a chosen order, not a magnitude. With the order fixed, the antisymmetric amplitude becomes a single-valued function of the occupation numbers,
```
Ψ(N'(β'_1), N'(β'_2), …, N'(β'_K)),    each N'(β'_k) ∈ {0, 1},
```
and `Ψ` is built by reading the determinant with its arguments placed in that fixed order. So far so good — the occupation-number representation exists, `2^K` basis states for `K` modes, each `N' = 0` or `1`.

Now the decisive question. I want the operator `a_λ` that removes a particle from mode `λ`. In the occupation-number basis it should take the state with `x_λ = 1` to the state with `x_λ = 0`, leaving the other occupations alone. But what is the coefficient? Because the determinant cares about *order*, to remove the particle in mode `λ` I have to first commute its `Φ` factor — through antisymmetry — past all the factors sitting to its left in the fixed ordering, and each such hop is a transposition that costs a minus sign. The factors to its left are exactly the occupied modes with index below `λ`. So the coefficient is `(−1)` raised to the number of occupied modes to the left of `λ`. Let me write the matrix element honestly, with `x = (x_1,…,x_K)` the bra occupations and `y = (y_1,…,y_K)` the ket occupations:
```
a_λ(x; y) = (−1)^{x_1 + x_2 + ⋯ + x_{λ−1}} · δ_{x_1 y_1} ⋯ δ_{x_{λ−1} y_{λ−1}} · δ_{x_λ, 1} δ_{y_λ, 0} · δ_{x_{λ+1} y_{λ+1}} ⋯ δ_{x_K y_K}.
```
Read it: all modes except `λ` are unchanged (`δ_{x_k y_k}`); on mode `λ` it requires the ket full (`y_λ=1`) and the bra empty (`x_λ=0`) — wait, let me be careful with the convention, `a` annihilates so the *ket* should have the particle and the *bra* not; I'll fix the orientation so `δ_{y_λ,1}, δ_{x_λ,0}` for `a`, and the creation operator gets the mirror; the *sign* is the thing that matters and it is `(−1)^{Σ_{k<λ} (\text{occupation})}`. And the creation operator:
```
a_λ^†(x; y) = (−1)^{x_1 + ⋯ + x_{λ−1}} · δ_{x_1 y_1} ⋯ δ_{x_{λ−1} y_{λ−1}} · δ_{x_λ, 0} δ_{y_λ, 1}... wait —
```
let me just fix it as: `a_λ^†` takes empty `λ` to full `λ` (`y_λ=0 → x_λ=1`) with the same left-parity sign. The point I want to nail is where the `(−1)^{Σ_{k<λ}}` comes from, and it comes from the determinant's reordering: to insert or extract the `λ`-th factor in the fixed order I jump it past the occupied factors to its left, an even or odd number of transpositions, "+ or − according to whether between the written-out `0` and `1` there stand an even or odd number of `1`'s." That sentence is the whole transformation in words.

So I have it as an explicit matrix. But matrices indexed by `2^K` bit-strings are unwieldy and they hide the structure. Let me re-express that sign factor as an *operator*. The number of occupied modes to the left of `λ` is `Σ_{k<λ} N_k`. The sign is `(−1)` to that power, which is the operator
```
∏_{k<λ} (−1)^{N_k} = ∏_{k<λ} (1 − 2 N_k),
```
because for a two-state mode `(−1)^{N} = 1 − 2N` (it's `+1` on the empty state, `−1` on the full one — check: `N=0 → 1`, `N=1 → 1−2 = −1`, yes). And remember `1 − 2N = −σ^z` from the single-mode identification. So the sign operator to the left of `λ` is `∏_{k<λ}(1 − 2N_k) = ∏_{k<λ}(−σ^z_k)`. This is the string. Let me therefore *define* the Fermi amplitude as the bare two-level lowering operator on mode `λ` *dressed* with this string:
```
a_λ = [∏_{k<λ} (1 − 2 N_k)] · b_λ,        a_λ^† = b_λ^† · [∏_{k<λ} (1 − 2 N_k)].
```
Equivalently, collecting the string into one symbol, write `v(λ) = ∏_{k ≤ λ}(1 − 2N_k)` (the convention of including the mode itself is harmless because the bare `b_λ` already handles mode `λ`; I'll use the inclusive product and check it works): `a_λ = v(λ)·b_λ`, `a_λ^† = b_λ^†·v(λ)`. First a sanity property: each factor `1 − 2N_k` has eigenvalues `±1`, so
```
v(λ)^2 = 1.
```
It's an involution — a parity operator. Good, that's exactly what a sign-carrier should be.

Now the real test: do these `a`, `a^†` *anticommute* across modes, as Fermi statistics demands? The mechanism has to be that the string of one operator, passing through the bare ladder operator of another, flips a sign. Let me compute the basic move: `b_p(q')` against the string `v(q'')`. The string `v(q'') = ∏_{q''' ≤ q''}(1 − 2N_{q'''})` contains the factor `(1 − 2N_{q'})` exactly when `q' ≤ q''`. Does `b` commute with `1 − 2N` on its own mode? Check it: on mode `q'`, `b(1 − 2N) = b − 2bN`, and `bN = b·b^†b = (1−N)... ` — let me just use `Nb = b·0`? Careful. `N b = b^† b b = b^†·0 = 0` since `b^2=0`, while `b N = b b^† b = (1−N)b = b − Nb = b`. So `bN = b`, `Nb = 0`, hence `b(1 − 2N) = b − 2b = −b`, and `(1 − 2N)b = b − 0 = b`. So `b` *anticommutes* with `(1 − 2N)` on its own mode: `b(1−2N) = −(1−2N)b`. That single anticommutation is what will drive everything. Moving `v(q'')` past `b_p(q')`:
```
b_p(q') v(q'') = − v(q'') b_p(q')   if  q' ≤ q''   (the string contains mode q', one sign flip),
b_p(q') v(q'') = + v(q'') b_p(q')   if  q' > q''   (the string does not reach mode q', they commute).
```
With that, I can verify the canonical relations. First, the consistency of `b^†` with its own string:
```
b_p^†(q') · {1 − 2N(q')} = − {1 − 2N(q')} · b_p^†(q'),
```
same anticommutation (since `b^†N = 0`, `Nb^† = b^†`, giving `b^†(1−2N) = −b^† = −(1−2N)b^†`). Good.

Now anticommutation of two annihilation operators, `a_p(q')` and `a_p(q'')`. Take `q' ≤ q''` for definiteness (the other case is symmetric):
```
a_p(q') a_p(q'') = v(q') b_p(q') · v(q'') b_p(q'').
```
I need to slide `b_p(q')` to the right past `v(q'')`. Since `q' ≤ q''`, by the rule above `b_p(q') v(q'') = − v(q'') b_p(q')`. So
```
a_p(q') a_p(q'') = − v(q') v(q'') b_p(q') b_p(q'').
```
The strings `v(q'), v(q'')` are products of commuting `(1−2N)`'s, so they commute with each other and `v(q')v(q'') = v(q'')v(q')`. Now do the same for the reversed product `a_p(q'') a_p(q')`, with `q'' ≥ q'`: sliding `b_p(q'')` past `v(q')` — but now `q'' > q'` would give `+`; careful, I have to compare the *same* ordering. Let me just compute `a_p(q'')a_p(q')` directly: `= v(q'')b_p(q'')·v(q')b_p(q')`. Slide `b_p(q'')` past `v(q')`: is `q'' ≤ q'`? No, `q'' ≥ q'`, and if `q'' > q'` the rule gives commute (`+`), so `b_p(q'')v(q') = + v(q')b_p(q'')`, giving `a_p(q'')a_p(q') = + v(q'')v(q') b_p(q'')b_p(q')`. Now the bare two-level operators on different modes commute, `b_p(q')b_p(q'') = b_p(q'')b_p(q')`, and the strings are equal. So
```
a_p(q') a_p(q'') = − v v b_p(q') b_p(q'') = − v v b_p(q'') b_p(q') = − a_p(q'') a_p(q').
```
There it is:
```
a_p(q') a_p(q'') = − a_p(q'') a_p(q').
```
And for the same mode `q' = q''`: `a_p(q')^2 = v(q')b_p(q')v(q')b_p(q')`, and `b_p(q')v(q') = ±v(q')b_p(q')` with the sign immaterial because what kills it is `b_p(q')^2 = b^2 = 0`. So `a_p(q')^2 = 0`. Single occupancy is preserved. The anticommutator `{a_p(q'), a_p(q'')} = 0` for all `q', q''`, including the diagonal. The same computation with one dagger gives the mixed relation; let me do the important one, the canonical anticommutation that defines the whole thing:
```
a_p^†(q') a_p(q'') + a_p(q'') a_p^†(q') = δ(q' − q'').
```
Off-diagonal `q' ≠ q''`: the same string-flip logic makes the two orderings cancel, exactly as for `aa`, so the anticommutator is `0`. On the diagonal `q' = q''`: the strings square to `1` (`v^2=1`), they wash out, and I'm left with the *single-mode* relation `b^† b + b b^† = N + (1 − N) = 1`. So `{a_p^†(q'), a_p(q')} = 1`. Combining:
```
a_p^†(q') a_p(q'') + a_p(q'') a_p^†(q') = δ(q' − q'').
```
These three — `{a,a}=0`, `{a^†,a^†}=0`, `{a^†,a}=δ` — are the canonical anticommutation relations, and they came out *forced* by the determinant's reordering sign. I did not postulate antisymmetry as an extra rule; I postulated the string, which was itself nothing but the exchange sign written as an operator, and antisymmetry came out as the algebra of the dressed amplitudes.

Now I should close the loop in the other direction, because the real claim of the program is that the algebra *implies* the physics, not the other way around. Suppose I only know the canonical anticommutation relations. Do the eigenvalues of `N_r = a_r^† a_r` come out `{0,1}`, and do different `N`'s commute? From `a_r^† a_r + a_r a_r^† = 1` and `a_r^2 = 0`:
```
N_r (1 − N_r) = a_r^† a_r (1 − a_r^† a_r) = a_r^† a_r · a_r a_r^†   [using 1 − a_r^† a_r = a_r a_r^†]
              = a_r^† (a_r a_r) a_r^† = a_r^† · 0 · a_r^† = 0,
```
so `N_r^2 = N_r` — eigenvalues `0` or `1`, the Pauli cap, *derived*. And `N_r N_s − N_s N_r = 0` follows directly from the anticommutators (the cross terms cancel in pairs), so the occupation numbers are simultaneously measurable and the joint occupation-number basis exists. So the existence of corpuscular particles (a well-defined integer count per mode) and the validity of the Pauli principle (that integer is `0` or `1`) are *consequences* of the multiplication rules of the wave amplitudes — precisely the statement I wanted, the Fermi analogue of how it worked for the Bose field. The two facts I needed,
```
N_r N_s − N_s N_r = 0   and   N_r (1 − N_r) = 0,
```
are both contained in the anticommutation algebra.

Let me make sure the matrix-element factor matches Pauli's fluctuation. The amplitude `a_r` has the factor `(1 − N_r)` sitting in `a_r a_r^† = 1 − N_r` exactly where the boson had `(1 + N_r)` in `b_r b_r^† = 1 + N_r`. In the corrected field expansion `ψ = Σ_r a_r sin(rπx/l)` (replacing the boson `b_r`), the density-fluctuation calculation that gave `n_r(1+n_r)` for the boson now runs with `\overline{a_r^† a_r · a_r a_r^†} = \overline{N_r(1 − N_r)}`, so the mean square comes out proportional to `n_r(1 − n_r)` — Pauli's Fermi form, and the end result of the fluctuation calculation is unchanged in structure, just sign-flipped in the parenthesis. That's the consistency check passed.

Now I want to confront the full equivalence to the configuration-space theory, because asserting "the antisymmetric determinant maps to occupation numbers" is not the same as proving the *energy operator* maps correctly. Take a one-body energy `V = V_1 + V_2 + ⋯ + V_N`, the same operator `V_i` measured on each particle, with single-particle matrix `H_{νμ}` in the chosen single-particle basis. In the `N`-particle configuration space `V` is the symmetrized matrix
```
H_{ν_1…ν_N; μ_1…μ_N} = Σ_i H_{ν_i μ_i} ∏_{j ≠ i} δ_{ν_j μ_j},
```
i.e. it changes one particle's state and leaves the rest as Kronecker deltas. I claim that on the antisymmetric functions this is matrix-equivalent, under the map `Ψ(N') = (1/√N'!) ψ(β')` from determinant to occupation numbers, to the operator
```
Ω = Σ_{κ, λ} H_{κλ} a_κ^† a_λ
```
on the occupation-number space. Let me actually verify the sign works out, because this is exactly where the string earns its keep. Apply `V` to a determinant. Where the occupation pattern is unchanged it's automatic. Where it moves a particle from occupied mode `l` to empty mode `i`, I have to bring both factors into the fixed order, and the determinant picks up `(−1)^z` with `z` the number of `β`'s sitting between the two slots — "between the written-out `0` and `1`." Reading the configuration-space action of `V` in the occupation basis,
```
\overline{Ψ}(x_1…x_K) = Σ_{j: x_j=1, l: x_l=0} ± H_{jl} Ψ(x_1,…, x_j → 0, …, x_l → 1, …, x_K)  +  Σ_j H_{jj} Ψ(x),
```
where the sign is `+` or `−` according to whether between the written-out `0` and `1` there is an even or odd number of `1`'s — that count is exactly `(\text{number of } 1\text{'s left of } x_l) − (\text{number of }1\text{'s left of } x_j)`. But that is *precisely* the sign produced by `a_l^† a_j`: the string of `a_l^†` counts the occupied modes left of `l`, the string of `a_j` counts those left of `j`, and their product is `(−1)` to the difference. So term by term,
```
V ψ  ⟷  Σ_{κλ} H_{κλ} a_κ^† a_λ  Ψ = Ω Ψ,
```
the signs match because the string was *built from* this very reordering count. An eigenfunction of `V` maps to an eigenfunction of `Ω`. And inner products are preserved: `(ψ, φ)` in configuration space, with the antisymmetry pulling the permutation sum into a single ordered sum and the `N!` cancelling the `1/√{N!}` normalizations, equals `(Ψ, Φ)` in the occupation space, `Σ_{x} \overline{Ψ(x)} Φ(x)`. So the map is unitary and the two theories are the same theory. (As a check on internal consistency: the total number `N = Σ_{β'} N(β') = Σ_{q'} N(q')` is invariant under the basis change `a_α(β') = Σ_{q'} Φ_{αp}(β',q') a_p(q')`, which is just the statement that the change of single-particle basis is a unitary transformation, as it must be.)

I should pause on the basis-change law itself, because it is *not* the boson law and the difference is informative. For bosons one had simply `b_α(β') = Σ_{q'} Φ_{αp}(β',q') b_p(q')`. For the Fermi amplitudes that bare linear law does *not* hold, because the strings depend on the ordering and a change of basis reshuffles the ordering. What survives is
```
a_α(β') = Σ_{q'} Φ_{αp}(β', q') a_p(q'),
```
with the `a`'s the *dressed* (string-carrying) amplitudes on each side — the bare `b`'s transform with extra sign corrections, but the physical `a`'s transform linearly and unitarily, which is exactly what is needed for `N` to be invariant and the anticommutation relations to be preserved. Curiously the dressed `a, a^†` have a *tighter* analogy to the boson `b, b^†` than the bare `b`'s do, even though the `b`'s were the literal copies of the boson construction. The contrast is worth stating plainly:

Bose: `b_α(β') b_β(β'') − b_β(β'') b_α(β') = 0`,   `b_α^†(β') b_α(β') = N(β')`,   `b_α(β') = Σ Φ b_p(q')`.

Pauli: `a_α(β') a_β(β'') + a_β(β'') a_α(β') = 0`,   `a_α^†(β') a_α(β') = N(β')`,   `a_α(β') = Σ Φ a_p(q')`.

The only change in the structural relations is commutator `→` anticommutator. Everything else — the number operator, the unitary basis change, the energy `Σ H a^† a` — has the identical form. The whole of Fermi statistics is the single sign flip `[\,,\,] → \{\,,\,\}`, achieved by the string.

There is one more thing nagging at me. I claimed the anticommutation relations *determine* the amplitudes, but determine them how uniquely? I gave a particular `2^K`-dimensional matrix realization (the string construction). Could there be inequivalent ones? Let me prove uniqueness, because if the relations pin the operators down to a unitary change of basis, then "the algebra is the physics" is airtight. Form the Hermitian combinations
```
α_κ = a_κ + a_κ^†,        α_{K+κ} = (1/i)(a_κ − a_κ^†),     κ = 1,…,K.
```
There are `2K` of them, and they are Hermitian. Compute their products from the canonical anticommutation relations. For `κ < K, λ < K`:
```
α_κ α_λ + α_λ α_κ = (a_κ + a_κ^†)(a_λ + a_λ^†) + (a_λ + a_λ^†)(a_κ + a_κ^†)
= {a_κ,a_λ} + {a_κ,a_λ^†} + {a_κ^†,a_λ} + {a_κ^†,a_λ^†}
= 0 + δ_{κλ} + δ_{κλ} + 0 = 2 δ_{κλ}.
```
The same computation for the imaginary combinations and the cross terms gives, uniformly,
```
α_κ α_λ + α_λ α_κ = 2 δ_{κλ}    for all κ, λ = 1,…,2K.
```
This is the Clifford / Dirac relation — the same algebra as Dirac's gamma matrices. In particular
```
α_κ^2 = 1,     α_κ α_λ = − α_λ α_κ   for κ ≠ λ.
```
So the `2K` matrices `α` each square to the identity and anticommute pairwise. The `2K` matrices `α` determine the `a` uniquely (invert the linear combination), so I just need the uniqueness of the `α`'s. Consider the finite set of products of the `α`'s together with `−1`. Each `α_κ^2 = 1`, and any product reorders to `±` a product of distinct `α`'s, so the distinct elements are `± ∏_{κ ∈ S} α_κ` over subsets `S ⊆ {1,…,2K}`: that's `2 · 2^{2K} = 2^{2K+1}` elements, and they're closed under multiplication — a group of order `2^{2K+1}`. (For `K=2` you can list all `32` elements explicitly: `1; ±α_1, ±α_2, ±α_3, ±α_4; ±α_1α_2, …; ±α_1α_2α_3, …; ±α_1α_2α_3α_4` — `32` in total, matching `2^{2·2+1}=32`.) Its center is `{1, −1}` (these are the only elements commuting with all `α`'s, since for `K≥1` every nontrivial product anticommutes with some `α`). The factor group by the center has order `2^{2K}` and is abelian, so it has `2^{2K}` one-dimensional representations — but these send `−1 → +1`, i.e. they make the `α`'s commute, which violates `α_κ α_λ = −α_λ α_κ`, so none of them is a faithful representation of *our* relations. How many conjugacy classes does the group have? The elements `1` and `−1` are each their own class; every other element `R` is conjugate to `−R` (because some `α` anticommutes with it: `α R α^{-1} = −R` when `R` contains an odd number of `α` factors not equal to `α`, and one can always find such an `α`). So the classes are: `{1}`, `{−1}`, and `{R, −R}` for the `(2^{2K+1} − 2)/2 = 2^{2K} − 1` remaining pairs — total `2^{2K} + 1` classes, hence `2^{2K} + 1` inequivalent irreducible representations. I already have `2^{2K}` of them (the one-dimensional ones), so there is *exactly one* more. Its dimension `d` must satisfy the order-equals-sum-of-squares identity
```
2^{2K+1} = (\text{sum of squared dimensions}) = 2^{2K} · 1^2 + d^2,
```
so `d^2 = 2^{2K+1} − 2^{2K} = 2^{2K}`, giving `d = 2^K`. So there is a *unique* irreducible representation of dimension `2^K` in which the `α`'s genuinely anticommute — and that is exactly the dimension and the structure of the string construction I built (`2^K` occupation-number states). Any representation of the canonical anticommutation relations that is faithful is this one, up to a similarity (unitary) transformation. The amplitudes `a, a^†` are therefore fixed by their algebra alone, up to a canonical transformation. The algebra *is* the physics.

Let me now make this concrete on the testbed I'll want to display it on — a one-dimensional system, since that's where a lattice of two-level systems and the fermion field meet most cleanly. Take the one-dimensional continuum with `∂²ψ/∂x² = ∂²ψ/∂t²` and `ψ(0,t)=ψ(l,t)=0`, expanded in standing waves `ψ = Σ_r a_r sin(rπx/l)` with the `a_r` the Fermi amplitudes — the density `N(x) = ψ^† ψ`, `N_r = a_r^† a_r` the number in the `r`-th translational mode, and the fluctuation comes out `\overline{Δ²} ∝ n_r(1 − n_r)` as required. But the version that makes the *power* of the string vivid is the discrete one: a chain of two-level systems — spins. Recall each mode is a spin-1/2, `a` dressed `σ^-`, `1 − 2N = −σ^z`. Write the dictionary on a lattice site `j`:
```
S^z_j = N_j − 1/2,     S^+_j = a_j^† e^{ iπ Σ_{l<j} N_l},     S^-_j = a_j e^{ −iπ Σ_{l<j} N_l},
```
with `e^{iπ N} = 1 − 2N = −σ^z`, so the string `e^{iπ Σ_{l<j} N_l} = ∏_{l<j}(−σ^z_l)` is exactly my `v`. Inverting: `a_j = e^{iπ Σ_{l<j} N_l} σ^-_j`, `a_j^† = e^{−iπ Σ_{l<j} N_l} σ^+_j`. The spins on different sites commute, the fermions on different sites anticommute, and the string is the converter — that's the same content as before, just read on the lattice.

Now watch the string pay off. Take the one-dimensional nearest-neighbor exchange Hamiltonian of the `XX` type,
```
H = − J Σ_j (S^x_j S^x_{j+1} + S^y_j S^y_{j+1}) = − (J/2) Σ_j (S^+_j S^-_{j+1} + S^-_j S^+_{j+1}).
```
Substitute the dictionary into a single bond term `S^+_j S^-_{j+1}`:
```
S^+_j S^-_{j+1} = a_j^† e^{iπ Σ_{l<j} N_l} · a_{j+1} e^{−iπ Σ_{l<j+1} N_l}.
```
The strings of two neighboring sites overlap on all sites `l < j`, and those cancel (`e^{iπX} e^{−iπX}=1`); what is left from the `j+1` string is the single extra factor on site `j` itself, `e^{−iπ N_j} = 1 − 2N_j`. So
```
S^+_j S^-_{j+1} = a_j^† (1 − 2N_j) a_{j+1}.
```
Now `a_j^†(1 − 2N_j) = a_j^† − 2 a_j^† N_j = a_j^† − 2 a_j^† a_j^† a_j = a_j^†` (since `a_j^† a_j^† = 0`). So `S^+_j S^-_{j+1} = a_j^† a_{j+1}`. The string evaporated on the bond — that is the miracle for *nearest-neighbor* hopping. Doing the same for the conjugate term,
```
H = − (J/2) Σ_j (a_j^† a_{j+1} + a_{j+1}^† a_j).
```
This is *quadratic* in the fermion amplitudes — a free (non-interacting) fermion gas hopping on a line. A many-body interacting spin problem has become free fermions, exactly solvable. On a periodic chain of `N` sites I diagonalize by Fourier transform `a_j = (1/√N) Σ_k ã_k e^{ikj}`:
```
H = − (J/2) Σ_k (e^{ik} + e^{−ik}) ã_k^† ã_k = Σ_k ω_k  ã_k^† ã_k,    ω_k = − J cos k.
```
The single-particle spectrum is `ω_k = −J cos k`; the ground state fills every mode with `ω_k < 0`, and the magnetization `⟨Σ_j S^z_j⟩ = ⟨Σ_j (N_j − 1/2)⟩` follows from counting the filled negative-energy modes. The magnon dispersion above the ground state is read straight off `ω_k`.

One caveat I should state, because it tells me exactly *when* the free-fermion miracle happens. Add an `S^z S^z` term — the `XXZ` chain, `H = −J Σ (S^x S^x + S^y S^y) − J_z Σ S^z_j S^z_{j+1}`. The new term is string-free (`S^z = N − 1/2` carries no string), but it is *quartic*:
```
− J_z Σ_j S^z_j S^z_{j+1} = − J_z Σ_j (N_j − 1/2)(N_{j+1} − 1/2),
```
a density-density *interaction* between neighboring fermions. So the transformation always works — it is exact for any `J_z` — but it produces *free* fermions only when the original Hamiltonian is bilinear in `S^±`, i.e. the `XX` point (`J_z = 0`) and the transverse-field Ising line; off those, the fermions interact and you've traded one hard problem for another (often still useful) hard problem. The string telescopes only across a single bond, so the freeness is a property of *nearest-neighbor* `S^+S^-` terms specifically — a longer-range `S^+_j S^-_{j+m}` leaves a genuine string `∏_{j ≤ l < j+m}(1 − 2N_l)` that does not collapse, giving a non-local (still exact) fermion term.

Let me write out the construction as code, on a finite register, so it is concrete and checkable: build the single-mode pieces, dress the lowering operator with the parity string `∏_{l<j}(1 − 2N_l)` to make the fermion amplitude, confirm the canonical anticommutation relations numerically, then map the `XX` chain to free fermions and read off the dispersion.

```python
import numpy as np

# ---- single two-level mode (one Fermi mode == one spin-1/2) ----
I  = np.eye(2)
sm = np.array([[0, 1], [0, 0]], dtype=complex)     # b = sigma^-  (lowering)
sp = sm.conj().T                                   # b^dagger = sigma^+
n1 = sp @ sm                                       # single-mode N, eigenvalues {0,1}
mz = I - 2 * n1                                     # 1 - 2N = -sigma^z, eigenvalues +-1

def on_mode(local, j, K):
    """Place a single-mode 2x2 operator on mode j of a K-mode register."""
    m = np.array([[1]], dtype=complex)
    for k in range(K):
        m = np.kron(m, local if k == j else I)
    return m

def jw_string(j, K):
    """The Jordan-Wigner parity string prod_{l<j} (1 - 2 N_l) = prod_{l<j} (-sigma^z_l)."""
    s = np.eye(2 ** K, dtype=complex)
    for l in range(j):
        s = s @ on_mode(mz, l, K)
    return s

def a(j, K):                                       # fermion amplitude  a_j = string . b_j
    return jw_string(j, K) @ on_mode(sm, j, K)

def adag(j, K):                                    # a_j^dagger = b_j^dagger . string
    return on_mode(sp, j, K) @ jw_string(j, K)

# ---- check the canonical anticommutation relations {a_i, a_j^dag} = delta_ij, {a_i,a_j}=0 ----
def anticomm(A, B):
    return A @ B + B @ A

K = 4
for i in range(K):
    for j in range(K):
        target = np.eye(2 ** K) if i == j else np.zeros((2 ** K, 2 ** K))
        assert np.allclose(anticomm(adag(i, K), a(j, K)), target)   # {a_i^dag, a_j} = delta
        assert np.allclose(anticomm(a(i, K), a(j, K)), 0)           # {a_i, a_j} = 0

# ---- XX spin chain  ->  free fermions, exactly ----
def spin(op, j, K):                                # S^+, S^-, S^z on site j (open chain)
    return on_mode(op, j, K)
J = 1.0
H_spin = np.zeros((2 ** K, 2 ** K), dtype=complex)
for j in range(K - 1):                             # -(J/2) sum (S^+_j S^-_{j+1} + S^-_j S^+_{j+1})
    H_spin += -(J / 2) * (spin(sp, j, K) @ spin(sm, j + 1, K)
                          + spin(sm, j, K) @ spin(sp, j + 1, K))
H_free = np.zeros((2 ** K, 2 ** K), dtype=complex)
for j in range(K - 1):                             # -(J/2) sum (a^dag_j a_{j+1} + a^dag_{j+1} a_j)
    H_free += -(J / 2) * (adag(j, K) @ a(j + 1, K) + adag(j + 1, K) @ a(j, K))
assert np.allclose(H_spin, H_free)                 # the string telescopes on every n.n. bond

# ---- periodic chain: dispersion omega_k = -J cos k ----
N = 8
ks = [2 * np.pi * m / N for m in range(N)]
omega = [-J * np.cos(k) for k in ks]               # single-particle spectrum; ground state fills omega_k<0
```

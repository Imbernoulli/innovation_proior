Let me start from the thing that actually hurts. Hartree–Fock gives me a determinant `|Φ0⟩` and about ninety-nine percent of the energy, and the remaining one percent — the correlation energy `ΔE = E_exact − E_HF` — is where all the chemistry is: the bond energies, the barriers, the differences I care about. So I need to recover correlation. And there is an obvious, honest way to do it: the exact state lives in the space of all determinants I can build from my finite orbital basis, so write

```
|Ψ⟩ = (1 + C1 + C2 + C3 + …) |Φ0⟩
```

where `C1` puts in all single excitations `Φ_i^a`, `C2` all doubles `Φ_ij^ab`, and so on, with coefficients I find by making `⟨Ψ|H|Ψ⟩` stationary — diagonalize `H` in the determinant space. If I keep everything up to `n`-fold for `n` electrons this is full CI and it is exact. It is also dead on arrival: the number of determinants is roughly `(nN)^n`, astronomical. So in practice I truncate — keep singles and doubles, CISD, because the Hamiltonian only has one- and two-body operators so it only directly connects determinants differing by at most two orbitals, and doubles are where the leading correlation sits.

And here is where I hit the wall, and I want to be precise about exactly what breaks, because the precise failure is the whole clue. Take two helium atoms — or any two closed-shell fragments A and B — and pull them infinitely far apart so they don't interact at all. The right answer is `E(A···B) = E(A) + E(B)`, obviously; nothing couples them. Now do CISD on the pair. To correlate A I need its double excitations; to correlate B I need *its* double excitations. But to have *both* pairs correlated *at the same time* — which is the physically correct state of the joint system — I need a determinant with a double excitation on A *and simultaneously* a double excitation on B. That is a *quadruple* excitation of the combined system. CISD does not contain quadruples. So CISD literally cannot represent the joint correlated state; it is forced to leave one fragment or the other uncorrelated in any given determinant. The energy of the pair comes out strictly above `E(A) + E(B)`, and — this is the killer — the error *grows with system size* and never goes away. For `N` non-interacting copies the per-copy correlation energy *shrinks* with `N`. That is not a small numerical blemish. It means I cannot tabulate CISD energies and subtract them to get a reaction energy, cannot trust a dissociation curve, cannot touch a solid. Truncated CI is not size-extensive, and size-extensivity is the one property I cannot do without.

Let me look at where the unphysical piece actually enters the equations, because I want to see the mechanism, not just the symptom. Extract the CISD energy by perturbation theory from the eigenvalue problem. You get the structure `E = E_ref + ⟨Φ0|H|h⟩ ⟨h|E − H|h⟩⁻¹ ⟨h|H|Φ0⟩`, and when you iterate this you find renormalization terms of the form `−ΔE · ⟨Ψ(1)|Ψ(1)⟩` floating around, where `ΔE` itself scales linearly with the number of particles (it's a closed, size-extensive quantity) and `⟨Ψ(1)|Ψ(1)⟩` *also* scales linearly. Their product scales like `n²`. So a term that ought to scale like `n` is contaminated by a term scaling like `n²`. In the full CI those bad terms get exactly cancelled — by contributions coming down from the higher excitations I threw away. Truncate, and the cancellation is destroyed.

This is the linked-diagram theorem talking. Brueckner saw the low-order cancellation in 1955; Goldstone proved it to all orders in 1957: the exact correlation energy and wavefunction are sums of *linked* diagrams only, and the *unlinked* diagrams — the disconnected blobs that multiply the reference — cancel against renormalization. An expansion that keeps unlinked diagrams scales wrongly with `n`; a purely linked expansion scales correctly. Many-body perturbation theory does it right precisely because, order by order, it is a linked-diagram theory (Kelly applied it to atoms in '69; the physicists had it well in hand). So MBPT is size-extensive at every order. But MBPT is a *finite-order* expansion — second order, third, fourth — a fixed-accuracy snapshot. It does not *resum* a whole class of effects to infinite order, and for anything strongly correlated a low order just isn't enough, while high order by hand is hopeless.

So I have two unsatisfactory poles. Variational truncated CI: not extensive. Finite-order MBPT: extensive but not resummed. I want the best of both — a method that is size-extensive by construction *and* sums an infinite class of correlation effects.

Let me stare harder at *why* CISD fails, because the failure is pointing at the fix. The problem was the missing quadruples. But what *kind* of quadruples did I need for the two separated fragments? Not some intricate, genuine four-electron correlation. I needed exactly: pair-on-A *times* pair-on-B. A *product* of two double excitations. The amplitude of that quadruple isn't an independent new number — it should just be the amplitude of A's double *times* the amplitude of B's double. Let me check that this is general and not a special trick of the separated case. Picture any molecule with two electron pairs sitting in well-separated regions. The leading four-electron effect is the two pairs correlating *independently and simultaneously*. That's a disconnected quadruple: a product of two pair amplitudes. How many such numbers are there? Two pair amplitudes, each roughly `n²N²` of them — but the *product* is determined by the factors, so the genuinely independent information is still just `∼ n²N²`. Compare that to a true, connected four-electron cluster — four indices up, four down — which is `∼ n⁴N⁴` independent numbers and physically much smaller. So the higher excitations that actually matter are *factorizable*: they are products of the low ones, carrying no new independent information.

That is the crack of light. If the important quadruples are *products* of doubles, I should not be adding them as independent linear terms `C4`. I should be *generating* them automatically as products. What mathematical object turns a sum into automatic products of itself?

The exponential.

Let me try it. Posit

```
|Ψ⟩ = e^{T} |Φ0⟩,    T = T1 + T2 + T3 + … ,
```

where `T2` is a double-excitation operator with its own amplitudes,

```
T2 = ¼ Σ_ijab t_ij^ab a†_a a†_b a_j a_i ,
```

and similarly `T1 = Σ_ia t_i^a a†_a a_i`. Now expand the exponential:

```
e^T = 1 + (T1 + T2 + …) + ½(T1 + T2 + …)² + … .
```

Watch what the `½ T2²` term does. It is a product of two double-excitation operators, so acting on `|Φ0⟩` it produces *quadruple* excitations — and their amplitude is literally `t_ij^ab · t_kl^cd`, the product of two doubles amplitudes. That is *exactly* the disconnected quadruple I needed for the two separated fragments, generated for free, with no new unknowns. The exponential builds in precisely the factorization that the physics demands. If I match this to the CI picture by writing `e^T = 1 + C`, I can read off the cluster-decomposition of the CI coefficients:

```
C1 = T1
C2 = T2 + ½ T1²
C3 = T3 + T1 T2 + (1/3!) T1³
C4 = T4 + ½ T2² + T1 T3 + ½ T1² T2 + (1/4!) T1⁴ .
```

So the CI doubles coefficient `C2` is the *connected* pair amplitude `T2` plus a disconnected piece `½ T1²`; the CI quadruple `C4` is the genuine connected `T4` plus the dominant disconnected `½ T2²` plus more products. The unknowns I actually solve for are only the *connected* cluster amplitudes `T1, T2, …`; everything disconnected is determined as products. And now the separated-fragments disaster cannot happen: even if I keep only `T2` (drop `T3, T4, …` entirely), `e^{T2} = 1 + T2 + ½T2² + …` *already contains* the product quadruples, hextuples, and all higher even excitations as disconnected products. The wavefunction for A···B factorizes, `e^{T_A + T_B} = e^{T_A} e^{T_B}`, because `T_A` and `T_B` commute (they act on disjoint orbital sets), and the energy is additive. Size-extensivity is automatic, structurally, before I even write an equation.

This is the cluster idea the nuclear physicists had been circling — Coester in '58, Coester and Kümmel in 1960 wrote the nuclear-matter wavefunction as an exponentiated correlation operator on a model state, an Ursell-type expansion, borrowing the same combinatorial device Ursell and Mayer used for the interacting gas in statistical mechanics. They had the *representation*. What they didn't have, for a real interacting many-fermion system, was a closed, finite set of equations you could actually solve — it was thought intractable. Let me see if I can reduce it to working equations for electrons.

Now, how do I get equations for the `t`'s? The instinct is variational: minimize `⟨Ψ|H|Ψ⟩ / ⟨Ψ|Ψ⟩` with `|Ψ⟩ = e^T|Φ0⟩`. Let me see if that's tractable. The numerator is `⟨Φ0| e^{T†} H e^{T} |Φ0⟩`. The trouble: `e^{T†}` is a sum of *de-excitation* operators, `e^T` a sum of excitations, and there's no reason for the product `e^{T†} … e^T` to terminate — it's an infinite series sandwiching `H`, and even truncating `T` at `T2` the variational expression never closes into a finite number of terms. So a variational treatment of the exponential ansatz does not give me finite equations. Wall.

Back up. I don't have to be variational. I have the Schrödinger equation `H e^T |Φ0⟩ = E e^T |Φ0⟩`; let me just *project* it. The clean move is to multiply on the left by `e^{−T}` first:

```
e^{−T} H e^{T} |Φ0⟩ = E |Φ0⟩ .
```

Define the **similarity-transformed Hamiltonian** `H̄ = e^{−T} H e^{T}`. The transformation leaves the eigenvalues alone (it's not unitary, but it preserves the spectrum), so `H̄` has the same `E`. Now project. Onto the reference:

```
⟨Φ0| H̄ |Φ0⟩ = E
```

gives the energy. Onto each excited determinant:

```
⟨Φ_ij^ab…| H̄ |Φ0⟩ = 0
```

gives one equation per amplitude — exactly as many equations as unknowns. I have a closed nonlinear system. But is it *finite*? Everything hinges on whether `H̄ = e^{−T} H e^{T}` is a finite object.

Expand it with the Hausdorff/BCH formula. The standard nested-commutator series is

```
H̄ = H + [H, T] + (1/2!) [[H, T], T] + (1/3!) [[[H, T], T], T] + (1/4!) [[[[H, T], T], T], T] + … .
```

Let me think about whether this terminates. Why would it? Look at one commutator `[H, T]`. The reason a similarity transform usually does *not* terminate is that `[H,T]` is generally as complicated as `H` itself, and nesting just grows it. But here something special holds. `T` is a pure excitation operator — it has *only* creation operators for particles and annihilation operators for holes; every elementary piece of `T` raises the excitation level. So `T` commutes with itself: `[T, T'] = 0` for any two cluster components. That means in `[H, T]` the only surviving terms are those where an operator *of H* contracts with an operator *of T* — a creation in `T` meeting an annihilation in `H`, or vice versa. A nested commutator `[[…[H, T]…], T]` therefore requires *each* successive `T` to contract with `H` (it can't contract with another `T`, since the `T`'s commute, and an uncontracted `T` sitting next to the already-formed object just commutes through and cancels in the commutator). So every `T` in a surviving nested term must share at least one index with `H`.

Now count. `H` has at most a *two-body* operator: four second-quantized operators, `a†_p a†_q a_s a_r`. Each `T` that survives must contract with at least one of these four legs of `H`. Once all four legs of `H` are contracted with `T`'s, there is nothing left in `H` for a fifth `T` to contract with — so the fifth commutator vanishes. The series **terminates exactly after the fourfold commutator**:

```
H̄ = H + [H, T] + ½[[H, T], T] + (1/3!)[[[H, T], T], T] + (1/4!)[[[[H, T], T], T], T] ,
```

and not one term further. This is the whole game. Even though `e^T` is an infinite series, the projected equations are a *finite* polynomial in the amplitudes — at most quartic in `T`. The infinite exponential gets tamed into a closed set of equations purely because the Hamiltonian is two-body. That termination is what makes the cluster idea *computable*, and it is exactly the thing the nuclear physicists had not pinned down for a usable electronic algorithm.

And there's a second gift hiding in the commutators. Because each surviving term requires `H` and the `T`'s to share indices — to be *contracted together* — every term in `H̄` is **connected**: any piece in which `H` and a `T` had no index in common would have to come from a plain product `H·T` with nothing linking them, but the commutator structure subtracts exactly those off. So

```
H̄ = (H e^T)_C ,
```

the subscript `C` meaning "connected part only." The amplitude equations `⟨Φ_ij^ab| H̄ |Φ0⟩ = 0` are therefore connected, the cluster amplitudes are connected, and so the wavefunction `e^T|Φ0⟩` is *linked*. I have re-derived the linked-diagram theorem — Hubbard's '57 statement that the exact wave operator is the sum of linked contributions — but now algebraically, as an identity about a similarity transform, and built directly into a solvable method. Size-extensivity isn't something I check afterward; it's forced by the connectedness of `H̄`.

Let me make the bookkeeping honest, because "connected" and "the surviving contractions" need real machinery or I'll make sign errors. I'll work in second quantization with the **Fermi vacuum** `|Φ0⟩` as my reference, and reinterpret operators in particle–hole language: `a†_a` and `a_i` *create* excitations (a particle above the sea, a hole in it), while `a_a` and `a†_i` *destroy* them. The point of normal-ordering relative to this vacuum, `{…}`, is that all the "destroy-excitation" operators stand to the right, so `⟨Φ0|{…}|Φ0⟩ = 0` for any nontrivial normal-ordered string — the vacuum expectation value of a normal-ordered product vanishes unless it's fully contracted. A **contraction** of two operators is `AB − {AB}`; relative to the Fermi vacuum the only nonzero contractions are `⟨a_i a†_j⟩-type` hole pairs and particle pairs giving `δ`'s. **Wick's theorem** then says any operator string equals the sum over all ways of contracting it, each as a normal-ordered remainder, so a vacuum expectation value collapses to its *fully contracted* terms — which is what I'll compute. This is the apparatus (Wick 1950, in the time-independent particle–hole form) that turns "which terms survive" from guesswork into a finite enumeration; importing it to the electronic problem is the bridge I need.

First put the Hamiltonian into normal order relative to `|Φ0⟩`. Take the bare

```
H = Σ_pq ⟨p|h|q⟩ a†_p a_q + ¼ Σ_pqrs ⟨pq||rs⟩ a†_p a†_q a_s a_r .
```

Normal-ordering each operator string and collecting the fully-contracted leftovers, the one-body part gives `Σ_pq f_pq {a†_p a_q} + Σ_i ⟨i|h|i⟩`, where `f_pq = ⟨p|h|q⟩ + Σ_i ⟨pi||qi⟩` is exactly the **Fock matrix** — the mean field reappears as the contraction of the two-body term against the occupied orbitals, not because I assumed it. The two-body part gives `¼ Σ_pqrs ⟨pq||rs⟩ {a†_p a†_q a_s a_r}` plus one-body and constant leftovers. Subtracting the reference expectation value `⟨Φ0|H|Φ0⟩ = E_HF`, I define the **normal-ordered Hamiltonian**

```
H_N = H − ⟨Φ0|H|Φ0⟩ = F_N + W_N,
F_N = Σ_pq f_pq {a†_p a_q},   W_N = ¼ Σ_pqrs ⟨pq||rs⟩ {a†_p a†_q a_s a_r} .
```

For a canonical HF reference `f_pq = ε_p δ_pq` is diagonal — the off-diagonal occupied–virtual block `f_ai` vanishes, which is just Brillouin's theorem. Working with `H_N` is cleaner because all the trivial internal contractions are already removed; everything from here is contractions of `H_N` against the `T`'s. The Schrödinger equation becomes `H_N e^T|Φ0⟩ = ΔE · e^T|Φ0⟩` with `ΔE = E − E_HF` the correlation energy, and the projections become

```
ΔE = ⟨Φ0| (H_N e^T)_C |Φ0⟩,
0  = ⟨Φ_ij^ab…| (H_N e^T)_C |Φ0⟩ .
```

Now let me make the *first* concrete model, the simplest one that is still size-extensive and captures the leading physics. The Hamiltonian is two-body, so it couples the reference *directly* to double excitations — and the leading correlation effect is pair correlation. Singles `T1` describe orbital relaxation, but for a canonical HF reference Brillouin's theorem (`f_ai = 0`) means singles do not couple to the reference at first order; their effect on the energy first appears only at fourth order. So the cheapest honest model is to keep `T = T2` alone. Call it coupled-cluster doubles. It's `n²N²` unknowns (the `t_ij^ab`), it's size-extensive (the `½T2²` term supplies the disconnected quadruples), and `e^{T2}` already includes the disconnected parts of quadruples, hextuples, and all higher even excitations. That is exactly what's needed for many electrons in different parts of a molecule, and it cannot be done with truncated CI at any comparable cost.

Let me derive the CCD equations explicitly. With `T = T2`,

```
e^{T2} = 1 + T2 + ½ T2² + … ,
```

but the BCH termination tells me only finitely many terms survive in each projection. The energy:

```
ΔE = ⟨Φ0| H_N (1 + T2 + ½T2² + …) |Φ0⟩_C .
```

`H_N|Φ0⟩` connected to `⟨Φ0|`: the constant part of `H_N` is already subtracted, `F_N` raises/lowers by zero net but `⟨Φ0|F_N|Φ0⟩ = 0`, and `⟨Φ0|H_N|Φ0⟩ = 0`. The only term that can fully contract back to the reference is `⟨Φ0| W_N T2 |Φ0⟩`: `W_N` must de-excite the double that `T2` created. Compute it. `T2` creates `a†_a a†_b a_j a_i` with amplitude `¼ t_ij^ab`; `W_N` brings `¼ ⟨pq||rs⟩ {a†_p a†_q a_s a_r}`; the fully-contracted vacuum expectation pairs the two particles `a,b` against the `W_N` annihilators and the two holes `i,j` against the creators. Doing the contractions (it is the only surviving term, and the combinatorial factors collapse the two `¼`'s),

```
ΔE = ¼ Σ_ijab ⟨ij||ab⟩ t_ij^ab .
```

A clean result: the correlation energy is just the doubles amplitudes contracted with the two-electron integrals. And if I set `t_ij^ab` to its first iteration — solving the rest of the equation with `t = 0` on the right — I'll recover `t_ij^ab = ⟨ij||ab⟩ / (ε_i + ε_j − ε_a − ε_b)`, and this energy becomes the MP2 energy. So MBPT(2) drops out as the first step of CCD; the method is a self-consistent resummation that *starts* at MP2 and iterates to infinite order in the doubles.

Now the amplitude equation, the heart of it:

```
0 = ⟨Φ_ij^ab| H_N e^{T2} |Φ0⟩_C = ⟨Φ_ij^ab| (H_N + H_N T2 + ½ H_N T2²) |Φ0⟩_C .
```

Three groups of terms — `H_N` alone, `H_N` times one `T2`, `H_N` times two `T2`'s (the quadratic term, where the BCH series stops for doubles because two `T2`'s already saturate the four legs of `W_N`). Let me enumerate the surviving connected contractions.

`⟨Φ_ij^ab| H_N |Φ0⟩` — `H_N` must itself create the double `ij→ab`. Only `W_N` can; the fully-contracted term is the bare integral

```
⟨ij||ab⟩   (the driver — this is what makes t nonzero in the first place).
```

`⟨Φ_ij^ab| H_N T2 |Φ0⟩_C` — `T2` puts in one double, `H_N` (one- or two-body) then connects to it to land on the target double `ij→ab`. Several distinct connected contractions appear:
- the Fock pieces `f_ac t_ij^cb` and `−f_ki t_kj^ab` (with their antisymmetric partners over `a↔b` and `i↔j`) — for canonical HF these are just the orbital-energy shifts that build the denominator;
- the **particle–particle ladder** `½ Σ_cd ⟨ab||cd⟩ t_ij^cd` — `W_N` scatters the two particles `c,d` of the existing double up to `a,b`;
- the **hole–hole ladder** `½ Σ_kl ⟨kl||ij⟩ t_kl^ab` — `W_N` scatters the two holes;
- the **particle–hole ring** `P(ij)P(ab) Σ_kc ⟨kb||cj⟩ t_ik^ac` — `W_N` mixes one particle and one hole, with the antisymmetrizer `P` summing the distinct index permutations.

`½ ⟨Φ_ij^ab| H_N T2² |Φ0⟩_C` — the genuinely new, *nonlinear* terms: two existing doubles, with `W_N` tying them together so the connected result is again a net double on `ij→ab`. These are the quadratic-in-`t` contributions, schematically `Σ_klcd ⟨kl||cd⟩ t·t` in several index patterns:

```
¼ Σ ⟨kl||cd⟩ t_ij^cd t_kl^ab    (ladder × ladder),
   Σ ⟨kl||cd⟩ t_ik^ac t_lj^db   (ring-type),
−½ Σ ⟨kl||cd⟩ t_ik^ab t_jl^cd  (and the P(ij) partner),
−½ Σ ⟨kl||cd⟩ t_ij^ac t_kl^bd  (and the P(ab) partner),
```

each with the right sign from the number of hole lines and loops, and each preceded by the antisymmetrizing permutation operator `P(ij/ab)` so the amplitude stays fully antisymmetric (`t_ij^ab = −t_ji^ab = −t_ij^ba`). These quadratic terms are exactly the contributions that, diagrammatically, sit inside the disconnected quadruple excitations — the products of pair correlations — folded back down into the doubles equation. They are what make the method a true infinite-order resummation rather than a finite order of perturbation theory, and they are why it stays extensive.

Collecting everything, the CCD amplitude equation is, with `t = 0` excluded, a set of coupled nonlinear equations, one per `(i<j, a<b)`:

```
0 = ⟨ij||ab⟩
    + ½ Σ_cd ⟨ab||cd⟩ t_ij^cd
    + ½ Σ_kl ⟨kl||ij⟩ t_kl^ab
    + P(ij)P(ab) Σ_kc ⟨kb||cj⟩ t_ik^ac
    + (Fock / denominator terms)
    + (the quadratic t·t terms above)
    − Σ ... (the disconnected-quadruple pieces, with their P antisymmetrizers).
```

I notice I can factor this. The two-`t` Coulomb contractions appear *inside* the linear terms too if I define intermediates. For instance, fold `½ Σ_klcd ⟨kl||cd⟩ t_ij^cd t_kl^ab` and the hole–hole ladder into a single hole–hole intermediate, and similarly a particle–particle one and a ring one:

```
I_oo(m,i) = f_mi + ½ Σ_nef ⟨mn||ef⟩ t_ef^in     (the m=i diagonal handled by the denominator)
I_vv(a,e) = f_ae − ½ Σ_mnf ⟨mn||ef⟩ t_af^mn
I_voov(a,m,i,e) = ⟨am||ie⟩ + ½ Σ_nf ⟨mn||ef⟩ t_af^in
I_oooo(m,n,i,j) = ⟨mn||ij⟩ + ½ Σ_ef ⟨mn||ef⟩ t_ef^ij .
```

Then the residual is compact:

```
R_ij^ab = ⟨ab||ij⟩
        + ½ I_vv(a,e) t_ij^eb  − ½ I_oo(m,i) t_mj^ab
        + I_voov(a,m,i,e) t_mj^eb
        + ⅛ ⟨ab||ef⟩ t_ij^ef + ⅛ I_oooo(m,n,i,j) t_mn^ab ,
```

with the result antisymmetrized over `a↔b` and `i↔j` to enforce the symmetry of the amplitude. The intermediates package the quadratic terms — e.g. `I_vv` and `I_oo` each carry a `t`, so when they multiply another `t` in `R` they produce exactly the `t·t` quadruple-product contributions, but evaluated once and reused, which is what keeps the cost at `∼ n²N⁴` instead of blowing up. At convergence `R_ij^ab = 0`; before convergence I treat `R` as the update direction.

How do I actually solve a coupled nonlinear system like this? Quasi-Newton would be murder. But the structure hands me a fixed-point iteration. Pull the diagonal Fock part out — for canonical HF it gives the orbital-energy denominator — and rearrange:

```
t_ij^ab ← R_ij^ab / D_ij^ab,    D_ij^ab = ε_i + ε_j − ε_a − ε_b ,
```

where `R` now means "everything except the diagonal term I divided by," evaluated at the current `t`. Start from `t = 0`; the first pass gives `R = ⟨ab||ij⟩` and the update is `t = ⟨ij||ab⟩ / D` — that is MP2, as I noted. Iterate to self-consistency (and accelerate with DIIS extrapolation on the amplitude error if I want it to converge fast). Each iteration is dominated by the particle–particle ladder `⟨ab||cd⟩ t_ij^cd`, which is `∼ n²N⁴` — the same scaling as CISD, but unlike CISD it is size-extensive and it sums the doubles to infinite order. I'm getting more physics for the same cost.

Let me sanity-check the whole construction on a case I can verify by hand, because if the equations are right they must give the exact answer where the exact answer is reachable. Take a two-electron system in a minimal basis — say a stretched H₂ in two spatial orbitals, four spin-orbitals: the two occupied (bonding, both spins) and two virtual (antibonding, both spins), built from a *canonical HF reference* so the Fock matrix is diagonal and `f_ai = 0` (Brillouin), which means I can legitimately drop singles. Two electrons. With only two electrons there are *no* genuine triple or quadruple excitations possible — you can't promote more electrons than you have. So `T3 = T4 = … = 0` *exactly*, not by truncation, and `T = T2` is the *complete* cluster operator. That means CCD must equal full CI exactly for two electrons. If my equations are correct, the converged `¼ Σ ⟨ij||ab⟩ t_ij^ab` must reproduce the full-CI correlation energy to all decimal places. This is the kind of identity I can check, and it pins down every sign and factor. (When I work it through with a consistent minimal-basis model — building the diagonal Fock matrix from the same integrals, iterating the residual to convergence, and diagonalizing the full `6×6` two-electron CI Hamiltonian over all four spin-orbitals, of which the reference-plus-doubles span the relevant block — the two correlation energies agree to machine precision. The equations are right.)

I should be clear about what I gave up. The projected equations are *not* variational — `H̄ = e^{−T}H e^T` is non-Hermitian, the energy is not an upper bound to the exact energy, and I no longer have the variational guarantee. But that is the price for the BCH termination: a variational treatment of the exponential never closes into finite equations, and finite, connected, size-extensive equations are worth more than a variational bound for the central task of getting accurate, *consistent* energy differences. I'll accept non-Hermiticity.

And singles? For a non-HF reference, or to let the orbitals relax, I'd add `T1` and project also onto singles `⟨Φ_i^a| H̄ |Φ0⟩ = 0`. Then `e^{T1+T2}` carries `T1`, `½T1²`, `T1 T2`, … and the equations pick up the corresponding contractions. That is the singles-and-doubles model. For the canonical-HF closed-shell case the doubles already carry the leading correlation and are correct through third order in perturbation theory, so the doubles model stands on its own as the first complete, computable realization of the cluster idea for electrons.

Let me put the doubles solver into code, mirroring exactly the equation I derived — the residual built from the four intermediates, the Jacobi update by the orbital-energy denominator, and the energy as `¼ ⟨ij||ab⟩ t`. The arrays are indexed `t2[a,b,i,j]` (two virtual indices, two occupied), `g[p,q,r,s] = ⟨pq||rs⟩` antisymmetrized, `f` the Fock matrix, `o`/`v` the occupied/virtual slices.

```python
import numpy as np

def doubles_residual(t2, f, g, o, v):
    """⟨Φ_ij^ab| (H_N e^{T2})_C |Φ0⟩ — the connected-cluster doubles equation.
    Intermediates fold the quadratic t·t (disconnected-quadruple) terms in."""
    # hole-hole and particle-particle intermediates carry one t each
    I_oo   = f[o, o] + 0.5 * np.einsum("mnef,efin->mi", g[o, o, v, v], t2)
    I_vv   = f[v, v] - 0.5 * np.einsum("mnef,afmn->ae", g[o, o, v, v], t2)
    # particle-hole ring and hole-hole ladder intermediates
    I_voov = g[v, o, o, v] + 0.5 * np.einsum("mnef,afin->amie", g[o, o, v, v], t2)
    I_oooo = g[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2)

    r  = 0.5 * np.einsum("ae,ebij->abij", I_vv, t2)        # particle-particle (+ quadratic)
    r -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2)        # hole-hole (+ quadratic)
    r += np.einsum("amie,ebmj->abij", I_voov, t2)          # particle-hole ring (+ quadratic)
    r += 0.125 * np.einsum("abef,efij->abij", g[v, v, v, v], t2)   # particle-particle ladder
    r += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2)         # hole-hole ladder (+ quadratic)

    r -= np.transpose(r, (1, 0, 2, 3))   # antisymmetrize a<->b
    r -= np.transpose(r, (0, 1, 3, 2))   # antisymmetrize i<->j
    r += g[v, v, o, o]                   # the driver  ⟨ab||ij⟩
    return r

def ccd_energy(t2, g, o, v):
    """ΔE = ¼ Σ ⟨ij||ab⟩ t_ij^ab"""
    return 0.25 * np.einsum("ijab,abij->", g[o, o, v, v], t2)

def solve_ccd(f, g, o, v, maxit=200, tol=1e-12):
    eps = np.diagonal(f)
    n = np.newaxis
    # 1 / D_ij^ab  with D = ε_i + ε_j − ε_a − ε_b
    inv_D = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    nv = f[v, v].shape[0]; no = f[o, o].shape[0]
    t2 = np.zeros((nv, nv, no, no))      # start at t = 0  → first step is MP2
    e_old = 0.0
    for it in range(maxit):
        r  = doubles_residual(t2, f, g, o, v)
        t2 = t2 + r * inv_D              # Jacobi update by the orbital-energy denominator
        e  = ccd_energy(t2, g, o, v)
        if abs(e - e_old) < tol:
            break
        e_old = e
    return t2, e
```

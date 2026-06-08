# IP = PSPACE

## The result

A polynomial-time **randomized** verifier interacting with an **all-powerful but untrusted** prover can
verify exactly the languages decidable in polynomial space:

> **Theorem (IP = PSPACE).** A language `L` has an interactive proof system if and only if `L ∈ PSPACE`.

This is far beyond NP: it places coNP, the entire polynomial-time hierarchy, `P^{#P}`, and all of
PSPACE inside IP. It was the first major **non-relativizing** result — there is an oracle
(Fortnow–Sipser) under which coNP has no interactive proofs, so the proof cannot treat computation as a
black box; it must, and does, exploit algebraic structure.

## The key idea: arithmetization + sum-check + Schwartz–Zippel

1. **Arithmetization.** Lift Boolean objects to low-degree polynomials over a large prime field `F_p`:
   `x∧y ↦ XY`, `¬x ↦ 1−X`, `x∨y ↦ 1−(1−X)(1−Y)`, and a clause `A∨B∨C` becomes
   `1−(1−A)(1−B)(1−C)`. A 3-CNF with `m` clauses becomes, as a product of the clause-polynomials kept
   un-expanded, a size-`O(m)` polynomial `Φ` of degree `≤ 3m` that equals the formula on `{0,1}^n`. Off
   the cube, `Φ` gives the verifier random points the prover can't anticipate.
2. **Sum-check.** To verify a claim `K = Σ_{b∈{0,1}^n} g(b)` for a polynomial `g` of degree `≤ d` per
   variable, strip one variable per round: the prover sends the univariate
   `h_j(X_j) = Σ_{rest∈{0,1}} g(r_1,…,r_{j-1}, X_j, rest)`, the verifier checks `h_j(0)+h_j(1)` against
   the running claim, then binds `X_j` to a fresh random `r_j` and recurses.
3. **Schwartz–Zippel soundness.** A cheating prover must lie about some round's polynomial; two distinct
   degree-`d` univariates agree on `≤ d` points, so a random field point exposes the lie with
   probability `≥ 1 − d/|F|`. This randomness is what lifts the verifier past NP.

## Sum-check, stated and proved

**Protocol.** Input: prime `p`, polynomial `g(X_1,…,X_n)` of degree `≤ d` in each variable, claimed sum
`K`. If one variable remains, the verifier evaluates `g(0)+g(1)` directly and compares it to the running
claim. Otherwise the prover sends the degree-`≤ d` univariate
`s_1(X_1) = Σ_{b_2,…,b_n} g(X_1,b_2,…,b_n)`; the verifier checks `s_1(0)+s_1(1)=K`, samples
`r_1 ∈ F_p`, and recurses on the smaller claim
`s_1(r_1)=Σ_{b_2,…,b_n} g(r_1,b_2,…,b_n)`. Iterating peels one variable per round until the direct
one-variable check applies.

**Completeness.** The honest prover sends the true `h_j` each round; all checks pass; accept w.p. 1.

**Soundness.** If `K ≠ Σ_b g(b)` then `Pr[reject] ≥ (1 − d/p)^n`. *Proof by induction on `n`.* For
`n=1`, the verifier evaluates `g(0)+g(1)` directly and rejects with probability 1 if it differs from `K`.
For `n>1`, a true `h_1` fails the round-1 check, so the prover sends `s_1 ≠ h_1`; the nonzero polynomial
`s_1−h_1` has `≤ d` roots, so `Pr_{r_1}[s_1(r_1) ≠ h_1(r_1)] ≥ 1 − d/p`. On that event the prover faces
a false `(n−1)`-variable claim and, by the induction hypothesis, fails it with probability
`≥ (1−d/p)^{n−1}`. Hence
`Pr[reject] ≥ (1−d/p)·(1−d/p)^{n−1} = (1−d/p)^n`. ∎ With `p ≫ dn`, this is `≈ 1`.

Applied to the arithmetized 3-CNF `Φ` (degree `d ≤ 3m`), with a prover-chosen prime `p ∈ (2^n, 2^{2n}]`
(so the integer count equals the count mod `p`), this gives `#SAT ∈ IP`, hence coNP ⊆ IP and PH ⊆ IP.

## PSPACE ⊆ IP via TQBF

`Ψ = Q_1 x_1 … Q_n x_n φ(x)` (PSPACE-complete TQBF, `φ` a 3-CNF). Arithmetize the quantifiers:
`∀x_i ↦ ∏_{x_i∈{0,1}} (·) = (·)|_{x_i=0}·(·)|_{x_i=1}` and
`∃x_i ↦ ⊔⊔_{x_i∈{0,1}}(·) = 1−(1−(·)|_{x_i=0})(1−(·)|_{x_i=1})`. Then `Ψ` is true iff the operator-applied
arithmetic value is `1`.

**The degree wall and the fix.** The product/OR operators *multiply* the two branches, so a variable's
degree **doubles** per operator, reaching `2^n · 3m` — untransmittable, breaking even completeness. Fix:
since `x^k = x` on `{0,1}`, insert a **linearization (degree-reduction) operator** after each quantifier
for every live variable,
> `L_i(P) = x_i · P(…,x_i=1,…) + (1−x_i) · P(…,x_i=0,…)`,

the unique polynomial of degree `≤ 1` in `x_i` that agrees with `P` on `{0,1}` (single-variable Lagrange
interpolation through `0,1`). This restores per-variable degree `≤ 1` without changing any cube value.
The verified object is the `O(n²)`-operator string
`1 = Q'_1 L_1 Q'_2 L_1 L_2 Q'_3 L_1 L_2 L_3 … Q'_n L_1…L_n Φ`,
where each `Q'_i` is `∏_{x_i}` for `∀x_i` or `⊔⊔_{x_i}` for `∃x_i`.

**Protocol.** Maintain a running claim `v` (start `v=1`). For each operator on variable `x_i`, the prover
sends a univariate `P̂(x_i)`; the verifier checks operator-consistency against `v`
— `P̂(0)P̂(1)=v` (`∏`), `1−(1−P̂(0))(1−P̂(1))=v` (`⊔⊔`), or `r_i P̂(1)+(1−r_i)P̂(0)=v` at the current
binding (`L_i`) — then binds `x_i` to a fresh random `r_i` and sets `v ← P̂(r_i)`. At the end all
variables are random; the verifier evaluates `Φ(r_1,…,r_n)` itself and checks it against `v`.

**Completeness:** honest prover ⇒ accept w.p. 1. **Soundness:** a false claim forces a lie about a round
polynomial. Quantifier rounds have degree `≤ 1`; the innermost `n` linearization rounds, where `Φ` enters,
have degree `≤ 3m`; all other linearization rounds have degree `≤ 2`. The lie survives a fresh random
point with probability `≤ deg/p`, so the union bound gives
`Pr[falsely accept] ≤ n/p + 3mn/p + (2/p)Σ_{i=1}^{n-1} i = (3mn+n²)/p`, negligible for a poly-bit prime
`p ≫ 3mn+n²`. Hence TQBF ∈ IP, so **PSPACE ⊆ IP**.

## IP ⊆ PSPACE (the easy direction)

For any verifier `V`, the maximum acceptance probability is the value of a game tree of polynomial depth
(poly rounds), branching `≤ 2^{poly}` (poly-length messages): prover nodes take the **max** over
children, verifier nodes take the coin-weighted **average**, leaves are accept/reject. Evaluate it
depth-first, reusing space — computable in polynomial space. Root `> 2/3` ⇒ accept, `< 1/3` ⇒ reject.
So `IP ⊆ PSPACE`. With `PSPACE ⊆ IP`, **IP = PSPACE**. ∎

## Verifier sketch

```python
# IP = PSPACE.  Verify TQBF  Psi = Q_1 x_1 ... Q_n x_n phi(x)  (phi a 3-CNF, m clauses)
# in F_p, p a poly-bit prime the verifier certifies.

def lit_value(binding, lit, F):               # lit = (var, negated?)
    x = binding[lit.var]
    return F.sub(1, x) if lit.negated else x

def Phi(binding, clauses, F):                 # arithmetized 3-CNF: AND of clauses (product)
    val = 1
    for clause in clauses:                    # clause = list of (var, negated?)
        all_false = prod(F, (F.sub(1, lit_value(binding, lit, F)) for lit in clause))
        val = F.mul(val, F.sub(1, all_false)) # 1 - (1-A)(1-B)(1-C), degree <= 3
    return val                                # total degree <= 3m

def verify_TQBF(Psi, prover, F):
    ops = build_operator_string(Psi)          # Q'_x1 L1 Q'_x2 L1 L2 ... ; O(n^2) ops
    binding, v = {}, 1                         # running claim: stripped expression == 1
    for op, i in ops:
        bound = degree_bound(op, i, Psi)                    # 1, 2, or 3m
        Phat = prover.send_univariate(op, i, dict(binding)) # univariate in x_i
        if degree(Phat) > bound: return False
        if op == 'PROD':                                   # forall : product over {0,1}
            if F.mul(ev(Phat,0,F), ev(Phat,1,F)) != v: return False
        elif op == 'OR':                                   # exists : 1-(1-.)(1-.)
            if F.sub(1, F.mul(F.sub(1,ev(Phat,0,F)), F.sub(1,ev(Phat,1,F)))) != v: return False
        else:                                              # L_i : linearization check
            r = binding[i]
            if F.add(F.mul(r,ev(Phat,1,F)), F.mul(F.sub(1,r),ev(Phat,0,F))) != v: return False
        r_i = F.random()                       # fresh random point => per-round error <= deg/p
        binding[i] = r_i
        v = ev(Phat, r_i, F)                    # carry claim to the reduced sub-problem
    return Phi(binding, Psi.clauses, F) == v    # final self-check at random point

def value_of_game_tree(history, V):             # IP subseteq PSPACE: poly depth, reused space
    if is_leaf(history):  return 1 if V.accepts(history) else 0
    if prover_moves_next(history):              # prover maximizes
        return max(value_of_game_tree(history+m, V) for m in prover_messages(history))
    return weighted_average(value_of_game_tree(history+c, V)  # verifier averages over coins
                            for c in coin_outcomes(history))
```

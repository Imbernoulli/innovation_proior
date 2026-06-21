# IP = PSPACE

## The Theorem

`IP = PSPACE`: the languages with polynomial-round interactive proofs are exactly the languages decidable in polynomial space.

The easy direction is `IP subseteq PSPACE`. For any fixed verifier, the optimal prover acceptance probability is the value of a polynomial-depth game tree: prover nodes take maxima, verifier-randomness nodes take averages, and leaves are accept/reject. A depth-first evaluation reuses space, so the value is computable in PSPACE.

## The Protocol Idea

Reduce every PSPACE computation to TQBF:

`Psi = Q_1 x_1 ... Q_n x_n phi(x_1,...,x_n)`,

with `phi` a 3-CNF. Arithmetize `phi` over a large prime field:

- `x AND y -> XY`
- `NOT x -> 1-X`
- `x OR y -> 1-(1-X)(1-Y)`

The resulting polynomial `Phi` agrees with `phi` on `{0,1}^n` and can be evaluated compactly from the clauses.

Arithmetize quantifiers:

- `forall x` becomes product over `x in {0,1}`.
- `exists x` becomes `1-(1-a)(1-b)` over the two branch values.

Naively stripping these operators causes exponential degree growth. The proof's critical step is degree reduction on the Boolean cube:

`L_i(P) = x_i P(...,1,...) + (1-x_i) P(...,0,...)`.

`L_i(P)` agrees with `P` at `x_i=0,1` and is linear in `x_i`. Interleaving these reductions keeps all round polynomials low-degree while preserving the quantified Boolean truth value.

## Verification

The verifier maintains a running claim `v` for the remaining operator suffix.

- Universal round: prover sends `p(x_i)`; verifier checks `p(0)p(1)=v`.
- Existential round: verifier checks `1-(1-p(0))(1-p(1))=v`.
- Linearization round: verifier checks `r_i p(1)+(1-r_i)p(0)=v` at the current binding.

After each check, the verifier samples a fresh random field point `r_i`, sets `v = p(r_i)`, and continues. At the end, it evaluates `Phi(r_1,...,r_n)` itself and accepts iff it equals the final claim.

## Soundness

If the original claim is false, a cheating prover must at some round send a univariate different from the true low-degree polynomial. Two different degree-`d` univariates agree on at most `d` field points, so a fresh random challenge lets the lie survive with probability at most `d/q`.

With Katz's operator proof bounds, the total false-accept probability is at most:

`(3mn+n^2)/q`.

Choosing a polynomial-bit prime `q` much larger than `3mn+n^2` gives bounded error, and repetition amplifies it.

## Why It Was Distinctive

The proof does not trust the prover and does not ask for a static certificate. It turns a PSPACE computation into algebraic low-degree claims, uses interaction to reduce exponentially large assertions one random challenge at a time, and uses finite-field root-counting to police lies. That non-black-box arithmetization step is why the theorem goes beyond earlier NP/proof-system examples and why it bypasses relativized oracle barriers.

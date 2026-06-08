# Zero-Knowledge Interactive Proofs (the simulation paradigm)

## The problem

A classical NP proof of `x ∈ L` over-delivers: the certificate it transfers is a *witness*
(a satisfying assignment, a Hamiltonian tour, a factorization), which the verifier re-checks and so
ends up holding — far more than the single fact `x ∈ L`. The goal is to prove `x ∈ L` while conveying
*nothing beyond its truth*, and to make "nothing beyond its truth" a precise, provable property.

## The key idea

"Knowledge" is **computational and resource-bounded**: the verifier learns nothing extra if everything
it sees could have been produced by itself in polynomial time, given only that `x ∈ L`. Everything it
sees is its **view** = (its own coin tosses) + (all messages received), a probability distribution. A
protocol is **zero-knowledge** if for *every* polynomial-time (possibly cheating) verifier `B'`,
holding any auxiliary input `H`, there is an efficient **simulator** `M` that, *without the prover*,
outputs a distribution indistinguishable from `B'`'s real view `View_{A,B'}(x,H)`. If the view is
reproducible without the prover, the interaction transferred nothing the verifier could not already
have computed. Interaction + (secret) randomness make this possible: the verifier is convinced by the
prover *passing random challenges* (no single transcript is a re-checkable certificate), and the
simulator wins by **rewinding** — guessing the verifier's challenge, building a transcript *backwards*
from a chosen answer, and retrying if the guess was wrong (impossible for a real prover, which must
answer *after* the challenge and needs the secret to answer).

## Definitions

**Interactive proof system.** `(A,B)` with prover `A` (unbounded) and verifier `B` (probabilistic,
poly-time in `|x|`), sharing input `x`, each with a private random tape, exchanging messages in turns.
`(A,B)` is an interactive proof system for `L` if:
- **Completeness:** `x ∈ L` ⇒ `B` accepts with probability `≥ 1 − |x|^{-k}` (every `k`, large `x`).
- **Soundness:** `x ∉ L` ⇒ for *every* prover strategy `A'` (arbitrary power), `B` accepts with
  probability `≤ |x|^{-k}` (every `k`). The error can be driven to `2^{-|x|}` by repetition.
The class of such `L` is **IP**.

**Indistinguishability of distribution families** `{U(x)}`, `{V(x)}` (parameter `x` ranges over `L`):
- **Equal** ⇒ *perfect*.
- **Statistical:** `Σ_α |Pr[U(x)=α] − Pr[V(x)=α]| < |x|^{-c}` for every `c`, large `x` (infinite-time
  judge, poly-many samples).
- **Computational:** for every poly-size nonuniform circuit family `{C_x}`,
  `|Pr[C_x(U(x))=1] − Pr[C_x(V(x))=1]| < |x|^{-c}` for every `c`, large `x`. (Nonuniform so the judge
  can have side information about `x` or protocol history wired in; samples of size 1 suffice.)

**View.** For a run of `(A,B')` on common input `x` and `B'`'s auxiliary input `H`, with random tapes
`ρ` (of `B'`) and messages `a_i` (from `A`), `b_i` (from `B'`), the view is
`View_{A,B'}(x,H) = (ρ, b_1, a_1, …, b_n, a_n)` — `B'`'s coins plus everything it saw. (Including `B'`'s
coins is essential — see the leak below.)

**Approximable.** A family `U={U(x)}` is *perfectly (statistically / computationally) approximable on
`L`* if a probabilistic Turing machine `M` running in *expected* polynomial time has `M(x)` equal to
(statistically / computationally indistinguishable from) `U(x)` for all `x ∈ L`.

**Zero-knowledge.** `(A,B)` is *perfectly (statistically / computationally) zero-knowledge on `L`* if
for *every* polynomial-time `B'`, the family `View_{A,B'}` is perfectly (statistically /
computationally) approximable on `L' = {(x,H) : x ∈ L, |H| = |x|^c}`. The property depends only on the
prover `A`, against all verifiers `B'`. A *zero-knowledge proof system* is an interactive proof system
that is also zero-knowledge.

**Why coins must be in the view (the leak).** For `L` = composites: `B` picks random `x ∈ Z*_n`, sends
`a = x² mod n`; `A` returns a random square root `y` of `a`. The message *text* `(a,y)` is trivially
simulatable (`pick y, set a=y²`), but with the coins `x` in the view, `y ≠ ±x` with probability `≥ ½`,
and then `gcd(x+y,n)` is a nontrivial factor of `n` — the view leaks the factorization. Text-only would
falsely look zero-knowledge; coins-in-view exposes the leak.

## Number-theoretic background

`Z*_x = {y : 1 ≤ y < x, gcd(x,y)=1}`. `y` is a quadratic residue (QR) mod `x` if `y ≡ w² (mod x)`;
`Q_x(y) = 0` if QR, `1` if nonresidue (QNR). Facts: (1) `y` QR mod `x` iff QR mod every prime factor;
(2) with the factorization, `Q_x` is poly-time; (3) the Jacobi symbol `(y/x)` is poly-time *without*
the factorization, and `(y/x) = −1 ⇒ Q_x(y) = 1` — the hard case is `(y/x) = +1`, where deciding `Q_x`
is the quadratic residuosity problem (conjectured ≈ factoring); (4) all residues have equally many
roots; (5) residues are closed under multiplication and inverse, and multiplying a residue by a
nonresidue gives a nonresidue. Languages:
`QR = {(x,y) : Q_x(y)=0}`, `QNR = {(x,y) : y ∈ Z*_x, (y/x)=+1, Q_x(y)=1}` — both in `NP ∩ co-NP`, not
known in BPP.

## The proven example: perfect zero-knowledge for QR

Repeat `m = |x|` times:
1. `A` sends a random quadratic residue `u` mod `x` (knowing a root `s`, `u = s²`).
2. `B` sends a random challenge bit `bit`.
3. If `bit = 0`, `A` sends a random root `w` of `u` (`w² ≡ u`); if `bit = 1`, a random root `w` of
   `u·y` (`w² ≡ uy`, possible because `y` is a residue so `uy` is).
4. `B` accepts the round iff (`bit=0`: `w²≡u`) or (`bit=1`: `w²≡uy`). Accept iff all `m` rounds pass.

**Completeness.** `(x,y) ∈ QR` ⇒ `y` and `u` are residues ⇒ `uy` is a residue ⇒ `A` always has the
needed root ⇒ `B` accepts with probability 1.

**Soundness (≤ 2^{-m}).** If `y` is a nonresidue, `u` and `uy` cannot both be residues: otherwise
`y = (uy)u^{-1}` would be a residue. Thus at most one challenge is answerable. `A'` commits to `u`
before seeing `bit`; the fair coin hits the unanswerable challenge with probability `≥ ½`. Over `m`
rounds, acceptance of a false claim has probability `≤ 2^{-m} = 2^{-|x|}`.

**Perfect zero-knowledge (the simulator, with rewinding).** For any poly-time `B'` (whose challenge is
a deterministic `bit = f(x,y,H,history,u)`), the simulator `M` reproduces the view *identically* in
expected poly time:

```
for each of the m rounds:
    DO FOREVER:                       # rewinding; expected 2 iterations
        bit' := random in {0,1}       # guess the challenge first
        w'   := random in Z*_x        # the root to reveal first
        if bit' = 0: u' := w'^2  mod x            # u' a random residue, w' a root of u'
        else:        u' := w'^2 * y^{-1} mod x    # u'*y = w'^2, w' a root of u'y; u' a random residue
        if f(x,y,H,view,u') = bit':   # B''s real challenge match the guess?
            append (u', bit', w'); break          # keep round; discard+rewind otherwise
```

Correctness: in both branches `u'` is a *uniformly* random residue (every residue has equally many
roots, and `y` a residue ⇒ `y^{-1}` permutes residues), so `B'`'s challenge `f(…,u')` is independent of
the guessed fair bit. The challenge need not be balanced, but the guess matches it with probability
exactly `½`; conditioned on a match, `(u',bit',w')` has exactly the real round distribution. Hence
`M`'s output equals `View_{A,B'}` — perfect, expected polynomial time.

## Companion: statistical zero-knowledge for QNR

`B` generates elements `w` of two types — type 1 `w = r²` (residue), type 2 `w = r²y` — and the
unbounded `A` classifies `w` by testing residuosity; if `(x,y) ∉ QNR` (`y` a residue) both types are
residues and `A` guesses `≤ ½` (soundness `≤ 2^{-m}`). To stay zero-knowledge, `B` must *prove* it
knows each `w`'s type — that it knows a root of `w` (type 1) or of `wy^{-1}` (type 2) — via a
cut-and-choose over randomly ordered pairs containing one value `r_{j1}²` and one value `r_{j2}²y`: on
challenge `i_j=0`, `B` opens the pair (well-formedness); on `i_j=1`, `B` reveals one root showing that
`w` times one component of the pair is a residue (consistency), without revealing the type bit. The
simulator `M` outputs the correct `answer = Q_x(w)` without unbounded power by finding a *second
special string* `i' ≠ i` (one `B'` answers): the two differing openings (`i_j=0` vs `i'_j=1`)
algebraically determine `Q_x(w)` in poly time. `M` rejection-samples for `i'`; with `k` special
strings, the second-string search takes expected `2^m/(k−1)` tries when `k ≥ 2`, and its unconditional
expected contribution is polynomial because the first `i` is special with probability `k/2^m`. The
only failure is `i` unique-special (`k=1`), probability `≤ 1/2^m`, giving statistical distance
`≤ 2m/2^m` over `m` iterations — statistical zero-knowledge. (Spending the already-budgeted `≈ 2^m`
time on factoring in that lone case makes it perfect.)

## What it establishes

A proof can convey *only* the validity of its statement: for `QR` (perfectly) and `QNR` (statistically)
there exist interactive proof systems that are complete, sound to `2^{-|x|}`, and zero-knowledge —
the first zero-knowledge proofs for languages not known to be efficiently recognizable. The simulation
paradigm — "the verifier's entire view is reproducible without the prover, hence it learned nothing it
could not compute itself" — is the definition's core, inherited from semantic-secure encryption and
realized by rewinding.

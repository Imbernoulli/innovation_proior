Let me start from the thing that bugs me about proofs. When I say a language `L` is in NP, what I'm
really saying is: there's a proof system. A prover with all the time in the world writes a string `α`,
and a verifier checks `x` and `α` in time polynomial in `|x|`, and accepts iff `x ∈ L`. Fine. That's
the standard picture of what it means to "communicate a proof." To prove a formula `F` is satisfiable,
the prover sends a satisfying assignment `I`; the verifier plugs it in and is convinced. To prove a
graph is Hamiltonian, send a tour.

But look at what just happened. I asked to be convinced of one bit — "`F ∈ SAT`" — and I walked away
holding a *satisfying assignment*. That's enormously more than the bit. I might not have been able to
find a satisfying assignment myself in polynomial time; finding one is the hard part. The proof handed
me the witness. Same with the tour. The proof over-delivers, every time, and it over-delivers exactly
the thing that was hard to know. The proof leaks its witness, and the witness is the secret.

So the question I actually want to answer: can a proof convey *only* that the statement is true, and
nothing else? Can I prove `x ∈ L` and have you end up knowing the single fact `x ∈ L` and nothing you
couldn't already have figured out?

The trouble is I don't even know how to *say* "nothing else." What is "knowledge" here? My first
instinct is Shannon information — measure the bits the verifier receives. But that's clearly wrong. If
I just *tell* you a true bit, "yes, `x ∈ L`," you've received a bit of information and that's exactly
what I wanted you to receive — that's fine, that's the whole point. Conversely, I could hand you a
gigabyte of random noise and you'd "receive" a lot of Shannon information and learn nothing. So raw
information content isn't it. There's another tradition — the model-theoretic one, where an agent
"knows" everything that logically follows from what it's told. But that agent has unlimited deductive
power; it knows all consequences instantly. In that world a proof of `x ∈ L` for `L ∈ NP` reveals
nothing new, because the verifier "already knew" `x ∈ L` as a logical consequence of `x` and the axioms
— it just hadn't *noticed*. That can't be my notion either, because the whole danger I'm worried about
— being handed a satisfying assignment — is a danger precisely *because* the verifier is
computationally limited and couldn't have found that assignment itself.

That last clause is the hinge. The verifier is a polynomial-time machine. The thing I want to forbid
the proof from transferring is something the verifier *could not have computed by itself, in
polynomial time, from `x` alone*. The satisfying assignment is dangerous because finding it is hard;
the bit "`x ∈ L`" is harmless because... wait, finding *that* bit is also hard (that's what makes `L`
nontrivial). Hmm. So I can't say "the verifier learns nothing it could compute" in the naive sense,
because by that standard learning `x ∈ L` is itself "learning something." I need to be more careful:
the verifier is *allowed* to learn the one fact it came for, `x ∈ L`. What it must not learn is
anything *extra* — anything beyond that fact, that it couldn't have computed *given that fact*.

Let me sharpen it to: knowledge is computational, and resource-bounded. The verifier learns nothing
extra if everything it sees in the interaction is something it could have produced for itself, in
polynomial time, *given only that `x ∈ L`*. Not given a witness — given the bare membership fact.

Now, how do I make "everything it sees" precise, and how do I make "could have produced for itself"
precise?

Let me think about what the verifier sees. In a static NP proof it sees `α`. But I already know a
static proof can't work — whatever string the verifier checks and accepts *is* a witness, by
definition re-checkable, so it can't be hidden. If the only thing the prover ever sends is a string
the verifier validates against `x`, that string carries the witness's content. I need the verifier to
become convinced *without* ever holding a single re-checkable certificate.

How can you be convinced of something without holding a checkable proof of it? You can be convinced by
*watching someone do something that would be impossible if the statement were false*. That's the move:
make the verifier ask the prover to do a randomly chosen task, where a true statement lets the prover
succeed every time, but a false statement lets it succeed only by luck. Then no single transcript is a
"proof" you could re-check — it's just a record of the prover passing a challenge — and yet, across
enough random challenges, only a prover who could actually pass them all (which requires the statement
to be true) survives. So I need **interaction** (the verifier poses challenges, the prover responds)
and I need **randomness** (the challenges are random, so the prover can't have pre-arranged the
answers). Static, deterministic NP proofs are exactly the case with no interaction and no randomness,
and that's the case that's forced to leak. The moment I add a coin the verifier flips and a back-and-
forth, I get room to maneuver.

And there's a subtlety about *whose* coins and whether they're secret. If the verifier publishes its
random challenge before the prover commits to anything, the prover sees the challenge coming. There's
a model like that — Arthur publishes random coins, Merlin sees them. That's good enough to *recognize*
hard languages. But for *hiding*, I suspect I want the verifier's coins to be its own and, where it
matters, secret — the prover answers a challenge it couldn't have anticipated, and the verifier holds
something the prover didn't get to shape. Let me keep that in mind and come back to it; it'll turn out
to matter for the hiding, not for the recognizing.

Let me first nail what an interactive proof system even *is*, the completeness and soundness, before I
worry about hiding. Two machines: a prover `A`, computationally unbounded, and a verifier `B`, bounded
to time polynomial in `|x|`. They share the input tape `x`; each has a private random tape it reads
left to right (flipping a coin = reading the next bit); they pass strings back and forth on
communication tapes, taking turns. `B` finally outputs accept or reject. I'll say `(A, B)` is an
interactive proof system for `L` if:
- **Completeness:** for `x ∈ L`, `B` halts and accepts with probability at least `1 − |x|^{-k}`, for
  every constant `k` and all long enough `x`. (The honest prover almost always convinces the honest
  verifier.)
- **Soundness:** for `x ∉ L`, for *any* prover strategy `A'` whatsoever — arbitrary power, deviating
  however it likes — `B` accepts with probability at most `|x|^{-k}`, for every `k`. (No cheating
  prover can sell a false statement.)

The probabilities are over the coin tosses of `A` and `B`. Note soundness has to hold against an
*arbitrary* `A'`, not the honest `A` — `B` doesn't get to trust who it's talking to; it only trusts
its own coins. And I can always drive the error down to `2^{-|x|}` by repeating and taking majority,
so the exact polynomial in the bound isn't precious. Call the class of languages with such systems IP.
This already contains languages — like matrix-group nonmembership, or graph nonisomorphism — that
aren't known to have static NP or BPP proofs, so interaction plus secret randomness genuinely buys
recognizing power. Good. But recognizing power is not my problem. My problem is the hiding.

So: zero-knowledge. I want to define "`(A, B)` releases no knowledge." Let me look hard at what the
verifier ends up holding after a run. It holds its own coins, and it holds every message it received.
Call that bundle — the verifier's coins together with all messages, in order — the verifier's **view**.
The view is the complete record of the verifier's experience. If knowledge is going to leak, it leaks
*into the view*; there's nowhere else for it to go. So the right object to reason about is the *view*,
and because the prover and verifier flip coins, the view isn't a fixed string — it's a *probability
distribution* over strings.

Now, "the verifier could have computed everything itself." I want to say: the distribution of the view
is something the verifier could have generated on its own, in polynomial time, from `x` alone (given
`x ∈ L`), with no prover in the room. If there's an efficient procedure `M` — a *simulator* — that,
given only `x`, outputs strings distributed just like the real view, then the real interaction gave the
verifier nothing it couldn't have manufactured by talking to itself. Whatever it could compute *after*
the interaction (from the view), it could already compute *before* (by first simulating the view, then
computing). So the interaction is informationally idle. That's zero-knowledge.

Wait. This is exactly the move Goldwasser and Micali made for encryption. Their whole definition of a
secure cryptosystem was: *whatever an eavesdropper can compute about the message from the ciphertext,
it can compute without the ciphertext* — because the ciphertext's distribution is reproducible without
the message. They formalized "the adversary learns nothing" as "there's a way to compute the same
thing without the secret," and they measured "looks the same" by indistinguishability of distributions
to a bounded judge — a poly-size circuit that can't tell an encryption of `m₁` from an encryption of
`m₂` by more than `1/poly`. That's precisely the template I need, lifted from "the ciphertext reveals
nothing about the hidden message" to "the whole transcript-and-coins reveals nothing beyond the public
fact `x ∈ L`." The simulator is the analogue of "compute the same thing without the ciphertext." I'll
reuse their indistinguishability-of-distributions hammer wholesale.

So I need to pin down "the simulator's output distribution is indistinguishable from the real view."
There are degrees of "the same," and they correspond to how much I demand. Let me lay out a judge who
gets a sample from one of two distributions `U(x)`, `V(x)` and must guess which. Three regimes:
- **Equal:** `U(x)` and `V(x)` are literally the same distribution. The judge is helpless no matter how
  much time or how many samples it has. This gives *perfect* zero-knowledge.
- **Statistically indistinguishable:** the total variation distance is tiny —
  `Σ_α |Pr[U(x)=α] − Pr[V(x)=α]| < |x|^{-c}` for every `c` and long `x`. Now even a judge with infinite
  computing power but only polynomially many samples gets essentially no signal. This gives
  *statistical* zero-knowledge.
- **Computationally indistinguishable:** no poly-size circuit `C` tells them apart by more than
  `1/poly`: `|Pr[C(U(x))=1] − Pr[C(V(x))=1]| < |x|^{-c}`. The judge is now resource-bounded. This is
  the weakest and most general, and it's the one that matters in the real world. Call it *computational*
  zero-knowledge, or just zero-knowledge.

Equal ⊂ statistical ⊂ computational, in strength of the guarantee. And I should be careful about the
judge for the computational case: I'll make it a *nonuniform* family of circuits, one per input length,
rather than a single uniform machine. Why nonuniform? Because the verifier I'm protecting against might
have arbitrary side information about `x` baked in, or might be sitting in the middle of some larger
protocol whose history it remembers — and I want "indistinguishable" to mean indistinguishable *even
to a judge with that stuff hard-wired in*. A circuit family lets me wire in whatever the adversary
could know. This matches how Goldwasser–Micali set up their adversaries as circuits, so it's the same
choice for the same reason. (And it suffices to hand the judge a single sample at a time — feeding it
polynomially many independent samples is no more powerful than one, since a circuit fed many samples
is just a bigger circuit fed one.)

Now the part I have to get right or the whole definition is worthless: the verifier might *cheat*. A
real adversary won't politely run the honest verifier's program. It'll deviate — pick its challenges
adversarially, in whatever way maximizes what it can squeeze out of the prover. So zero-knowledge can't
just say "the honest verifier learns nothing." It has to say: for *every* polynomial-time verifier
strategy `B'`, even a malicious one, the view `B'` obtains is simulatable. Notice this makes the
zero-knowledge property depend only on the *prover* `A`, not on the honest `B` at all — it's really `A`
that is or isn't zero-knowledge, against all possible `B'`. (Mirror image of soundness, which depended
only on `B`, not on `A`.) For every cheating `B'` I must exhibit a simulator `M` whose output is
indistinguishable from `B'`'s real view.

And one more thing the cheating verifier might have: prior knowledge. Suppose `B'` already holds some
side information `H` about `x` — maybe it learned `H` from the history of some earlier protocol, maybe
someone told it. I want the proof to *stay* zero-knowledge in the presence of `H`: interacting with `A`
shouldn't let `B'` combine `H` with the interaction to compute something it couldn't have computed from
`H` alone. So I'll give `B'` an extra input tape holding `H` (length polynomial in `|x|`), and I'll
demand the simulator gets `H` too: the real view `View_{A,B'}(x, H)` must be approximable by `M(x, H)`.
If I *didn't* give `B'` this `H`, I couldn't prove the facts I obviously want — like "running the
protocol twice in a row is still zero-knowledge," where the second run's verifier has the first run's
transcript as history. The `H` is what makes the definition compose. (This need for the auxiliary input
is subtle enough that I keep rediscovering it from a few directions.)

Let me also stress-test the definition: I claimed the view is the verifier's coins plus the messages.
Do I really need the *coins* in there, or would the transcript of messages alone do? The composite-number
example decides the matter. Take `L` = composite integers. On input `n`, the verifier secretly picks a
random `x` in `[1, n]` relatively prime to `n`, computes `a = x² mod n`, and sends `a`. The prover
(unbounded) sends back a random square root `y` of `a mod n`. Now: the *text* of this interaction —
just `a` and `y` — is trivially simulatable; I can pick `y` at random and set `a = y² mod n` without any
prover. So if the view were only the message text, this would look zero-knowledge. But include the
verifier's coins `x` in the view, and watch: `y` is a random square root of `x² mod n`, chosen
independently of `x`. A composite `n` has at least four square roots of `a`; with probability at least
`½`, the `y` the prover returns is a root *different from `±x`*. When that happens, `gcd(x + y, n)` is a
nontrivial factor of `n` — because `x² ≡ y² (mod n)` means `n | (x−y)(x+y)`, yet `n` divides neither
factor. So the verifier, from its view, factors `n` with probability `½`. That is enormous knowledge —
it's the factorization — and no efficient simulator could produce a view with this property unless
factoring is easy. The text-only view hid the leak; the coins-in-view exposes it. So the coins *must*
be in the view. Good — the definition is now demanding genuine secrecy, not bookkeeping.

I should also notice: I've been assuming the simulator runs in strict polynomial time, but maybe I have
to let it run in *expected* polynomial time. Consider a protocol where the prover must produce a string
`β` satisfying some rare predicate `P` — say `P` holds for a `n^{-20}` fraction of strings, and the
only way anyone knows to find such a `β` is to keep drawing random strings until one works. A simulator
trying to reproduce the prover's message has the same problem: it must rejection-sample until it lands a
`β` with `P(β)`. That's *expected* polynomial time, not strict. So I'll allow the simulator expected
poly time. (Hold this thought — it's going to be exactly what one of my protocols needs.)

OK. I have a definition. Now I have to prove it isn't vacuous: exhibit a protocol that I can *prove*
satisfies completeness, soundness, *and* zero-knowledge, for a language where it actually matters —
something not known to be efficiently decidable, so that "hides the witness" has teeth.

Let me pick quadratic residuosity. Recall `Z*_x = {y : 1 ≤ y < x, gcd(x,y)=1}`; `y` is a quadratic
residue (QR) mod `x` if `y ≡ w² (mod x)` for some `w`, else a nonresidue. Define `Q_x(y) = 0` if `y` is
a QR, `1` otherwise. The reasons this is the right test case: the predicate `Q_x` is the kind of thing
where, given `x`'s factorization you can decide it in polynomial time, but the Jacobi symbol `(y/x)` —
computable *without* the factorization — only resolves the easy case. If `(y/x) = −1`, `y` is certainly
a nonresidue. But when `(y/x) = +1`, deciding whether `y` is a residue is the quadratic residuosity
problem, conjectured as hard as factoring; the best algorithm is to factor `x` first. So
`QR = {(x,y) : y is a QR mod x}` is in `NP ∩ co-NP` but not known to be in probabilistic poly time. If I
can prove `(x,y) ∈ QR` to you in zero-knowledge, that's the first zero-knowledge proof for a language
not known to be efficiently recognizable. The obvious NP proof would send `x`'s factorization or a
square root — exactly the secrets I must not leak. Perfect target.

I also want two algebraic facts to lean on. First, every quadratic residue mod `x` has the *same*
number of square roots, independent of which residue it is. Second, the residue class is closed under
multiplication and inverse, and multiplying a residue by a nonresidue stays a nonresidue. In the form
I will actually need: if both `u` and `uy` are residues, then `y = (uy)u^{-1}` is a residue too; and if
`u` is a residue while `y` is not, then `uy` cannot be a residue. These are the load-bearing gears.

I (the prover) want to convince you `(x,y) ∈ QR`, i.e. `y` is a residue, without revealing a root of
`y`. A random challenge should force me to demonstrate either that some public element is a residue or
that multiplying it by `y` is a residue, because knowing a root of `y` lets me answer both while a false
claim lets me answer at most one. Repeat the following `m = |x|` times:
- I pick a random quadratic residue `u` mod `x` — I generate it so that I know one of its square
  roots — and send you `u`.
- You flip a coin `bit ∈ {0,1}` and send it to me as your challenge.
- If `bit = 0`, I send you a random square root `w` of `u` (so `w² ≡ u`). If `bit = 1`, I send you a
  random square root `w` of `u·y mod x` (so `w² ≡ uy`).
- You check: if `bit = 0`, that `w² ≡ u`; if `bit = 1`, that `w² ≡ uy`. Accept iff every round checks
  out.

Why can the honest prover always answer? If `(x,y) ∈ QR`, then `y` is a residue, so `uy` is a residue
(residue times residue), so `uy` has a square root and I can produce it. And `u` is a residue by
construction, so it has a root too. So for either challenge bit I have the root on hand. **Completeness**
holds: an honest prover passes every round, `B` accepts with probability 1. Done.

**Soundness.** Suppose `y` is *not* a residue. Then for any `u`, it cannot be that *both* `u` and `uy`
are residues — because if `Q_x(u) = 0` and `Q_x(uy) = 0`, multiplicativity forces `Q_x(y) = 0`,
contradicting that `y` is a nonresidue. So at most one of the two possible challenges (`bit = 0` asking
for a root of `u`, `bit = 1` asking for a root of `uy`) is answerable — one of `u`, `uy` is a
nonresidue and has *no* square root, so no `w` can satisfy that check, not even for an unbounded prover.
Crucially, I (the cheating prover `A'`) must send `u` *before* seeing your bit. So whatever `u` I pick,
you flip a fair coin afterward, and with probability at least `½` you ask the question I can't answer.
Per round, a cheating prover survives with probability at most `½`. Over `m = |x|` independent rounds,
the chance you're fooled into accepting a false `(x,y)` is at most `1/2^m = 2^{-|x|}` — below `|x|^{-k}`
for every `k`. **Soundness** holds. (This is an interactive proof system for QR.)

Now the real test: is it **zero-knowledge**? And not against an honest you — against an arbitrary
polynomial-time cheating `B'` that picks its bits however it likes, possibly as some complicated
deterministic function of everything it's seen, possibly using side information `H`. I have to build a
simulator `M` that, with no prover, produces output distributed exactly like `B'`'s view.

Let me first look at the view. It's `B'`'s random tape `R`, and then, round by round, the values
`U_i` (the `u` it received), `BIT_i` (the bit it sent — which, since `B'` is some fixed strategy, is a
deterministic function `f(x, y, H, history, u_i)` of what `B'` has seen), and `W_i` (the root it
received). So the view is `R, U_1, BIT_1, W_1, …, U_m, BIT_m, W_m`.

The real interaction has the order that makes soundness work: the prover sends `u` *first*, then sees
the bit, then produces the matching root. The prover can do that because it secretly knows a root of
`y`. The simulator has *no* root of `y` and no prover. If it sends a `u` and then `B'` asks for a root
of `uy`, the simulator is stuck — it can't produce a root of `uy` without knowing one for `y`.

So the simulator cheats by reversing the order: pick the answer first, then derive a question that
matches it. Concretely, for the next round, having already fixed the view up to round `i`:

```
DO FOREVER:
    bit' := a random bit in {0,1}        # guess what B' will ask
    w'   := a random element of Z*_x     # the root I'll pretend to reveal
    if bit' = 0:
        u' := w'^2 mod x                 # u' is a random residue; w' is a root of it
    else:
        u' := w'^2 * y^{-1} mod x        # then u'*y = w'^2, so w' is a root of u'y; u' is a residue
    # Now feed u' to B' and see what bit it actually asks:
    if f(x, y, H, view-so-far, u') = bit':   # did my guess match B''s real challenge?
        output (u', bit', w') and HALT
    # else: throw this round away, rewind B', and try again
```

Why is this correct — why does the surviving `(u', bit', w')` have *exactly* the real conditional
distribution? Two things to check: that `u'` is distributed right, and that the guess matches with the
right probability.

First, `u'` is a uniformly random quadratic residue, in *both* branches. If `bit' = 0`, `u' = w'²` and
as `w'` ranges uniformly over `Z*_x`, `w'²` ranges over the residues — and since every residue has the
same number of roots, `w'²` is *uniform* over residues. If `bit' = 1`, `u' = w'² · y⁻¹`; as `w'²` is
uniform over residues and `y⁻¹` is a fixed element, `u'` is again uniform over residues (multiplying
the uniform-over-residues distribution by a fixed unit just permutes the residues, since `y` is a unit
and — because `(x,y) ∈ QR` — `y` is itself a residue, so `y⁻¹` maps residues to residues bijectively).
So regardless of my guess `bit'`, the `u'` I show `B'` is a uniformly random residue — *identically*
distributed to the real `u` an honest prover would send. That's the crux: `B'` cannot tell from `u'`
which branch I took, because the distribution of `u'` is the same either way.

Second, because `B'`'s challenge is a deterministic function of `(x, y, H, history, u')`, and `u'` has
the same distribution no matter which `bit'` I guessed, the real challenge `f(…, u')` is independent of
my guess `bit'`. It need not be balanced; `B'` could always ask 0. But for each fixed `u'`, my guessed
bit is still an independent fair bit, so it matches `f(…, u')` with probability exactly `½`. The loop is
a geometric trial with success probability `½`: it halts after `2` iterations in expectation. And
*conditioned on* halting, the output `(u', bit', w')` is precisely a uniform residue `u'`, the genuine
challenge `bit'` that `B'` produces on it, and a random root `w'` of (`u'` if `bit'=0`, `u'y` if
`bit'=1`) — which is exactly the real conditional distribution of that round's `(U_i, BIT_i, W_i)`.
Stitching the rounds together (and
carrying `B'`'s random tape `R` along, chosen once at the start), `M` outputs a view distributed
*identically* to the real `View_{A,B'}(x, y, H)`. Not approximately — *identically*. And `M` runs in
expected polynomial time (expected two tries per round, `m` rounds).

That's it. The protocol is **perfectly zero-knowledge** for QR. And I see now, concretely, why
interaction and randomness were essential and why secret coins matter: the simulator's whole power is
*rewinding* — it guesses `B'`'s random challenge, builds a transcript backwards from a chosen answer to
a matching question, and if `B'` would actually have asked something else, it rewinds `B'` and retries.
A real prover can't build the transcript backwards (it's pinned to answer *after* the challenge, and it
needs the secret to answer the hard branch); the simulator can, precisely because it controls `B'`'s
coins in its head and can take as many do-overs as it likes. The challenge being a fresh bit the prover
can't anticipate is what forces soundness; the same challenge being something the *simulator* can guess-
and-retry is what gives zero-knowledge. Same coin, two faces.

Let me push to the harder language: quadratic *non*residuosity. `QNR = {(x,y) : y ∈ Z*_x, (y/x)=+1,
Q_x(y) = 1}` — `y` has Jacobi symbol `+1` but is a nonresidue. The asymmetry: now the prover wants to
convince me a thing is a *non*residue, which is `co-NP`-ish, and the root-revealing move that worked for
QR (produce a root of `uy`) doesn't directly apply, because nonresidues don't have roots to show.

A nonresidue gives me no root to reveal, so I should stop asking the prover for a root and instead ask
it to classify samples. Let the *verifier* `B` generate test elements of two kinds. `B` picks a random
`r` and a bit, and forms `w = r² mod x` (type 1: a residue) or `w = r²y mod x` (type 2). If
`(x,y) ∈ QNR`, then type-2 elements `r²y` are nonresidues (residue times nonresidue) while type-1 are
residues — so the *unbounded* prover `A` can tell which type `w` is, by testing whether `w` is a
residue, and answer correctly every time. But if `(x,y) ∉ QNR` (i.e. `y` is secretly a residue), then
`r²y` is *also* a residue, so type 1 and type 2 are *both* residues, identically distributed, and `A`
cannot tell them apart better than guessing. So `A` answering `w`'s type correctly, every time,
convinces `B` that `y` must be a nonresidue. **Soundness** is exactly:
when `y` is a residue, `w` and the auxiliary material give `A` no information about the hidden type-bit,
so `A` predicts it with probability at most `½` per round, `1/2^m` over `m` rounds.

But — and this is the danger — for this to be zero-knowledge, `B` had better actually *know* the type
of each `w` it sends, and `B` had better have generated each `w` *honestly* as `r²` or `r²y`. If `B`
could send an *arbitrary* `w` (not of the prescribed form) and get `A` to tell it whether `w` is a
residue, then `B` has turned `A` into an oracle for the quadratic-residuosity predicate on inputs of
`B`'s choosing — `A` would be leaking exactly the hard-to-compute knowledge I'm trying to protect.
That's a catastrophic leak. So I have to *force* `B` to prove, for each `w`, that it knows which type
`w` is — that it knows a root of `w` (type 1) or a root of `wy⁻¹` (type 2) — *without* revealing which.

So `B` proves "I know a square root of `w` or a square root of `wy⁻¹`" by a cut-and-choose. Alongside
`w`, for each `j = 1, …, m`, `B` picks random `r_{j1}, r_{j2}` and forms `a_j = r_{j1}²`,
`b_j = r_{j2}²y`. Then it randomly orders the pair as `(a_j, b_j)` or `(b_j, a_j)`, so the prover sees
one residue and one residue-times-`y` but not a label saying which is which. `B` sends `w` and all the
pairs. Then `A` sends a random challenge vector `i = i_1 … i_m`. For each `j`: if `i_j = 0`, `B` opens
`pair_j` fully (reveals `r_{j1}, r_{j2}`), proving the pair really is one residue and one
residue-times-`y` — i.e. *well-formed*; if `i_j = 1`, `B` reveals a single square root showing that
`w` times one component of `pair_j` is a residue. If `w` was type 1, that component is the residue
`a_j`; if `w` was type 2, that component is the residue-times-`y` value `b_j`. The random ordering keeps
this consistency check from revealing the type-bit directly. `A` checks the revealed values are
consistent; if so, `A` (unbounded) computes `answer = Q_x(w)` and sends it; `B` checks `answer` equals
the type-bit it used for `w`. A cheating `B` that didn't honestly know `w`'s type can answer the
`i_j = 0` challenge or the `i_j = 1` challenge but not both for a malformed pair, so the cut-and-choose
catches it with probability `½` per pair — so an honest-looking `B` really does know each `w`'s type.

Now I have to simulate even a cheating `B'`. The expected-time simulator I anticipated earlier earns
its keep. The simulator `M` runs `B'`, receives `w` and the pairs, sends a random challenge `i`, and
`B'` replies with the openings `v`. The hard part: `M` must output the right `answer = Q_x(w)`, but `M`
is bounded — it can't just test residuosity of `w` (that's the hard problem!). So how does `M` learn
`Q_x(w)` without unbounded power?

Call a challenge string `i` *special* if `B'` actually responds to it (doesn't abort). `M` has already
seen `B'` respond to its sent `i`, so `i` is special. If `M` can find a *second* special string
`i' ≠ i`, then somewhere they differ in a coordinate `j` — say `i_j = 0` and `i'_j = 1`. From the
`i_j = 0` opening `M` learns roots `s,t` proving that `pair_j` contains `s²` and `t²y`, possibly in the
other order. From the `i'_j = 1` opening it learns a value `v'` such that `(v')² w^{-1}` is one of the
two components of `pair_j`. If `(v')² w^{-1} = s²`, then `(v'/s)² = w`, so `w` is a residue. If
`(v')² w^{-1} = t²y`, then `(v'/t)² = wy`, so `w = (v'/t)² y^{-1}` is a residue times a nonresidue and
is therefore a nonresidue. The two openings pin down `Q_x(w)` in polynomial time, without factoring.
So `M` just needs a second special string, then it can output the honest `answer`.

How does `M` find a second special string in expected polynomial time? It rejection-samples:

```
ALGORITHM (find a special i' ≠ i):
    repeat up to 2^m times: pick a random m-bit i'; if i' ≠ i and i' is special, STOP and return i'.
    if none found: test all m-bit strings in order, looking for any special i' ≠ i.
    if still none: i was the UNIQUE special string — output "?".
```

The expected-time analysis: fix `B'`'s inputs and random tape, and let `k` be the number of special
strings among the `2^m` possibilities. `M`'s originally-sent `i` was random, so it lands on a special
string with probability `k/2^m`. Conditioned on having a special `i` and on `k ≥ 2`, the search for a
*second* special string is like flipping a coin whose heads-probability is `(k−1)/2^m` (the fraction of
strings that are special and `≠ i`); the expected number of trials to get one is `2^m/(k−1)`. The
unconditional contribution is therefore about `(k/2^m) · (2^m/(k−1)) · poly = (k/(k−1)) · poly`, which
is polynomial for all `k ≥ 2`. The only bad case is `k = 1`: `i` is the *unique* special string, `M`
finds no second one and outputs "?". But that case happens only when `M`'s random `i` was *the* one
special string, probability `1/2^m`. So `M`'s output differs from the real view only on an event of
probability at most `1/2^m` per iteration — a statistical-distance gap of at most `2/2^m` per iteration,
`2m/2^m` over `m` iterations.
That's negligible. So `M` reproduces the view up to negligible statistical distance: **statistical
zero-knowledge** for QNR. (And with a little more work — when `M` discovers `i` is the unique special
string, have it spend the roughly `2^m` time it's already willing to spend on *factoring* `x` and then
deciding `Q_x(w)` directly — the gap closes entirely and it's perfect; the extra work doesn't change
the expected running time by more than a polynomial factor, since `M` only does it after already having
spent `2^m`.)

Let me step back and make sure I see the shape of what I've built, because the definition is the real
contribution and the protocols are the proof that it isn't empty. I wanted to prove a statement true
while conveying nothing beyond its truth. "Nothing beyond its truth" couldn't be Shannon information
(harmless bits count as information) and couldn't be model-theoretic knowledge (an unbounded knower
already knows everything); it had to be *computational and resource-bounded* — the verifier learns
nothing extra if everything it sees could have been produced by itself in polynomial time given only
`x ∈ L`. The complete record of "everything it sees" is its *view* (coins plus messages), a probability
distribution. The verifier learns nothing iff that distribution is reproducible by an efficient
*simulator* with no prover — and I borrowed exactly the indistinguishability-of-distributions machinery
from semantic-secure encryption to say "reproducible," in three grades (perfect / statistical /
computational). The definition has to quantify over *all* cheating polynomial-time verifiers and hand
each one its side information `H`, or it doesn't compose; and the view has to include the verifier's
coins, or it misses leaks like the `gcd(x+y, n)` factorization. The simulator may need expected (not
strict) polynomial time, because reproducing rare-event messages forces rejection sampling. And the one
mechanism that makes simulators possible — *rewinding*: guess the verifier's secret challenge, build a
transcript backwards from a chosen answer, and retry if the guess was wrong — is the same mechanism, run
in reverse, that makes a real prover unable to cheat. Then, to show the definition is alive, I proved
quadratic residuosity has a perfectly zero-knowledge interactive proof (the random-residue / coin /
root protocol, sound to `2^{-|x|}`, simulated identically by the rewinding loop) and quadratic
nonresiduosity a statistically zero-knowledge one (the two-types cut-and-choose, simulated by the
find-a-second-special-string loop) — the first zero-knowledge proofs for languages not known to be
efficiently recognizable.

I can now write the QR protocol and its simulator as the algorithms I would actually run.

```python
# Perfectly zero-knowledge interactive proof that (x, y) is in QR (y a quadratic residue mod x).
# A = prover (knows a square root of y), B = verifier (poly time). Repeat m = |x| rounds.
# Soundness error <= 2^{-m}; the simulator below reproduces B''s view IDENTICALLY.

import random
from math import gcd

def is_in_Zx(t, x):
    return 1 <= t < x and gcd(t, x) == 1

def random_unit(x):
    while True:
        t = random.randrange(1, x)
        if gcd(t, x) == 1:
            return t

def inverse(a, x):
    return pow(a, -1, x)

# ----- the interactive proof -----
def prover_round(x, y, root_y):           # root_y: a square root of y mod x (A's secret)
    s = random_unit(x)
    u = (s * s) % x                       # u: a random quadratic residue; A knows the root s of u
    def respond(bit):
        if bit == 0:
            return u, s                    # w with w^2 = u
        else:
            w = (s * root_y) % x           # w^2 = s^2 * y = u*y
            return u, w
    return u, respond

def verifier_challenge():
    return random.getrandbits(1)           # B's secret coin: the challenge bit

def verifier_check(x, y, u, bit, w):
    if bit == 0:
        return (w * w) % x == u % x        # w^2 == u
    else:
        return (w * w) % x == (u * y) % x  # w^2 == u*y

def run_QR_proof(x, y, root_y, m):
    for _ in range(m):
        u, respond = prover_round(x, y, root_y)
        bit = verifier_challenge()
        _, w = respond(bit)
        if not verifier_check(x, y, u, bit, w):
            return False                   # B rejects
    return True                            # B accepts (completeness: always, if (x,y) in QR)

# ----- the simulator: produces B''s view with NO prover and NO root of y -----
# B_prime(view_so_far, u) -> bit  models an arbitrary cheating verifier's challenge (deterministic
# in its coins/history/aux input H, which we treat as fixed inside B_prime).
def simulate_QR_view(x, y, B_prime, m):
    view = []
    for _ in range(m):
        while True:                        # rewinding loop: expected 2 iterations
            bit = random.getrandbits(1)    # guess the challenge first
            w = random_unit(x)             # pick the root we'll "reveal" first
            if bit == 0:
                u = (w * w) % x            # u a random residue, w a root of u
            else:
                u = (w * w * inverse(y, x)) % x   # u*y = w^2, so w a root of u*y; u a random residue
            if B_prime(view, u) == bit:    # did B''s real challenge match our guess?
                view.append((u, bit, w))   # keep this round; its distribution == the real one
                break                      # else discard, rewind B_prime, retry
    return view
```

The causal chain, end to end: a static NP proof leaks its witness because the verifier re-checks it, so
I add interaction and randomness to let the verifier be convinced by the prover *passing random
challenges* rather than by holding a certificate; "the verifier learns nothing extra" can only be a
*computational* statement, so I define it as the verifier's *view* being reproducible by an efficient
*simulator* given only `x ∈ L`, with indistinguishability borrowed from semantic-secure encryption and
graded perfect/statistical/computational; the definition must range over all cheating verifiers (it's
the prover that is zero-knowledge), give each its side information (so it composes), and put the
verifier's coins in the view (so it catches the `gcd` leak); the simulator wins by *rewinding* — guess
the secret challenge, build the transcript backwards, retry — which is impossible for a real prover and
is exactly why a real prover gives nothing away; and quadratic residuosity (perfect) and
nonresiduosity (statistical) are the witnesses that this definition is satisfiable for languages whose
membership no one knows how to decide efficiently.

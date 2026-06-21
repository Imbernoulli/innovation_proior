Let me start from the pain. I want to prove that some explicit, large complexity class has no
nonuniform ACC circuits of polynomial size. ACC means constant depth, polynomial size, unbounded
fan-in AND/OR/NOT, plus MOD_m gates for an arbitrary constant m — where a MOD_m gate fires iff m
divides the sum of its inputs. The bottom-up program got us AC0 (parity is hard, Furst–Saxe–Sipser
and Ajtai, then exponential via Yao and Håstad), then AC0[p] for prime p (Razborov on majority,
Smolensky on MOD_q via low-degree polynomial approximation over F_p). And then it died. Composite
modulus m has no field, so the polynomial method has nothing to stand on, and for over twenty years
ACC has been a wall. The embarrassment that keeps me up: I can't even rule out that EXP^NP is
computed by depth-three circuits made of nothing but MOD_6 gates. MOD_6 versus MOD_7 — how could a
composite modulus be so much more powerful than a prime one? It can't be, surely. But I can't prove
it.

So forget trying to find a *simple* function hard for ACC; that's the bottom-up route and it's
exactly where everyone has been stuck. Flip the quantifiers. Take a *complicated* function — push
all the way up to nondeterministic exponential time, NEXP — and try to rule out *weak* circuits for
*it*. Even NEXP ⊄ ACC is open. If I could get that, it'd be the first crack in the wall.

But there's a thicket of barriers in the way, and I have to keep them in view the whole time or I'll
waste months on a doomed approach. Relativization: Baker–Gill–Solovay built oracles, and Aaronson
tells me there are oracles A with NEXP^A ⊆ ACC^A. So any argument that still works when I bolt an
oracle onto every machine and circuit is dead on arrival — it can't possibly separate NEXP from ACC,
because relative to A they don't separate. That immediately kills pure diagonalization, which is the
one unconditional tool I really trust: diagonalization relativizes, it proves the *uniform* NEXP ≠
NP but says nothing nonuniform. Algebrization closes the obvious algebraic loophole too. And then
natural proofs (Razborov–Rudich): if my argument hands me a constructive, large combinatorial
property that's useful against ACC, and ACC can compute pseudorandom functions, the property breaks
the PRFs — self-defeating. The AC0[p] proofs are natural. So whatever I do cannot look like them.

Let me pick the target carefully, because the choice will determine how hard the rest is. I want a
problem in NEXP that, if it's not in ACC, drags the whole class with it — so an NEXP-complete
problem. The trouble is NEXP-completeness is underexplored; the list is short. But there's a clean
factory for NEXP problems: take an NP problem and make it *succinct*. Given a problem P, define
SUCCINCT P: the input is a small circuit C with n inputs and poly(n) size; let T(C) be its 2^n-bit
truth table; ask whether T(C), read as an instance of P, is a yes-instance. So I'm only solving the
*highly compressible* instances of P — the exponentially long ones with a polynomial-size circuit
description. Papadimitriou–Yannakakis showed that for the natural NP-complete problems, the succinct
version is NEXP-complete. SUCCINCT 3SAT is the obvious pick: 3SAT is studied to death, and the
succinct version is *very* NEXP-complete — there's a super-efficient reduction from any NEXP
language to it. Good. SUCCINCT 3SAT it is. Show it's not in ACC and I'm done.

Now, why should SUCCINCT 3SAT have a time lower bound at all? It does, and it's clean. By the
efficient Cook–Levin theorem for NEXP — I'll come back to *why* it's efficient, the input count is
the whole ballgame — any L ∈ NTIME[2^n] reduces in poly(n) time to a SUCCINCT 3SAT circuit C_x with
n + O(log n) inputs and poly(n) size, with x ∈ L iff the decompressed formula is satisfiable. So if
SUCCINCT 3SAT had a nondeterministic algorithm running in, say, 2^{n − ω(log n)} time on poly-size
circuits, then every L ∈ NTIME[2^n] could be decided nondeterministically in
2^{(n + O(log n)) − ω(log n)} · poly(n) = o(2^n) time. That's NTIME[2^n] ⊆ NTIME[o(2^n)], which the
nondeterministic time hierarchy theorem (Seiferas–Fischer–Meyer; Žák) forbids. So: SUCCINCT 3SAT
cannot be solved in 2^{n − ω(log n)} time, even nondeterministically, on n-input poly-size circuits.
A concrete, strong time lower bound. The time hierarchy is going to be my contradiction engine.

Hold on, that input count. If C_x had 3n + O(log n) inputs instead of n + O(log n), the same
arithmetic would only give a lower bound against 2^{n/3 − ω(log n)} algorithms, and "beat 2^{n/3}"
is a vastly harder target than "beat 2^n" — no one has that even for 3SAT. So I really need the
n + O(log n) version. Where does it come from? The naive Cook–Levin tableau for a time-t computation
is a t × t grid of cells, each cell determined by three neighbors, so it's a t²-ish 3-CNF. With
t = 2^n that's a circuit with ~2n or 3n inputs to index the grid — too many. But the sharpened
reductions (Cook 1988; Robson 1991; Fortnow–Lipton–van Melkebeek–Viglas) reduce a time-t random
access computation to an O(t log⁴ t)-size formula whose i-th bit is computable in poly(log t) time
from i. Quasi-linear size means the index i needs only log t + O(log log t) bits; with t ≈ 2^n
that's n + O(log n) inputs. That single constant in the exponent is the difference between a target
I can hit and one I can't. So I lean entirely on the efficient Cook–Levin.

Now the real question. I have a *time* lower bound for SUCCINCT 3SAT. I want a *circuit* lower bound.
Why on earth would those be related? Here's the intuition I keep coming back to, and it's
hand-wavy but it won't let go of me: a polynomial-size circuit for a function means the function's
truth table is *highly compressible and regular*. A circuit that's completely representative of a
function's behavior on exponentially many inputs, but is itself tiny, feels much closer to a
*polynomial-time algorithm* than to an exponential-time one. So if SUCCINCT 3SAT had small circuits,
maybe these short representatives of exponential computation could be discovered, and exploited,
*algorithmically* — fast. If having small circuits secretly means you have so much structure that a
fast algorithm can use it, then small circuits for SUCCINCT 3SAT would buy a fast algorithm for
SUCCINCT 3SAT, contradicting the time lower bound I just established. That's the shape of the thing
I want: a theorem that *spins circuits into algorithms*. If SUCCINCT 3SAT ∈ ACC implies a
faster-than-2^n nondeterministic algorithm for SUCCINCT 3SAT, then by the time lower bound,
SUCCINCT 3SAT ∉ ACC, i.e. NEXP ⊄ ACC. Let me try to make that real.

First attempt, the obvious one. I'm allowed nondeterminism. So guess a poly-size ACC circuit C that
supposedly decides SUCCINCT 3SAT, then run it on my input. But — how do I check C is *correct*?
Naively I'd have to verify that for all 2^n inputs y, C(y) = 1 iff T(y) is a satisfiable exponentially
long formula. That's a 2^n verification, and worse, checking each y is itself the SUCCINCT 3SAT
problem. Circular and too slow. Dead end on the nose.

Let me steal from program checking and the interactive-proof world. Blum–Kannan program checkers,
or better, NEXP = MIP: Babai–Fortnow–Lund showed every NEXP problem has a probabilistic
polynomial-time verifier with access to an oracle trying to prove membership. So I could guess an
"oracle circuit" C, run the PPT checker A treating C as the oracle, and accept iff A^C accepts.
Write it down:

  SATALG(x): guess a poly-size circuit C; run the PPT checker A for SUCCINCT 3SAT on x with C as
  oracle; accept iff A^C(x) accepts.

It doesn't work, and the reason is structural. SATALG is a nondeterministic guess followed by a
*randomized* computation that can err on rejection. So x could be a no-instance and still have an
accepting path — that's not a sound nondeterministic algorithm. And I can't just derandomize: it's
open whether BPP = NEXP, and the MIP characterization *fails* if you replace the PPT verifier with
a nondeterministic one — a nondeterministic poly-time machine with an oracle is only as strong as NP.
So randomness is load-bearing in NEXP = MIP, and I can't afford it. Wall.

But staring at the failure, something nags. Program checking and MIP are inherently *black-box*: they
only probe input/output behavior of the oracle. My box, though, is special — by assumption it's a
small ACC circuit. In a nondeterministic algorithm I can guess that circuit and *cut it open*. I can
look at its gates. Surely seeing the internals is worth something a black box can't give me. Let me
chase that: what's the gap between a black box and a small circuit, and can I exploit it?

The hardest simple black-box problem: given a box on n inputs, does it output 1 somewhere? An
adversary answers 0 to every query until forced, so this needs 2^n queries — genuinely hard for
black boxes. Replace the box with a circuit and the same question is *Circuit-SAT*. So the gap I'm
hunting is exactly: can analyzing a circuit's *structure* beat the 2^n you'd need for a black box?
Let me sanity-check the extreme: assume the strongest possible separation, that circuits are totally
easy — P = NP, Circuit-SAT in P. Can I then get a circuit lower bound? Yes — this is essentially the
old Karp–Lipton–Meyer observation, and re-deriving it tells me the *mechanism*.

Suppose P = NP and, for contradiction, EXP ⊆ P/poly. Take any 2^n-time machine A, say one-tape.
On input x, nondeterministically guess a poly-size circuit C whose truth table T(C) is the
(exponentially long) computation history of A(x); such a C exists because EXP ⊆ P/poly. To verify C,
universally — using co-nondeterminism — quantify over all steps i and all tape cells j, and check
that C's claim about cell j at step i is consistent with its claims about cells j−1, j, j+1 at step
i−1 (one-tape locality). That's evaluating C at four index pairs, poly time. If every local check
passes, accept iff C says A(x) accepts. This is a Σ₂ computation: guess, then universally verify.
But P = NP collapses Σ₂ to P. So I've simulated an arbitrary 2^n-time machine in P — contradicting
the time hierarchy. Hence EXP ⊄ P/poly. The mechanism is exactly the one I wanted: **guess a circuit
encoding an exponential object, and verify it by purely LOCAL consistency checks.** Locality is what
makes the verification cheap.

The catch is the assumption. P = NP is absurdly strong. Can I weaken it? Following the argument, a
half-exponential-time Circuit-SAT algorithm (f with f(f(n^k)^k) ≤ 2^{n/2}) suffices instead. But
even that is a fantasy — the best Circuit-SAT algorithms save only a subexponential factor over 2^n,
nowhere near half-exponential, and ETH says 3SAT needs 2^{Ω(n)}. So as a route to an *unconditional*
bound, Karp–Lipton–Meyer is vacuous: the algorithmic hypothesis is one nobody can supply. Wall —
but an instructive one. I have the mechanism (guess-and-locally-verify against the time hierarchy);
I need to *cut the algorithmic assumption down to something achievable*.

Let me regroup and list what I've actually learned, because I think the pieces are closer than they
look. (1) If I assume NEXP has small circuits, then *lots* of expensive computations get small
circuits, and with nondeterminism I can guess as many helper circuits as I want, each encoding more
information that might help me verify the others. That's a resource I haven't spent. (2) I've used
*no* special property of ACC yet — everything so far is about P/poly. That's fine; I'll bring ACC in
later, but it tells me the framework is generic.

Lean on resource (1). Impagliazzo–Kabanets–Wigderson — "in search of an easy witness" — proved
something I can use precisely. If NEXP ⊆ P/poly, then not only does SUCCINCT 3SAT have small
circuits, but for every circuit x succinctly encoding a *satisfiable* 3-CNF, there is another small
circuit W_x whose truth table is a *satisfying assignment* to that formula. Compressible satisfiable
formulas have compressible satisfying assignments. And if NEXP ⊆ ACC, then since P ⊆ ACC too, the
Circuit-Value problem is in ACC, so I can convert W_x into an *ACC* witness circuit by plugging its
encoding into an ACC circuit for Circuit-Value. So under NEXP ⊆ ACC, every satisfiable succinct
instance has an ACC circuit encoding a satisfying assignment.

This reframes everything. Don't guess a circuit that *decides* SUCCINCT 3SAT — I can't check that.
Guess a circuit that *encodes a satisfying assignment* — a *witness*. Witnesses are locally
checkable: a satisfying assignment is correct iff every clause is satisfied. That's the locality I
exploited in Karp–Lipton–Meyer, now in the right place. New algorithm:

  SATALG2(x): guess a poly-size circuit W_x; accept iff T(W_x) is a satisfying assignment to the
  formula F_x = T(x).

By IKW, under NEXP ⊆ P/poly, this is correct. But verifying "T(W_x) satisfies F_x" naively is again
2^n — evaluate W_x and x on everything. So I have to make the verification cheap. Checking a 3-CNF
is local: for each clause, at least one of its three literals is satisfied, and that needs only
logarithmic workspace. So SATALG2 runs in polynomial *space* — try all poly-size W_x, and for each,
run a logspace satisfaction check that, when it needs a bit of the formula, evaluates x at the right
index, and when it needs a bit of the assignment, evaluates W_x at the right index. Polynomial space,
but still 2^n time. Can I cut the *time*?

To check that the assignment satisfies the formula, I iterate over clauses by
feeding clause-indices i into x. For clause i, x prints its three variable-indices and sign bits.
Feed each variable-index into a copy of W_x to get that variable's assigned value. Compare against
the sign bits to see if clause i is satisfied. Build a circuit D(i) that takes the index i, runs one
copy of x (to print clause i's three literals) and three copies of W_x (to read off the three
assigned values), and outputs 0 iff clause i is satisfied. Then: the assignment encoded by W_x
satisfies the formula iff D(i) = 0 for *every* i iff D is *un*satisfiable. So checking the witness is
exactly one Circuit-SAT call on D — not 2^n separate checks.

  SATALG3(x): guess poly-size W_x; build D out of x and W_x; accept iff D is unsatisfiable.

And now if Circuit-SAT on poly-size circuits can be solved in 2^{n − ω(log n)} time, SATALG3 runs in
2^{n − ω(log n)} time, contradicting the SUCCINCT 3SAT time lower bound. So I've proved: if Circuit-
SAT is solvable in 2^{n − ω(log n)} time, then NEXP ⊄ P/poly. The algorithmic assumption has dropped
from "P = NP / half-exponential" to "shave a 1/n^{ω(1)} factor off brute force" — a *minor*
improvement over exhaustive search. That feels achievable. Plenty of SAT algorithms beat
2^{n − ω(log n)} for CNF and even for AC0; the only trouble is they don't obviously generalize to
unrestricted circuits.

So now bring in ACC, finally. The plan: replace "Circuit-SAT" by "ACC-SAT" and replace P/poly
assumptions by ACC assumptions, and hope SATALG3 still goes through. Where does it break? Under
NEXP ⊆ ACC, by IKW + Circuit-Value-in-ACC I can guess an *ACC* witness W'_x. Good — the witness is
ACC. But the input circuit x is unrestricted; SUCCINCT 3SAT lets x be any circuit at all. So the
circuit D, which contains a copy of x, is *not* ACC. An ACC-SAT algorithm can't run on D. That's the
hole.

  SATALG4(x): guess ACC witness W'_x; guess an ACC circuit x' equivalent to x; verify x ≡ x' (???);
  build the ACC circuit D out of x' and W'_x; accept iff D is unsatisfiable.

The (???) is the whole problem. Why even hope x' exists? Because I've assumed NEXP ⊆ ACC, hence
P ⊆ ACC, and x computes a polynomial-time function (it's a poly-size circuit, evaluating it is in P),
so there *exists* an equivalent ACC circuit x' — but it's nonuniform, possibly impossible to
construct. Fine, I'll guess it. The question is how to *verify* my guess x' actually equals x. The
textbook equivalence check is to test satisfiability of E(i) = (x(i) ∨ x'(i)) ∧ (¬x(i) ∨ ¬x'(i)) —
satisfiable iff they disagree somewhere. But E *contains a copy of x*, so E is unrestricted, and I'm
right back where I started. Guess an ACC E' equivalent to E? Then I need to verify *that*, and E, E'
are only bigger than x, x'. Infinite regress. This looks genuinely impossible. I'm stuck.

Let me sit with the obstruction. The output of x is one bit. Comparing one bit of x against one bit
of x' is what forces me to recompute x, which is the unrestricted thing I'm trying to get rid of.
But when I *evaluate* x on an input i, I don't just produce one bit — I produce a bit on *every wire*
of x. The gate values. That's a huge amount of structured, redundant information, and — here's the
point — it's all *locally* checkable: each gate's output must be the gate-function applied to its
inputs' outputs. Locality again. And under P ⊆ ACC, that whole gate-value table is itself computable
in ACC, because computing the value of gate j of circuit x on input i is a polynomial-time function
of (x, i, j).

This is the error-correcting-code instinct, or arithmetization: don't just carry the message, carry
redundant information *about* the message so that local inconsistencies expose any corruption. I was
trying to verify one output bit; instead I should force my guessed circuit to commit to *all the
intermediate values* and then catch it if any local gate-relation is violated. Guessing *more*
information makes verification *easier*, not harder.

Let me build it. Represent x by its set of gate tuples: for each gate j, a tuple ⟨j, j₁, j₂, g⟩ where
g ∈ {AND, OR, NOT, INPUT} is the gate type and j₁, j₂ are the indices of the gates feeding j (with
the convention that the first n+O(log n) gates are the INPUT gates, j₁ = j₂ = 0 there, and every
gate has fan-in ≤ 2). The function f(x, j) that prints ⟨j₁, j₂, g⟩ from j is polynomial-time
computable, because x ↦ (its description) is, so under P ⊆ ACC there's an ACC circuit D(x, j) for it
— the gate *connection* circuit. Its correctness I can check *directly*, with no SAT call: just
evaluate D(x, ·) on all j = 1, …, |x| (only poly-many gates) and confirm it reproduces the actual
circuit x. Cheap, well under 2^n. That's my regress-stopping base case.

Next, the gate *values*. Let E(x, i, j) be an ACC circuit that, given input i to x and a gate index
j, outputs the value carried on gate j when x is evaluated on i. Again polynomial-time in (x, i, j),
so ACC under P ⊆ ACC; I guess it. Now verify E using the already-verified D. Build a circuit
VALUE(i, j): feed j to D(x, ·) to get the connections j₁, j₂ and type g; compute v₁ = E(x, i, j₁),
v₂ = E(x, i, j₂), and v = E(x, i, j); then VALUE outputs 0 iff E's claims are locally consistent at
gate j on input i — i.e.
  if g = INPUT: 0 iff j is among the first n+O(log n) indices and the j-th bit of i equals v;
  if g = NOT:   0 iff v = ¬v₁;
  if g = AND:   0 iff v = v₁ ∧ v₂;
  if g = OR:    0 iff v = v₁ ∨ v₂.
Each of these is a constant number of gates on top of D and E, so VALUE is ACC. Because D is already
known correct, VALUE(i, j) = 1 happens exactly when E lies about gate j on input i. Therefore VALUE
is *un*satisfiable iff E prints the correct value of every gate of x on every input — and that's one
ACC-SAT call. If it's satisfiable, reject the guess. The number of inputs to VALUE is
n + O(log n), so the ACC-SAT call runs in 2^{n}/n^{k} time by the (soon-to-be-built) ACC-SAT
algorithm; that's the dominant cost and it's below 2^n.

Once E is verified, x' falls out for free: x(i) = E(x, i, j*) where j* is the index of the output
gate of x. So set x' = E(·, j*), guaranteed equivalent to x and ACC. (Equivalently, just use E
itself.) The regress is broken: D is checked directly (poly time), E is checked using D plus one
ACC-SAT call, and x' is read off E. Three stages, each grounded in the previous, base case
self-evident. Write the assembled algorithm:

  SATALG5(x):
    guess ACC witness W'_x;
    guess ACC gate-value circuit E (and gate-connection circuit D);
    verify D directly on all gates of x;                          # poly time
    build VALUE from D, E; if VALUE is satisfiable, reject;       # one ACC-SAT call: E correct
    set x' = E(·, j*);                                            # ACC, equivalent to x
    build the ACC circuit D' out of x' and three copies of W'_x;  # D'(i)=0 iff clause i satisfied
    accept iff D' is unsatisfiable.                               # one ACC-SAT call: W'_x satisfies

Every circuit fed to ACC-SAT is now genuinely ACC, with n + O(log n) inputs and quasipolynomial size.
And — I want to be precise about what properties of ACC I actually used here, because it matters for
how far this generalizes. I used: (a) ACC contains AC0 (so the constant-size local checks and the
ORs/ANDs of copies stay in the class), and (b) ACC is closed under *composition* — feeding the
outputs of ACC circuits into the inputs of another ACC circuit (with polynomial blowup) stays ACC,
which is what makes VALUE and D' ACC. That's *it*. Nothing else about ACC. So this "spinning" half
of the argument works verbatim for *any* circuit class C that contains AC0 and is closed under
composition — TC0, NC1, formulas, P/poly. The class-specific work is entirely quarantined in the
SAT algorithm. Let me state the connection cleanly: if ACC-SAT on circuits with n + O(log n) inputs
and quasipolynomial size can be solved in O(2^n / n^k) time for every k, then NEXP ⊄ ACC. Now I owe
the SAT algorithm.

ACC circuit satisfiability, beating 2^n. I have the structural gift: every ACC circuit has an
equivalent SYM⁺ representation — a symmetric function of ANDs of variables — of quasipolynomial size
(Yao; Beigel–Tarui; Allender–Gore; Green et al.). Precisely: a depth-d size-s ACC circuit with
MOD_m gates becomes an equivalent SYM⁺ circuit of size s^{O(log^{f(d,m)} s)} with ANDs of
O(log^{f(d,m)} s) fan-in, where f(d, m) ≤ m^{O(d)}, computable in about that much time. For
subexponential s this blowup is tolerable.

Why does this particular normal form help, rather than some other low-depth shape? A *symmetric*
function depends only on the *count* of its true inputs — how many of the bottom ANDs evaluate to 1.
So satisfiability of a symmetric g(AND₁, …, AND_K) on a given assignment reduces to: compute how many
ANDs are 1, then apply g to that count. And "for every assignment, count how many of these monomials
are on" is exactly the kind of thing a counting/transform primitive can do in bulk. That is the
handle the symmetric top gate gives me, and it's why SYM⁺ — and not an arbitrary depth-two circuit —
is what I want. Let me make sure I can actually *build* the SYM⁺ form, because the constants live in
the construction (this is the Yao/Beigel–Tarui/Allender–Gore machinery; I'll re-walk it so I trust
it). First make the circuit a tree (fan-out 1) at the cost of raising s to s^{O(d)}. Then: replace
each OR and AND by low-degree probabilistic constructions over MOD_p of polylog-fan-in ANDs, all
sharing one set of poly(log s) random bits (the Valiant–Vazirani randomization), and replace NOT by
a MOD gate. Eliminate composite moduli by the Chinese Remainder Theorem: m | x iff p_i^{e_i} | x
for every prime power in the factorization, so MOD_m = AND of MOD_{p_i^{e_i}}; and a MOD_{p^e} gate
becomes a constant-fan-in AND of MOD_p of ANDs, using that p^e | x iff p | C(x, p^i) for all
i = 0, …, e−1. Derandomize the probabilistic inputs by enumerating all poly(log s) settings and
taking a MAJORITY of the copies — that's the symmetric top gate appearing — at a quasipolynomial
size cost. Then push the remaining layers of MOD_p gates up into the top symmetric gate, one layer
at a time, using the modulus-amplifying polynomials of Beigel–Tarui: the polynomial
  P_k(x) = (−1)^k (x−1)^k ( Σ_{i=0}^{k−1} C(k+i−1, i) x^i ) + 1
satisfies P_k(x) ≡ 0 (mod p^k) when x ≡ 0 (mod p) and P_k(x) ≡ 1 (mod p^k) when x ≡ 1 (mod p);
setting Q_k(x) = 1 − P_k(x^{p−1}) and invoking Fermat's little theorem, Q_k(Σ y_i) ≡ 1 (mod p^k) iff
p divides Σ y_i, i.e. Q_k expresses a MOD_p gate as a polynomial mod p^k. Each Q_k of a sum is a
symmetric multivariate polynomial of degree ≤ k(p−1), expand it into ≤ (f f')^{O(k(p−1))} monomials,
each an AND of polylog variables times a small coefficient (simulate the coefficient by duplicating
monomials), so a layer of MOD_p over ANDs becomes a *single sum mod p^k of ANDs* — a new symmetric
function consuming that layer. Iterating a constant number of times (constant depth), the whole
circuit collapses to a symmetric function of quasipolynomially many ANDs of variables, where the
final symmetric function has the form
  F(v) = MAJORITY( ( ( v mod p_1^{k_1} ) mod p_2^{k_2} ) … mod p_{d'}^{k_{d'}} ),
which is still symmetric (a symmetric function composed with sum-mod-p^k is symmetric), and
evaluable in time comparable to the circuit size. One more bookkeeping point: I want the bottom ANDs
to have *no negated* variables, because the evaluation method below counts monomials and prefers
plain products. A SYM of ANDs-with-negations can be turned into a SYM of ANDs-without-negations by
expanding each (1 − x_i) factor and tracking the signs of the coefficients: split the monomials into
those with positive and negative coefficients, scale the positive ones by 2^ℓ for ℓ with 2^ℓ > K',
and define the new symmetric function to read off A − B from the binary form A·2^ℓ + B of the count.
Fine — that's a clean SYM⁺ with plain-variable ANDs.

Now the algorithmic core: given a SYM⁺ circuit, evaluate it on *all* 2^n assignments fast. Brute
force is 2^n · poly(s). I want roughly 2^n + poly(s) — amortized poly(n) per assignment. The
satisfiability question then is just: does any of the 2^n outputs equal 1?

Let me find the evaluation method by thinking about what's really being computed. The SYM⁺ output on
an assignment depends only on the number of ANDs that fire. So I want, for every assignment, the
count of satisfied ANDs. Let the ANDs be indexed by their variable-subsets: AND_j is over the subset
G_j ⊆ [n]. Define f(S) = #{ j : G_j = S }, a table of 2^n entries built in O(2^n + s · poly(n)) time
by bucketing the s ANDs by their subset. Now the count of ANDs satisfied by the assignment "set
exactly the variables in T to 1" is g(T) = Σ_{S ⊆ T} f(S) — because AND_j fires under T iff all its
variables are set, i.e. G_j ⊆ T. That g is precisely the *zeta transform* of f over the subset
lattice. And the zeta transform is computable by Yates's 1937 dynamic program in O(2^n · poly(n)):
set g_0 = f and, for i = 1, …, n,
  g_i(T) = g_{i−1}(T) + g_{i−1}(T \ {i})  if i ∈ T,   else  g_i(T) = g_{i−1}(T);
each pass is O(2^n · poly(n)), n passes, and an induction shows g_n(T) = Σ_{S ⊆ T} f(S) = g(T). So
in O(2^n · poly(n)) time I have, for every assignment T, the count of satisfied ANDs; apply F to
each count (poly(s) to tabulate F, then a lookup per assignment) and I've evaluated the SYM⁺ circuit
everywhere. Satisfiable iff some count makes F output 1. Clean, and it's the proof I'd actually
present — minimal machinery.

I should record that there are two other ways to do this same bulk evaluation, because they reveal
why the method is robust. One uses fast matrix multiplication: split the n inputs into halves A and
B of n' = (n+1)/2 each; form a 2^{n'} × s matrix M_A with M_A(i, j) = 1 iff assignment i to A doesn't
force AND_j to 0, and an s × 2^{n'} matrix M_B likewise for B; then (M_A M_B)(i, k) = number of ANDs
set to 1 by the combined assignment (i, k). Setting N = 2^{n'}, the middle dimension s ≤ 2^{0.1 n'} =
N^{0.1} is polynomially smaller than the outer N — exactly the regime where Coppersmith's rectangular
matrix multiplication multiplies an N × N^{.1} by an N^{.1} × N matrix in O(N² log² N) operations —
near-optimal — so the product costs N² · poly(n) = 2^{n} · poly(n), and scanning its entries against
the symmetric function decides satisfiability. The third way: a multilinear polynomial in coefficient form can be
converted to its values on all of {0,1}^n in O(2^n · poly(n)) by the recursive coefficient-to-point
algorithm (a multidimensional FFT) — split p = x_1 q_1 + q_2, recurse on q_1, q_2, merge with
R(2^n) = 2 R(2^{n−1}) + O(2^n poly n) = O(2^n poly n). Three routes, same destination. The DP one is
cleanest, so that's the one I lead with; the matrix-multiplication one is the most likely to
generalize.

But all three run in *2^n* time, and I promised *below* 2^n. The coefficient-to-point view tells me
exactly what I need: to beat 2^n I must evaluate a polynomial in *fewer than n* variables. My circuit
has n inputs, so I seem to be stuck — I can't just throw away inputs. Unless I trade inputs for size.
Circuit satisfiability is "does the OR over all 2^n assignments fire?" Split off a subset of k of the
n inputs: for each of the 2^k settings of that k-subset, substitute it into C to get a circuit on the
remaining n − k inputs, and take the OR of all 2^k copies. Call it the *k-blowup* C'. Then C' has
only n − k free inputs; size O(2^k · s); it's still ACC (one extra depth level, since ACC has
unbounded-fan-in OR); and C is satisfiable iff C' is. I've brute-forced k variables to *buy fewer
variables* in exchange for more size — and I'm happy to pay size, because the SYM⁺ decomposition
tolerates quasipolynomial blowup and the evaluation cost is governed by the *number of variables*.

Now assemble. Given a depth-d ACC circuit C of size s = 2^{n^ε} on n inputs:

  ACCSAT(C):
    k ← n^{1/(2 f(d,m))}
    C' ← k-blowup of C                       # n − k inputs, size ≤ 2^k · s, still ACC
    (g, h) ← SYM⁺ decomposition of C'        # symmetric g of K monomials h, K = 2^{O(k f + log^f s)}
    evaluate h on all 2^{n−k} points         # zeta-transform DP, O(2^{n−k} poly(n) + K) time
    output satisfiable iff g(h(·)) = 1 somewhere

Let me check the exponents land below 2^n. Writing e for f(d, m), the decomposition of the k-blowup
gives a symmetric gate over K ≤ s^{e(e · log^e s)} ANDs. With s = 2^{n^ε} this is
K ≤ 2^{e n^ε (e n^{εe})}, and choosing k = n^{1/(2e)} forces K ≤ 2^{n^{2/3}} for all small enough ε
and large n (the extra 2^k size factor from the blowup only raises log s' from n^ε to n^ε + n^{1/(2e)},
which keeps the exponent below n^{2/3} for small ε). The evaluation runs in
O(2^{n−k} · poly(n) + K) = O(2^{n − n^{1/(2e)}} · poly(n)), since 2^{n^{2/3}} ≪ 2^{n − n^{1/(2e)}}.
So ACC-SAT on subexponential-size depth-d ACC circuits runs in 2^{n − Ω(n^δ)} time for a δ > 0
depending only on d and m. That is comfortably faster than 2^n — in fact a savings of a 2^{Ω(n^δ)}
factor, far more than the mere 1/n^{ω(1)} the connection needs.

Both halves now exist, so combine. Suppose, for contradiction, NEXP ⊆ ACC. Then P ⊆ ACC, so I have
the gate-connection and gate-value circuits; and by IKW (easy witnesses) plus Circuit-Value-in-ACC,
every satisfiable succinct instance has an ACC witness. Take any L ∈ NTIME[2^n] and input x. By the
efficient Cook–Levin, build C_x with n + O(log n) inputs. Run SATALG5: guess and verify (via the
three-stage D, E, VALUE construction, each an ACC-SAT call below 2^n) an ACC circuit x' equivalent
to C_x; guess the ACC witness W'_x; build the ACC circuit D' that is unsatisfiable iff W'_x encodes a
satisfying assignment; decide D' with the ACC-SAT algorithm in O(2^n / n^k) time. The total is
O(2^n / n^k), so every L ∈ NTIME[2^n] is in NTIME[o(2^n)] — actually NTIME[o(2^n / n^k)] — which the
nondeterministic time hierarchy theorem forbids. Contradiction. Therefore NEXP ⊄ ACC: SUCCINCT 3SAT,
and hence all of NEXP, has no nonuniform polynomial-size ACC circuits.

A few extensions fall out, and they're worth pinning down because they tell me how sharp the result
is. For the larger class E^NP I don't even need the IKW easy-witness machinery — I can get the
witness directly. Consider the E^NP machine that, on (x, i), computes C_x, then binary-searches for
the lexicographically smallest satisfying assignment to F_{C_x} using its NP oracle (each query
"is there a satisfying A' ≤ A?" is in NP), and outputs the i-th bit. That's an E^NP function; if
E^NP ⊆ ACC of size S(n), this function has an S(3n)-size ACC witness circuit. Plug that in and the
same contradiction needs only an O(2^n / n^c)-time ACC-SAT algorithm versus circuits of size about
n·S(2n) + S(3n); since my ACC-SAT runs in 2^{n − Ω(n^δ)} on subexponential circuits, I get the full
exponential size–depth tradeoff: for every d and m there is a δ > 0 and a language in E^NP with no
depth-d, MOD_m, 2^{n^δ}-size ACC circuits. For NEXP the price is the IKW route, which only yields
*polynomial-size unrestricted* witnesses; converting them to ACC via the folklore lemma (P ⊆ ACC
makes Circuit-Eval ACC, so every circuit family has an ACC equivalent of nested-S size) costs a
couple of applications of S, so the NEXP size lower bound is only "half-exponential"-flavored, not
fully exponential — but it still kills polynomial and even quasipolynomial size, and with one more
nesting it extends down to sub-third-exponential size and to grotesque classes just above NTIME[2^n].

And the barriers — let me confirm I actually threaded them, because that's the whole reason the
bottom-up program couldn't do this. Natural proofs: the argument never extracts a combinatorial
property of the hard function; it leans on the *completeness* of SUCCINCT 3SAT and on
diagonalization through the time hierarchy, so there's no P-natural property to harvest (and anyway
there's little evidence ACC even has pseudorandom functions). Relativization and algebrization: the
one place I use a non-black-box fact is the ACC-SAT algorithm, and it uses the SYM⁺ *structure* of
ACC circuits — every known faster-than-2^n SAT algorithm breaks the moment you add an oracle or an
algebraic extension to the instance, precisely because beating exhaustive search forces you to
exploit structure a black box hides. There are oracles with NEXP^A ⊆ ACC^A, and my proof correctly
does *not* relativize, because it reaches inside the circuit. The mechanism is the inversion: a
faster satisfiability algorithm for a circuit class — an algorithmic *upper* bound — is exactly what
spins, through the time hierarchy, into a circuit *lower* bound for that class. Improved
exponential-time SAT algorithms are now the only thing standing between here and lower bounds for
TC0, for unrestricted P/poly, for NP — because the spinning half already works for every robust
class; only the SAT algorithm needs ACC.

To recap the causal chain in one breath: I want NEXP ⊄ ACC; target the NEXP-complete SUCCINCT 3SAT,
which has a strong time lower bound from the efficient Cook–Levin reduction (n + O(log n) inputs) and
the nondeterministic time hierarchy; the Karp–Lipton–Meyer mechanism (guess a circuit for an
exponential object, verify by *local* checks) shows fast circuit-analysis spins into a lower bound,
but only under an unachievable assumption; weaken the assumption by guessing a *witness* circuit
(IKW easy witnesses) and reducing its verification to *one* SAT call via clause-locality (SATALG3);
fix the "x is unrestricted" obstruction by guessing the *gate-value* circuit of x and checking it
with local gate-consistency plus a SAT call, breaking the regress with the directly-checkable
gate-connection base case (SATALG5), which needs only AC0 ⊆ C and closure under composition; then
deliver the missing piece, a faster-than-2^n ACC-SAT algorithm, by converting ACC to a symmetric
function of ANDs (SYM⁺), k-blowing-up to reduce variables, and bulk-evaluating via the zeta-transform
dynamic program (or Coppersmith rectangular matrix multiplication, or coefficient-to-point FFT) in
2^{n − Ω(n^δ)} time; the two halves collide at the time hierarchy and out drops NEXP ⊄ ACC.

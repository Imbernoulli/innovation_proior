# Context

## Research question

I want to manufacture randomness cheaply. Two concrete shortages drive this.

First, every randomized procedure — a Monte-Carlo simulation, a primality test, and above all a
one-time pad — consumes truly random bits at a rate I cannot afford. The one-time pad is perfectly
secure but needs a fresh random bit for every message bit. If I could take a short truly random
*seed* and deterministically expand it into a long string that is "as good as random" for the
purpose at hand, I would have traded an impossible amount of randomness for a small amount.

Second, and more ambitiously: a *random function* is even more expensive than a random string. A
function from k-bit strings to k-bit strings, chosen uniformly, is the ideal object in countless
protocols (authentication, identification, hashing, a "random oracle"). But specifying one requires
writing down its value at each of the 2^k inputs — about k·2^k bits — an astronomically large key.
Can I produce a function that is *easy to choose* (a few random bits), *easy to evaluate* (poly time
per input), and yet *indistinguishable from a truly random function* to anyone with bounded
computing power? That would make the convenient fiction of "a random function" into a usable tool.

The precise problem: define what it means for a deterministic, efficiently-computable object (a
string-generator, or a function) to be "random enough," and then *construct* such objects from some
plausible hardness assumption — and do it constructively, so the generator is an actual algorithm.

## Background

**Why the classical measures of randomness do not generate randomness.** Kolmogorov, Chaitin,
Solomonoff measure the randomness of an individual string by the length of its shortest program: a
k-bit string is random if no program shorter than k bits outputs it. This is an inherent property of
the individual string and the right notion of information content. It is non-constructive; the set
of Kolmogorov-random strings is non-recursive (uncomputable). Generalizations that require the
description to be both short *and* efficient (Sipser, Hartmanis, Levin's early notes) inherit the
same character.

**Statistical-test randomness.** The pragmatic tradition (Knuth) tests a pseudorandom sequence
against a battery of statistical tests — roughly the right count of 0s and 1s, no short repeats,
the right run-lengths. The linear congruential generator x_{i+1} = a·x_i + b mod n passes these
and produces well-mixed numbers cheaply. Blum, Blum and Shub observe that a sequence can have a
hard problem embedded in it and still have biased high-order bits.

**Unpredictability as a candidate.** Shamir built a generator from RSA whose next *number* is as
hard to predict as inverting RSA. That is a real step — hardness is now *embedded*, not hoped for —
but it produces numbers, not bits.

**The hardness primitives on the table.** Diffie–Hellman's one-way function idea is now formalized:
a one-way function f is efficiently computable but, for every efficient algorithm A and random x,
Pr[f(A(f(x))) = f(x)] is negligible — easy forward, hard to invert (we ask A only for *some*
preimage, since f may not be injective). A one-way *permutation* is additionally a bijection on
{0,1}^n; candidates include RSA (x ↦ x^e mod N) and modular exponentiation x ↦ g^x mod p, whose
inverse is the discrete logarithm. A subtle, load-bearing fact about one-way permutations: the
inverse being *hard to compute in full* does NOT make individual bits of the input unpredictable —
f(x) determines x information-theoretically, and known one-way candidates satisfy algebraic
identities (from RSA's x, y and inverses one reads off the inverse at x·y), so most bits of x leak.
So inverting f is hard as a whole, yet that hardness does not obviously localize to any particular
output of an efficiently-computable predicate on x.

**The proof tool that keeps recurring.** When two distributions are connected by a chain of
intermediate "hybrid" distributions, each adjacent pair differing by one application of a
primitive, an overall distinguisher of advantage δ across m hybrids must distinguish some adjacent
pair with advantage δ/m (a random-index argument; the per-pair advantages telescope to the
end-to-end advantage). With m polynomial and the per-step advantage forced negligible, δ is
negligible. This hybrid argument is the recurring engine of the field.

**The open problem in the air.** Blum and Micali constructed, from the discrete-log assumption, a
generator whose *next bit* is unpredictable. Brassard, and Blum–Goldwasser–Micali, asked whether the
exponentially-long pad such a generator implicitly defines could be *randomly accessed* — could one
compute the i-th bit, for i up to 2^k, in poly(k) time, and would such random access be
"randomness preserving"? That question is on the table, unresolved.

## Baselines

**Linear congruential generators (Knuth; analysed by Plumstead).** x_{i+1} = a·x_i + b mod n. Fast,
passes Knuth's statistical batteries, used everywhere for simulation.

**Shamir's RSA-number generator.** Emits a sequence of numbers x_i from a secret seed such that
predicting the next number is as hard as inverting RSA.

**Blum–Micali cryptographically-strong sequence generator (the immediate ancestor).** From a
one-way permutation f on a domain D (e.g. discrete-exp f(s)=g^s mod p), produce a bit sequence from a
secret seed. They define security by the *next-bit test*: for every efficient
predictor C and every i, Pr[C(b_1…b_i) = b_{i+1}] < 1/2 + negligible — no efficient observer of the
prefix guesses the next bit better than a coin. They prove their generator passes this test by a
reduction to the intractability of discrete log. Blum and Micali conjecture that passing the
next-bit test implies passing all efficient tests (true indistinguishability from uniform), and note
Yao's claim of equivalence. The construction gives a polynomial-length consecutive sequence from one
seed. The separate easy-access question, highlighted for pads based on squaring modulo a Blum
integer, asks whether the exponentially-long pad as a whole can be randomly accessed.

**Goldwasser–Micali probabilistic encryption / bit security.** Defines a single bit's encryption to
be "bit-secure" if no small circuit guesses the encrypted bit better than 1/2 + negligible, and
proves (via a separator/hybrid argument) that encryptions of long strings are unseparable iff the
single-bit scheme is bit-secure. This supplies the template of "indistinguishability of two
ensembles, reduced to a one-bit hardness by a hybrid."

## Evaluation settings

The yardsticks are definitional, not empirical. A candidate generator or function family is judged
against the class of all probabilistic polynomial-time *distinguishers* (equivalently, polynomial-
size circuit families in the non-uniform setting), under a complexity assumption such as the
intractability of the discrete logarithm modulo a prime, or, more generally, the existence of a
one-way function / one-way permutation. The relevant "instances" are the security parameter k → ∞,
with advantages measured as functions of k and the target being negligible (smaller than 1/Q(k) for
every polynomial Q, for all large k). For a string generator the natural tests are: the next-bit
test (predict bit i+1 from bits 1…i), and arbitrary poly-time statistical tests on the output
string. For a function family the test is given *oracle* access: a poly-time algorithm queries the
function at points of its choice and must decide whether it is talking to a member of the family
(keyed by a short random index) or to a uniformly random function. A still stronger benchmark is the
adaptive "chosen-exam" prediction game: query f adaptively, then name a fresh point x and try to
recognize f(x) among random alternatives. The computational models are the Turing machine and,
interchangeably, the Boolean circuit with poly(k) gates.

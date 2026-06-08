I keep coming back to the same shortage. Every randomized thing I want to run is starving for random
bits. The one-time pad is the sharpest case: it is perfectly secure and it is unusable, because it
demands a fresh truly random bit for every single bit of message. If two people could share a short
secret and *grow* it into a long pad that behaves like random, the pad would become practical. So the
question I actually care about is not "what is randomness" in the abstract — it is: can I take a short
truly random seed and deterministically stretch it into a long string that is good enough to *use* in
place of random?

And the more I stare at it, the more I want the same trick one level up. The object I would really
love is a random *function* from k-bit strings to k-bit strings. It shows up everywhere — to
authenticate messages, to identify a party, to hash, to play the role of a "random oracle." But a
truly random such function is monstrous to write down: I have to fix its value independently at each
of the 2^k inputs, about k·2^k bits of table. I cannot store that, cannot communicate it. So can I
get a function I can *choose* with a handful of bits, *evaluate* in polynomial time at any single
input, and that nonetheless *passes* for a truly random function to anyone who can only do a bounded
amount of computation? If I could, the comfortable lie "let f be a random function" would turn into
an actual algorithm.

So I need, first, a definition of "good enough" — and then a construction. Let me get the definition
right, because everything downstream depends on it, and I have a feeling the usual definitions are
exactly wrong for my purpose.

Start with the most respectable notion: Kolmogorov complexity. A string is random if its shortest
description is as long as itself; randomness is incompressibility. It is beautiful and it is the
correct notion of information content of an *individual* string. But watch what happens when I try to
*generate* with it. It is non-constructive — there is no algorithm even deciding which strings are
random; the set is not recursive. And worse, structurally: any algorithm that reads fewer than k truly
random bits and outputs a k-bit string has, by construction, just handed me a description of that
output shorter than k bits. So nothing efficient that takes a short seed can ever output a
Kolmogorov-random string. The definition I most respect forbids the operation I most need. Stretching
is impossible under it, period. The generalizations that ask the short description to also be
*efficient* don't escape this — they keep the same wall. Incompressibility is simply the wrong axis
if the goal is to manufacture usable randomness from little.

Drop down to the engineer's notion, then: a string is random if it passes statistical tests — right
proportion of 0s and 1s, no suspicious runs, the right block frequencies. The linear congruential
generator x_{i+1} = a·x_i + b mod n sails through Knuth's whole battery and is cheap. But it is a
disaster the moment someone *tests for the wrong thing*: Plumstead showed you can reconstruct the
entire sequence — all future outputs — from a few terms, even when a, b, n are all secret. The lesson
is brutal and it reframes the whole problem: passing *some fixed list* of tests means nothing,
because the adversary just runs a test that isn't on the list. The natural killer test here is
"predict the next output," and the LCG fails it completely. Even Blum, Blum and Shub's example makes
the same point from the other side — you can embed a genuinely hard problem in a sequence and the
sequence can still fail to look random, with biased predictable high-order bits.

So I am pulled toward two demands that seem to be in tension. From the LCG disaster: I must be safe
against *every* test, not a fixed list. From the Kolmogorov disaster: I cannot demand information-
theoretic randomness, or I can't stretch at all.

The resolution is to notice what "every test" should mean. I do not need to fool a test that runs
forever and can try all 2^k seeds — such a test trivially breaks any generator, since the output has
a short description (the seed). What I actually need is to fool every test that an *adversary can
actually run*: every *efficient* test. That is the whole move. "Random enough" is not an absolute,
information-theoretic property of the string — it is *relative to a model of computation with bounded
resources*. A short, highly-compressible string can still be random enough, as long as no efficient
procedure can exploit its compressibility. The randomness of an event is relative to who is looking
and how much they can compute.

Let me make that precise, because it has to carry real proofs. I'll call two ways of producing strings
*computationally indistinguishable* if no efficient (probabilistic polynomial-time) algorithm A,
handed a sample, can tell which source it came from: for every such A,

    | Pr[A(real) = 1] − Pr[A(ideal) = 1] | ≤ negligible(k),

where "negligible" means smaller than 1/Q(k) for every polynomial Q and all large k. A generator
G: {0,1}^k → {0,1}^{P(k)} with P(k) > k is *pseudorandom* if its output on a random seed is
computationally indistinguishable from a uniformly random P(k)-bit string. This is exactly strong
enough to do what I want: if G's output fools every efficient test, then I can drop it in place of
true randomness inside *any* efficient computation and no efficient observer of the outcome will
notice — which is precisely the substitution the one-time pad and the Monte-Carlo simulation need.
The definition is the engineer's "passes all tests," but with "all tests" honestly cashed out as
"all efficient tests," and that single restriction is what makes stretching possible again.

Good. But "fools every efficient test" is a quantifier over all algorithms — how on earth do I ever
*prove* a concrete G satisfies it? I can't enumerate tests. I need one canonical, checkable test that
captures all of them. The next-bit test is the obvious candidate, and it's the one Blum and Micali
already use: G passes it if for every efficient predictor C and every position i,

    Pr[ C(b_1 … b_i) = b_{i+1} ] < 1/2 + negligible.

That is, seeing any prefix of the output, no efficient predictor guesses the next bit better than a
coin. It's a single, natural, *checkable* property — and intuitively it should be necessary, since a
truly random string is unpredictable. The real question is whether it's *sufficient*: does
unpredictability of the next bit imply indistinguishability from *all* efficient tests?

If that equivalence holds, my life is transformed: to build a pseudorandom generator I only have to
build a next-bit-unpredictable one, which I can hope to do by *embedding a hard problem* so that
predicting the next bit means solving the hard problem. Let me try to prove the equivalence.

One direction is almost free. Suppose G is next-bit unpredictable; I want it to pass an arbitrary
test. Actually let me do the useful contrapositive on the *other* direction first, because it's the
trivial one. A predictor *is* a test: given a string, run the predictor on a prefix, and output 1 iff
its guess of the next bit was right. On G's output it's right with probability 1/2 + ε; on a truly
random string nobody can predict the next bit, so it's right with probability exactly 1/2. The test's
advantage is ε. So *if G is predictable, G fails a statistical test* — i.e. passing all tests implies
unpredictability. That's the easy half.

The half I actually need is the converse: unpredictability implies passing all tests. Suppose, for
contradiction, that some efficient test T distinguishes G's output y = G(x) (length ℓ = P(k)) from a
uniformly random ℓ-bit string, with advantage ε that is not negligible. I have a test; I must
manufacture a *predictor*, to contradict unpredictability. The bridge between "tells real from random"
and "predicts a bit" is the hybrid argument, and I'll build the hybrids by *swapping the string from
fully-random to fully-pseudorandom one bit at a time*.

Define, for i = 0, 1, …, ℓ, the experiment exp_i: draw y = G(x) and an independent uniform string r
of length ℓ; feed T the string consisting of the first i bits of y followed by the last ℓ − i bits of
r. Then exp_0 hands T an entirely random string (it's all r), and exp_ℓ hands T G's actual output
(it's all y). Write p_i = Pr[exp_i → 1]. The two ends are exactly T's two worlds, and by assumption

    |p_ℓ − p_0| ≥ ε.

If p_ℓ − p_0 is negative, I replace T by the test 1 − T. That flips all p_i to 1 − p_i, keeps the
same absolute advantage, and makes the signed end-to-end gap positive. So I may assume

    p_ℓ − p_0 ≥ ε.

Now the signed gaps Δ_i = p_{i+1} − p_i telescope:

    Σ_{i=0}^{ℓ−1} Δ_i = p_ℓ − p_0 ≥ ε.

So some position j has Δ_j = p_{j+1} − p_j ≥ ε/ℓ. This is the only loss in the averaging; I am not
paying another hidden factor. The two neighbors exp_j and exp_{j+1} differ in exactly one place: bit
j+1 of the string fed to T is a fresh random bit in exp_j and is the real bit y_{j+1} in exp_{j+1}.
T feels that one-bit substitution with a positive signed edge. That is exactly the leverage I need to
predict y_{j+1}.

Build the predictor A for this j. It is handed the first j bits of y. It picks a random guess g for bit j+1, and
a fresh random tail r of length ℓ − j − 1. It forms z = (first j bits of y) · g · r and runs T(z). If
T outputs 1, A outputs g; otherwise A outputs 1 − g. The idea: T is more likely to say "1" (looks
like the real prefix continuing) when the bit I plugged in matches the real bit, so T's verdict votes
on whether my guess was right.

Let me grind the probability, because this is exactly where these arguments quietly go wrong. Write
y_{j+1} for the true bit. Let z_1 = (first j bits of y) · y_{j+1} · r be the string with the *correct*
bit, and z_2 = (first j bits of y) · (1 − y_{j+1}) · r the string with the flipped bit. My z equals
z_1 when g = y_{j+1}, and equals z_2 when g = 1 − y_{j+1}. Then

    Pr[A correct] = Pr[ T(z)=1 ∧ g = y_{j+1} ] + Pr[ T(z)=0 ∧ g = 1 − y_{j+1} ].

Since g is a uniform bit independent of everything, g = y_{j+1} with probability 1/2 and in that case
z = z_1, while g = 1 − y_{j+1} with probability 1/2 and then z = z_2:

    Pr[A correct] = 1/2 · Pr[T(z_1)=1] + 1/2 · Pr[T(z_2)=0]
                  = 1/2 · ( Pr[T(z_1)=1] + 1 − Pr[T(z_2)=1] )
                  = 1/2 + ( Pr[T(z_1)=1] − Pr[T(z_2)=1] ) / 2.

Now I relate that residual to the hybrid gap. In exp_j, bit j+1 is a *truly random* bit, so the
string T sees is z_1 or z_2 each with probability 1/2:

    p_j = Pr[exp_j → 1] = 1/2 · ( Pr[T(z_1)=1] + Pr[T(z_2)=1] ).

In exp_{j+1}, bit j+1 is the real bit, so T sees exactly z_1:

    p_{j+1} = Pr[exp_{j+1} → 1] = Pr[T(z_1)=1].

Subtracting,

    p_{j+1} − p_j = ( Pr[T(z_1)=1] − Pr[T(z_2)=1] ) / 2 = Pr[A correct] − 1/2.

So

    Pr[A correct] = 1/2 + ( Pr[exp_{j+1}→1] − Pr[exp_j→1] )
                  = 1/2 + Δ_j
                  ≥ 1/2 + ε/ℓ.

The sign is already handled by complementing T before I choose j. The position j can be fixed for the
infinitely many k on which the distinguisher has this edge; if I want the random-index version, the
same telescoping says a uniformly chosen j has average signed edge (p_ℓ − p_0)/ℓ. Since ℓ = P(k) is
polynomial and ε is non-negligible, ε/ℓ is non-negligible — A predicts the next bit too well,
contradicting unpredictability.

That closes the loop: **next-bit unpredictability is equivalent to indistinguishability from all
efficient tests.** This is the load-bearing equivalence. It means a generator's pseudorandomness — a
universally-quantified statement over all tests — can be *established* by defeating one concrete
adversary, the next-bit predictor. And I notice the slack has to be *negligible* (the 1/Q(k)), not
some fixed constant ε_0: the proof divided the advantage by ℓ, so if I only knew "unpredictable up to
a fixed constant" the per-step ε/ℓ could collapse. The right definition carries 1/Q(k) precisely so
the hybrid survives.

There's a small bonus that falls out and I'll want it later. Indistinguishability doesn't care about
the *order* of the bits: if G is a pseudorandom generator, so is the generator G~ that runs G and
emits its bits in reverse. (Any test for G~ becomes a test for G by reversing its input before
running it; the reversed input of a uniform string is uniform, so distinguishing G~'s output from
random would distinguish G's. Since G is pseudorandom, G~ is too.) I flag this because the way I'm
about to *build* a generator will naturally produce bits "from the deep end backward," and this tells
me I'm free to emit them forward in real time.

Now, how do I actually build a next-bit-unpredictable generator? "Predicting the next bit ⇒ solving
something hard." So I want to *embed a hard problem* such that each output bit is the answer to an
instance no efficient predictor can guess. The hardness primitive I have is a one-way function: easy
to compute, hard to invert. A one-way *permutation* f on {0,1}^n is even nicer — a bijection that is
hard to invert. Here is my first instinct: walk the orbit. Start at a random seed s_0, set
s_{i} = f(s_{i-1}), and output some bit of each s_i. Because f is a permutation and s_0 is uniform,
every s_i is uniform, so I never lose entropy as I walk.

But which bit do I output? Here I hit the wall that the inverse of a one-way function is *not* the
same as a random string. f being hard to invert means I can't recover the *whole* preimage; it does
*not* mean any particular bit of the preimage is hard to guess. Worse, the candidate one-way
permutations all satisfy algebraic identities — from RSA values at x and y you read off the value at
x·y — so most bits of x leak from f(x). If I output an arbitrary bit of s_i, an efficient predictor
might compute it from the visible part and my "random" bits are anything but. Outputting a generic
bit of the orbit doesn't work.

What I need is one bit of x that is provably *unpredictable from f(x)* even though f(x) determines x.
A *hard-core* predicate B: efficiently computable from x, but given only f(x), no efficient algorithm
predicts B(x) better than 1/2 + negligible. The hardness of inverting f, I want concentrated into a
single bit. For the discrete-log permutation f(s) = g^s mod p, whose inverse is the discrete
logarithm, the useful bit can be stated two equivalent ways. From the state s, it is just the
half-interval bit of the exponent, so I can compute it efficiently. From only the image g^s, the same
question asks whether that group element is the g-principal square root of its square. That is the
hard view: predicting this bit from g^s is as hard as discrete log.

Why? If an oracle tells me that principal-square-root bit exactly, I can recover a discrete log from
right to left. Given y = g^a, I first test whether y is a quadratic residue. That tells me the last
bit of a: residue means even, nonresidue means odd. If it is odd, I multiply by g^{-1} to make the
index even. Now I can compute the two square roots of the current element, and the oracle tells me
which one is principal; replacing the current element by that root divides the remaining index by 2.
Repeating peels off every bit. If the oracle only has a 1/2 + ε edge, I randomize the instance before
asking it: multiply a target square by g^{2r} for random r, so the shifted square is distributed over
the quadratic residues while preserving which shifted root corresponds to the original root except
for a known wraparound event. Repeating this and taking the majority concentrates the edge by the
weak law of large numbers. So a noticeable predictor for the bit becomes a discrete-log solver, and
under the discrete-log assumption the bit is hard-core.

Now the generator can walk an orbit without consuming fresh randomness. Start from a uniform seed
s_0, set s_i = f(s_{i-1}), and define b_i = B(s_{i-1}) for i = 1, 2, …, ℓ. Each step costs one
application of f and reads one hard-core bit of the current state. The single-step stretch
G(x) = f(x) · B(x) already shows the principle: because f is a permutation, f(U) is uniform when U is
uniform, and the appended bit is unpredictable from that uniform-looking image. Iterating the same
state update gives any polynomial number of output bits.

The proof wants the orbit in the direction where the predictor's prefix is computable from f(x). So I
first look at the reversed output

    b_ℓ b_{ℓ−1} … b_1.

Suppose a next-bit predictor C, after seeing c bits of this reversed sequence, predicts the next one
with advantage ε. Given only f(x), I can compute f(x), f^2(x), …, f^c(x), and therefore I can compute

    B(f^c(x)), B(f^{c−1}(x)), …, B(f(x)).

Those are exactly the c prefix bits that precede B(x) in a reversed orbit segment. Feeding them to C
gives a prediction for B(x) from f(x) with the same advantage ε. Because f is a permutation, x is
uniform exactly when the corresponding orbit state is uniform, and the map from seeds to that state is
a bijection; no density or conditioning error slips into the reduction. That contradicts the hard-core
property. Therefore the reversed orbit sequence is next-bit unpredictable, hence pseudorandom by the
equivalence. And since bit-reversal preserves indistinguishability, the online sequence
b_1 b_2 … b_ℓ is pseudorandom too. This is also why I needed a *permutation*: it keeps every orbit
state uniform, makes the orbit shift a bijection, and prevents the walk from collapsing into a biased
image where the hard-core statement would no longer be the right distribution.

So I have it: a one-way permutation plus a hard-core bit yields a pseudorandom generator, stretching
a short seed to any polynomial length, indistinguishable from uniform to every efficient test.

Now the ambition I actually care about: the *function*. I want a short key to index a function from
{0,1}^k to {0,1}^k that no efficient test, even one allowed to *query* the function at points of its
choice, can tell from a truly random function. That's a strictly harder object than a generator,
because the adversary is no longer handed a fixed string — it adaptively probes, and the answers must
look mutually random across an exponential domain. I have a tool that turns a little randomness into a
little more randomness: a pseudorandom generator. Can I bootstrap a *string-stretcher* into a
*function*?

Let me start from the easy-access question that was nagging at the field, because it points right at
this. A generator with seed s implicitly defines an exponentially long pseudorandom pad b_0 b_1 b_2 …;
the open problem was whether I can fetch the i-th bit, for i up to 2^k, in poly(k) time, randomness-
preservingly. But "fetch the i-th chunk of a 2^k-long pseudorandom object, in poly time" is *exactly*
a function from the k-bit index i to a k-bit value — a keyed function indistinguishable from random.
The two problems are the same problem. So let me chase the construction.

Here's the naive attempt. Take a generator G that doubles its input, G: {0,1}^k → {0,1}^{2k}. From
seed K it gives 2k pseudorandom bits. I want 2^k different k-bit outputs, addressable by a k-bit
index. If I just chop one long G-output into chunks, I only get poly-many chunks before I run out, not
2^k. I hit the wall: a single application of G gives me a fixed, polynomial amount of stretched
material; I cannot read 2^k independent-looking values off of it. Stretching once is not enough; I
need the addressing to be *exponential* while the work per address stays polynomial.

Stare at the doubling. G takes one k-bit seed to *two* k-bit blocks. I have been reading that as "k
bits of new randomness." Read it instead as *branching*: one random-looking seed becomes two
random-looking seeds. If I call the two halves G_0(s) and G_1(s) — left half, right half — then from
one seed I get two children seeds, each itself a full k-bit seed I can recurse on. Apply it to each
child: now four. The doubling isn't "a little more randomness," it's a *node with two children.*
Iterate the branching and I am building a binary tree: depth 1 gives 2 seeds, depth 2 gives 4, depth
d gives 2^d. To get 2^k leaves I go to depth k — and the path from the root to a leaf is a string of k
bits, left/right at each level. The k-bit *input* is the address; it is the path. And a leaf is
reached by exactly k doublings, so I can compute any single leaf in k applications of G *without ever
materializing the other 2^k − 1 leaves*. That is the whole construction, and it is forced: branching
is the only way to turn "stretch a seed by a constant factor" into "address exponentially many seeds
in polynomial time," and binary input bits demand exactly two children per node, i.e. exactly the
length-doubling G.

Let me write it down. Put the key K = s at the root of a full binary tree of depth k. For an internal
node holding label v, its left child holds G_0(v) and its right child holds G_1(v). For input
x = x_1 x_2 … x_k, define

    f_K(x) = G_{x_k}( G_{x_{k-1}}( … G_{x_1}(K) … ) ),

the label of the leaf reached by following x's bits from the root. Picking a function is picking the
k-bit root K — k random bits, done. Evaluating costs k applications of G — polynomial. The whole
family is {f_K}. The only thing left, and the only hard thing, is: is f_K indistinguishable from a
truly random function under oracle access?

I'll prove it the way the equivalence taught me — a hybrid, swapping pseudorandom for truly random
one *layer* at a time. Imagine the idealized experiment where, instead of deriving the tree from one
root by G, I place *independent truly random* k-bit labels at every node of level i, and then derive
all deeper levels deterministically by applying G downward; the oracle answers a query x with the
label of x's leaf. Call this hybrid A_i. Look at the two ends. A_0 places a single random label at the
root (level 0) and derives the entire tree by G — that's *exactly* the real f_K. A_k places
independent random labels at every leaf (level k) — and a function whose value at each input is an
independent uniform string is *exactly* a truly random function. So A_0 is my construction and A_k is
the ideal, and a distinguisher with oracle access, advantage ε between them, must by the triangle
inequality distinguish some adjacent A_i, A_{i+1} with advantage ε/k.

What is the difference between A_i and A_{i+1}? In A_i, the level-(i+1) labels are obtained by feeding
each random level-i label through G to make its two children. In A_{i+1}, the level-(i+1) labels are
*themselves* fresh random. So adjacent hybrids differ by exactly: "two children = G(random parent)"
versus "two children = fresh random." That is precisely a pseudorandom-generator challenge — one
application of G against uniform. If I could distinguish A_i from A_{i+1}, I could distinguish G's
output from random.

One real problem, and it's the subtle heart of the proof: level i has 2^i nodes, exponentially many,
so I cannot actually lay down 2^i random labels — that's not a polynomial reduction. But I don't have
to. The distinguisher is polynomial; it makes at most P(k) queries; those queries touch only the
poly-many nodes that lie on their root-to-leaf paths. So I instantiate the tree *lazily*: I only ever
assign a label to a node when a query path reaches it, generating it on the fly. Across all queries
only polynomially many nodes are ever born, hence only polynomially many level-i labels are ever fed
through G. Now the swap "G(random) vs random" happens at only polynomially many places, and
distinguishing A_i from A_{i+1} reduces to distinguishing a polynomial-length tuple
(G(K_1), …, G(K_t)) of generator outputs from t·2k uniform bits — which is itself a tiny hybrid over
the t seeds, each step one G-challenge. So a single G-distinguisher is recovered.

Let me tally the advantage, because the accounting is the proof. Suppose a poly-time test, with oracle
access and at most P(k) queries, distinguishes my f_K from random with advantage at least 1/Q(k) for
infinitely many k. Let p_i be the probability that this test outputs 1 when its oracle is simulated by
A_i. Then |p_0 − p_k| > 1/Q(k). Build a string-test against G as follows: pick a level i uniformly in
{0, …, k−1}; take the given list U of at most P(k) independent 2k-bit strings and use them as the
first-seen level-i nodes' G-images — for a first-seen query prefix y_1 … y_i, take the next unused
string u = u_0 u_1 from U and answer that node's 0-child / 1-child with (the G-derived subtrees
rooted at) u_0 and u_1 respectively, deriving deeper levels by G; reuse stored values for repeated
prefixes. If U consists of genuine G-outputs on random seeds, this simulates A_i exactly; if U
consists of uniform 2k-bit strings, it simulates A_{i+1}. Averaging over the random level, the two
acceptance probabilities differ by

    | (1/k) · Σ_{i=0}^{k−1} p_i − (1/k) · Σ_{i=0}^{k−1} p_{i+1} |
      = (1/k) · |p_0 − p_k|
      > 1/(k · Q(k)).

The 1/k is exactly the random-level cost. If a definition of string testing only hands me one sample
from G rather than a polynomial list, I expand this last step into a tiny hybrid over the at most
P(k) first-seen level-i prefixes; one of those swaps still gives a one-sample G-distinguisher with
only another polynomial loss. Since k · Q(k) is polynomial, the level test has a non-negligible
advantage against G — contradicting G's pseudorandomness. Therefore no efficient test distinguishes
f_K from a random function: the tree family is a pseudorandom collection of functions. From any
pseudorandom generator I get exponentially many random-looking functions, each named by a short key
and evaluable in k generator calls.

There's a tidier way to package the same proof that I like better as bookkeeping, though it's the same
animal. Instead of counting levels, count *G-invocations*. Along the at-most-P(k) query paths the
oracle invokes G at most M = P(k)·k times total. Define hybrids H^0, …, H^M where H^m replaces the
first m of these G-invocations by fresh random doublings and leaves the rest real. Then H^0 is f_K
(every doubling real) and H^M is a random function on the queried points (every touched doubling fresh random), and
adjacent H^{m−1}, H^m differ by exactly one G-invocation swapped for uniform. A distinguisher between
H^{m−1} and H^m gives a G-distinguisher directly: embed the challenge string y ∈ {0,1}^{2k} as the
m-th doubling — if y = G(seed) the simulation is H^{m−1}, if y is uniform it is H^m — and run the test.
Random m ∈ {1,…,M}, telescope, advantage ε/M with M polynomial. Same conclusion, the swap localized
to a single PRG call. Either way the engine is the hybrid and the content is "one doubling is
pseudorandom."

For strings, the right notion
turned out to be "unpredictable next bit," equivalent to "indistinguishable from random." There ought
to be the analogous statement for functions, and it's the cleanest way to say what I've built. Say a
function family can be "polynomially inferred" if some efficient adversary, after adaptively querying
f at points of its choice, can then name a *fresh* point x — one it never queried — and recognize f(x)
among random alternatives better than chance: a "chosen-exam." A truly random function obviously
cannot be inferred — the value at an unqueried point is independent of everything seen. I claim my
family passes the same test: it *cannot* be polynomially inferred, and in fact "cannot be inferred"
is equivalent to "passes all efficient statistical tests for functions." This is Yao's next-bit
equivalence lifted one level — unpredictability of an unqueried value ⟺ indistinguishability of the
whole function.

The proof is the same shape as the next-bit argument, with the "exam point" playing the role of the
"next bit." One direction is immediate: if an inferrer wins the chosen exam with advantage, a
statistical test runs that inferrer, answers its oracle queries, then at the fresh exam point gives it
the real value and an independent random value in random order; the test outputs 1 iff the inferrer
identifies the real value. A random function makes the exam independent, so the success probability is
1/2; a win on my family becomes a distinguishing edge.

The converse is the part with the constant. Suppose a statistical test T distinguishes the family from
a random function, asks exactly P(k) distinct queries, and after complementing T if needed has
p^{P(k)}_k − p^0_k > 1/Q(k), where p^i_k is the probability that T outputs 1 when its first i queries
are answered by f from the family and all later queries are answered by independent random k-bit
strings. To infer, I choose i uniformly in {0, …, P(k)−1}, run T, answer the first i queries from my
oracle, and when T asks query i+1 I stop and name that fresh point as the exam. I receive f(x) and an
independent random y in random order. I pick one of the two values uniformly, call it z, and feed z
back to T as the answer to query i+1; all later T queries get fresh random strings. If T outputs 1, I
guess that z was the real f(x); if T outputs 0, I guess z was the random value. For a fixed i, if z is
random then T is in experiment i, and if z is real then T is in experiment i+1. Averaging over the
uniformly chosen i gives

    Pr[ guess correct ] = (1/(2 P(k))) · Σ_{i=0}^{P(k)−1} ( (1 − p^i_k) + p^{i+1}_k )
                         ≥ 1/2 + 1/(2 · P(k) · Q(k)),

because the sum collapses as consecutive p^i_k terms cancel, leaving only the end-to-end gap
p^{P(k)}_k − p^0_k. That is a non-negligible edge on the exam. So a non-inferable random-looking
function is equivalent to passing all tests, and my construction does both.

The whole causal chain, end to end: I refuse the information-theoretic notion of randomness because it
forbids stretching, and I refuse a fixed battery of tests because the adversary tests for what's not
on the list — so I define "random enough" as fooling *every efficient* test, computational
indistinguishability. I then prove that this universal notion is captured by one checkable property,
next-bit unpredictability (the hybrid that swaps random bits for pseudorandom one position at a time,
costing a 1/ℓ factor), which lets me *prove* pseudorandomness by defeating a single predictor. I embed
hardness so that predicting the next bit means inverting a one-way permutation: I distill the
inversion difficulty into one provably-unpredictable hard-core bit per orbit step, and reading that
bit along the orbit of the permutation yields a pseudorandom generator from a short seed. Finally I
read the generator's *doubling* not as "more bits" but as *branching* — one random-looking seed into
two — and iterating the branch k times builds a depth-k binary tree whose 2^k leaves are addressed by
the k-bit input, each reachable in k generator calls; a hybrid that replaces one tree level (or one
doubling) of pseudorandom labels by truly random labels at a time, instantiated lazily so it stays
polynomial, proves the leaves are indistinguishable from a random function. A short key, exponentially
many random-looking functions, computed on demand — and they are exactly the functions no efficient
chosen-exam can predict.

    # The construction, made concrete (length-doubling PRG -> PRF tree).
    # G: {0,1}^k -> {0,1}^{2k};  G0, G1 are its left/right k-bit halves.

    def G0(s, k):            # left half of one PRG doubling
        return prg_double(s, k)[:k]

    def G1(s, k):            # right half of one PRG doubling
        return prg_double(s, k)[k:]

    # the BMY pseudorandom generator: one hard-core bit per step of a one-way permutation.
    def prg_bits(seed, length):
        s = seed
        out = []
        for _ in range(length):
            out.append(hard_core_bit(s))          # the one provably-unpredictable bit
            s = one_way_permutation(s)             # advance along the orbit; stays uniform
        return out                                  # order does not matter (proved), so emit forward

    # the GGM tree: a short key K indexes a random-looking function on k-bit inputs.
    def f_K(K, x, k):
        label = K                                   # root holds the key
        for bit in x:                               # walk the k-bit input down the tree
            label = G0(label, k) if bit == 0 else G1(label, k)   # left/right child = one doubling
        return label                                # leaf label = f_K(x); only k PRG calls, never 2^k

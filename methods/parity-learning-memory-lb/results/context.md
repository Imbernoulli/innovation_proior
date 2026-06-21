# Context: Time, Space, and the Sample Cost of Parity Learning

## Research question

A learner wants to recover an unknown $x \in \{0,1\}^n$, chosen uniformly at random, from a
stream of linear samples. At each step $t$ it sees $(a_t, b_t)$ where $a_t$ is uniform over
$\{0,1\}^n$ and $b_t = a_t \cdot x \pmod 2$ is the inner product. The samples arrive one at a
time; once a sample has gone by, the learner cannot see it again unless it deliberately stored
it in its memory. This is **parity learning**.

Parity learning is trivially solvable two different ways, sitting at opposite corners of a
resource trade:

- **Lots of memory, few samples.** Collect $n$ linearly independent equations $a_t \cdot x = b_t$
  and solve the system by Gaussian elimination. This needs only $O(n)$ samples but stores up to
  $\Theta(n^2)$ bits (an $n \times n$ system over $\mathbb{F}_2$).
- **Little memory, many samples.** Enumerate candidate strings: hold one candidate $x'$ at a time
  (about $n$ bits), keep drawing samples until one rules $x'$ out, move to the next candidate.
  This uses $n + o(n)$ bits of memory but, in the worst case, an exponential number of samples
  before the right candidate survives.

The question is whether this trade is **forced**. Is there a learning problem that is easy with
quadratic memory yet provably blows up to an exponential number of samples the moment the memory
budget drops below quadratic — say, down to linear, the size of a single sample? If parity were
such a problem, it would be the first formally proved example showing that a *large memory* can be
indispensable for *fast* learning: not because the learner lacks computational power, but purely
because it cannot hold enough state. Settling this for parity is the goal.

The stakes reach past learning theory. A sample lower bound of this kind would say something about
the structure of computation (a strong time-space tradeoff, with "time" measured in samples) and
would have an immediate cryptographic reading: a shared inner-product bit $a \cdot x$ used as a
one-time pad is safe against a memory-bounded eavesdropper exactly when no such eavesdropper can
reconstruct $x$.

## Background

**Parity, characters, and Fourier analysis over $\mathbb{F}_2^n$.** For $a \in \{0,1\}^n$ the
function $\chi_a(x) = (-1)^{a \cdot x}$ is a character of the group $\mathbb{F}_2^n$. The $2^n$
characters form an orthonormal basis of real functions on $\{0,1\}^n$: any $f$ has a Fourier
expansion $f = \sum_a \widehat f(a)\,\chi_a$ with $\widehat f(a) = \mathbb{E}_x[f(x)\chi_a(x)]$,
and Parseval's identity $\mathbb{E}_x[f(x)^2] = \sum_a \widehat f(a)^2$ holds. The key structural
fact is that distinct parities are *orthogonal*: $\mathbb{E}_x[\chi_a(x)\chi_{a'}(x)] = 0$ for
$a \neq a'$. This orthogonality is the engine behind every hardness result for parity.

**The inner product as a randomness extractor.** For uniform $a$ and any fixed nonzero $x$, the
bit $a \cdot x$ is an unbiased coin. More generally, the inner-product two-source extractor (the
Lindsey-lemma / Hadamard phenomenon) says inner products mix weak sources toward uniform; the
statistical-distance-to-uniform of $a\cdot x$ is controlled by the min-entropy of the sources.
This "inner product extracts" intuition is standard equipment.

**Linear-algebraic structure of the consistent set.** After the learner has seen samples
$(a_1,b_1),\dots,(a_t,b_t)$, the set of strings consistent with everything seen is
$\{x' : a_i \cdot x' = b_i \ \forall i\}$ — an intersection of hyperplanes, i.e. an **affine
subspace** of $\{0,1\}^n$. Its dimension starts at $n$ and can drop by at most one per sample.
Write $A(n)$ for the set of all affine subspaces of $\{0,1\}^n$, and $U_w$ for the uniform
distribution over $w \in A(n)$. These objects are elementary and were available to anyone.

**The statistical-query (SQ) model and parity (Kearns 1998; Blum–Furst–Jackson–Kearns–Lipton–
Sellie 1994).** An SQ learner never sees raw labeled examples; it issues queries of the form "what
is $\mathbb{E}_{(x,\ell)}[\psi(x,\ell)]$ for this bounded $\psi$?" and gets an answer accurate to
some tolerance $\tau$. Kearns proved parities are *not* efficiently SQ-learnable: for any single
query function $\chi$, the value $P_\chi(f) = \Pr[\chi = 1]$ has *variance* $O(2^{-n})$ over a
uniformly random target parity $f$ — that is, any one bounded statistic of a labeled example is
almost the same number whatever the true parity is, so it reveals essentially nothing. The proof
is a second-moment computation that, at bottom, is the pairwise orthogonality of the $2^n$
parities. The upshot: to distinguish among parities you need exponentially many tolerance-$\tau$
queries. SQ-hardness is the canonical reason "parity is hard for restricted learners."

**From SQ to communication (Steinhardt–Valiant–Wager 2015).** SVW recast resource-bounded
learning as complexity classes. $\mathrm{MEM}(b)$ is the class of concepts learnable from
$\mathrm{poly}(n)$ examples by an algorithm using $\le b$ bits of memory; $\mathrm{COM}(b)$ uses
an algorithm that may extract $\le b$ bits of information *from each example*; $\mathrm{sCOM}(b)$
is the streaming one-message-per-example version. Their main theorem is an equivalence: for any
constant $C$,
$$\mathrm{COM}(1) = \mathrm{COM}(C\log n) = \mathrm{sCOM}(1) = \mathrm{sCOM}(C\log n) = \mathrm{SQ}.$$
A protocol extracting $b$ bits per example can be simulated by an SQ algorithm asking $2^b m$
queries at tolerance $\tau = 1/(2^{b+1}m)$. Pushing the SQ-parity lower bound through, SVW get
that any communication protocol learning parity with $\le n/4$ bits per example needs $2^{\Omega(n)}$
examples, and a striking multiparty corollary (each of many parties sees a few examples, broadcasts
$\le n/16$ bits, and you need $2^{\Omega(n)}$ parties).

**Classical time-space tradeoffs for computation (Borodin–Cook; Beame–Jayram–Saks 1998;
Ajtai 1999; Beame–Saks–Sun–Vee 2000; for SAT: Fortnow 1997, Fortnow–Lipton–van Melkebeek–Viglas
2005, R. Williams 2006–07; survey: van Melkebeek 2007).** A rich line of work proves time-space
lower bounds for *computing* explicit functions, in the branching-program model. The best known
say: any branching program computing certain explicit $f$ needs either $\ge n^{1-\epsilon}$ memory
or time $\ge \Omega(n\sqrt{\log n / \log\log n})$. For SAT in the uniform setting one gets memory
$\ge n^{1-\epsilon}$ or time $\ge n^{1+\delta}$. Two features matter: (i) these are *sub-quadratic*
time lower bounds — even at logarithmic memory, quadratic-time lower bounds for computing a
function are not known; and (ii) the model gives the algorithm the *input for free* — the input is
always re-readable and its storage is not charged as memory.

**Bounded-storage cryptography (Maurer 1992; Cachin–Maurer 1997; Aumann–Rabin 1999; Aumann–Ding–
Rabin 2002; Vadhan 2003; Dziembowski–Maurer 2004).** Here security rests not on computational
hardness but on a cap on the *adversary's memory*. The standard template streams a long public
random string that the adversary cannot fully store; legitimate parties use a shared key to extract
bits the adversary knows little about.

## Baselines

**Gaussian elimination (the high-memory learner).** Maintain a reduced system of independent
equations $a_i \cdot x = b_i$. After $O(n)$ samples the system has full rank and $x$ is read off.
Cost: $O(n)$ samples, $\Theta(n^2)$ memory.

**Candidate enumeration (the low-memory learner).** Hold one candidate string at a time, draw
samples until it is contradicted, advance. Cost: $n+o(n)$ memory but up to $2^{\Theta(n)}$ samples.

**SQ lower bound for parity (Kearns 1998).** Exponentially many tolerance-$\tau$ statistical queries
are required to learn parity, because any single bounded statistic has $\approx 2^{-n}$ correlation
with the choice of target parity.

**SVW communication / sCOM lower bound for parity (Steinhardt–Valiant–Wager 2015).** Any protocol
compressing each example to $b$ bits before the next example arrives needs $2^{\Omega(n)}$ examples
to learn parity. SVW conjectured (Conjecture 1.1) that any parity learner needs either $\ge n^2/4$
bits of memory or $\ge 2^{n/4}$ samples, and that parity should therefore separate efficient
bounded-memory learning from PAC learning.

**Branching-program time-space lower bounds for computing functions (BJS98, Ajtai99, BSSV00,
and the SAT line).** These give the most general non-uniform tradeoffs known, with the input
available for re-reading at any step (not charged as memory).

## Evaluation settings

This is a question about *provable limits*, so the yardsticks are the resource axes and the model,
not benchmark datasets.

- **Problem instance.** Parity learning of size $n$: target $x \sim U(\{0,1\}^n)$, samples
  $(a_t,b_t)$ with $a_t \sim U(\{0,1\}^n)$ and $b_t = a_t \cdot x \pmod 2$.
- **Resources measured.** (1) Memory, in bits; (2) the number of samples $m$ ("time"); (3) the
  success probability $\Pr[\text{learner outputs (a set containing) } x]$.
- **Computational model.** The branching program: a layered directed multigraph, $m+1$ layers of
  at most $d$ vertices, where layer $t$ = time step $t$ and a vertex = a memory state, so $\log_2 d$
  is the memory in bits and $m$ is the number of samples. Each non-leaf vertex has one outgoing edge
  per possible sample $(a,b) \in \{0,1\}^n \times \{0,1\}$; an input stream traces a computation-path
  from the start vertex to a leaf, whose label is the output. This is the strongest non-uniform
  model: the learner has unbounded computation, and only memory ($\log d$) and samples ($m$) are
  charged. A lower bound here binds every uniform algorithm too.
- **Target regime to certify.** Width up to $d = 2^{\Theta(n^2)}$ (i.e. up to roughly quadratic
  memory) and length up to $m = 2^{\Theta(n)}$ (subexponentially many samples); the claim to test is
  whether, in some sub-quadratic-memory band, the success probability is forced to be exponentially
  small.

## Code framework

This is a theorem-and-proof result; the "framework" is the proof scaffold — the objects that exist
before any new idea, with empty slots for the contribution. No numerical experiment is run; below is
the skeleton of definitions and obligations that the proof will fill in.

```python
# --- ambient objects, all elementary / pre-existing -------------------------
# F2^n = {0,1}^n with inner product a.x = sum a_i x_i mod 2.
# A(n) : the set of affine subspaces of F2^n (intersections of hyperplanes a.x = b).
# U_w  : uniform distribution over an affine subspace w in A(n); U_n := uniform over F2^n.
# Characters chi_a(x) = (-1)^{a.x}; Fourier transform / Parseval available over F2^n.

class AffineSubspace:
    """An affine subspace of F2^n: {x : Mx = c} for some linear system. dim known."""
    def dimension(self): ...
    def uniform_distribution(self): ...        # returns U_w

# --- the learner as a branching program (the standard model) ----------------
class BranchingProgram:
    """Layered DAG. layers = m+1, width <= d. log2(d) = memory bits, m = #samples.
    A vertex = a memory state; an edge labeled (a,b) = the transition on reading a sample.
    A leaf carries an output guess about x."""
    def __init__(self, length_m, width_d): ...
    def vertices_in_layer(self, t): ...
    def edge(self, u, a, b): ...               # the vertex reached from u on sample (a,b)
    def computation_path(self, samples): ...    # start-vertex -> leaf
    def output(self, leaf): ...                 # the guess produced at a leaf

# --- what we are handed and what we must produce -----------------------------
# GIVEN: a branching program for parity learning with length m <= 2^{a n},
#        width d <= 2^{c n^2}, run on x ~ U_n and honest samples (a_t, b_t=a_t.x).
# GOAL : upper-bound its success probability Pr[ x in output ].

def progress_toward_x(program, path):
    """A monotone quantity that measures how much the learner has pinned down x
    along its computation-path. The whole proof hinges on choosing this well."""
    # TODO: the right way to summarize 'what the learner knows about x' at a vertex,
    #       and the right potential to track along the path.
    pass

def per_step_advance_probability(program, step):
    """Probability that one fresh random sample makes real progress on x."""
    # TODO: bound how often a single uniform a_t can push the learner forward.
    pass

def success_upper_bound(program, n, c, alpha):
    """The theorem: if memory < ~quadratic and samples are sub-exponential,
    the success probability is exponentially small."""
    # TODO: assemble the per-step bound over m steps and over the program's states
    #       into Pr[success] <= O(2^{-alpha n}).
    pass
```

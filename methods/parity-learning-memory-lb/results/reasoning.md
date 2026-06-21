Let me start from the thing that actually bugs me. Parity learning has two completely different
algorithms, and they live at opposite ends of a resource axis. One: gather $n$ independent linear
equations $a_t \cdot x = b_t$ and run Gaussian elimination — $O(n)$ samples, but I'm holding an
$n \times n$ system over $\mathbb{F}_2$, so $\Theta(n^2)$ bits of memory. Two: guess and check —
keep a single candidate $x'$ in $n$ bits, draw samples until one of them refutes $x'$, then move on
— tiny memory, but in the worst case I burn through an exponential number of samples before the true
$x$ is the only survivor. So I can buy down samples with memory, or buy down memory with samples.

The question that won't let go: is this trade *forced*? Could there be a clever algorithm in the
middle — say linear memory, the size of just one sample — that still learns $x$ from a polynomial
number of samples? Intuitively it feels like the answer is no, that you genuinely need to *remember*
a lot to learn fast. But "intuitively no" is worth nothing here. And the embarrassing state of
affairs is that nobody has a non-trivial sample lower bound for *any* learning problem even when the
memory is capped at the length of one sample. Not parity, not anything. So if I could prove that a
sub-quadratic-memory learner needs exponentially many samples for parity, that would be the first
example, period, of memory being provably crucial for fast learning.

What tools do I have for "this learner can't do it"? The natural reflex is the statistical-query
lower bound. Parity is the textbook SQ-hard problem: if a learner only ever interacts with the data
through bounded-tolerance averages $\mathbb{E}[\psi(x,\ell)]$, then it needs exponentially many
queries, because — and this is the heart of it — for any single bounded statistic $\chi$ of a labeled
example, the value $P_\chi(f) = \Pr[\chi=1]$ has variance only about $2^{-n}$ over a uniformly random
target parity $f$. Let me actually see why, because the *reason* matters. Fix any $\chi:\{0,1\}^n
\times \{0,1\} \to \{0,1\}$. Over a uniform random $x$ and a uniform random parity $f$,
$\mathbb{E}_f[P_\chi(f)] = 2^{-n}\sum_x \mathbb{E}_f[\chi(x,f(x))]$, and for each fixed $x$ the bit
$f(x)$ is an unbiased coin over the choice of $f$, so $\mathbb{E}_f[\chi(x,f(x))]$ is just the average
of $\chi(x,0)$ and $\chi(x,1)$. The second moment $\mathbb{E}_f[P_\chi(f)^2]$ brings in cross terms
$\mathbb{E}_f[\chi(x,f(x))\chi(y,f(y))]$, and for $x \neq y$ the pair $(f(x),f(y))$ is a *uniform*
pair over the random parity (two distinct points impose two independent linear constraints), so those
cross terms factor and the variance collapses to $O(2^{-n})$. At bottom this is the pairwise
orthogonality of the $2^n$ characters. Any one bounded look at one labeled example is essentially the
same number no matter which parity is the truth.

So SQ says: per-query, you learn almost nothing. And Steinhardt, Valiant and Wager turned that into a
clean *communication* statement: a learner that compresses each example to $b$ bits before the next
arrives can be simulated by an SQ algorithm with $2^b m$ queries at tolerance $\sim 1/(2^{b+1}m)$,
so compressing parity samples to, say, $n/4$ bits each forces $2^{\Omega(n)}$ samples. They built a
whole framework, $\mathrm{COM}(b)$, $\mathrm{sCOM}(b)$, $\mathrm{MEM}(b)$, and showed
$\mathrm{COM}(1) = \mathrm{sCOM}(\text{O}(\log n)) = \mathrm{SQ}$.

A communication bound constrains *bits per sample*: each example must be squeezed to $b$
bits *in isolation* before the learner reads the next one. But a memory-bounded learner doesn't have
to compress any single example to few bits. It reads each sample in full. The only thing it can't do
is carry a large *accumulated* state across samples. The state after $t$ steps can depend, adaptively,
on the entire history $(a_1,b_1),\dots,(a_t,b_t)$ — it just has to fit in $b$ bits *total*. That is a
strictly more powerful adversary than a per-sample compressor. SVW saw this too: they could prove the
memory version only "for algorithms whose memory states correspond to subspaces," and admitted a full
proof eluded them. So the SQ/communication machinery, as it stands, will not settle the memory
question. The difficulty is exactly that memory is *cumulative and adaptive*, not per-sample.

Maybe the classical time-space tradeoff literature for *computing* functions saves me — branching
programs, the Beame–Jayram–Saks and Ajtai and Beame–Saks–Sun–Vee results, the SAT line. But two
things kill that hope. First, those results only get *sub-quadratic* time lower bounds; even at
logarithmic memory, nobody knows a quadratic-time lower bound for computing an explicit function. I
want *exponential* time (samples), which is way out of their reach. Second, and this is the real
reason it's out of reach: in those models the input is stored *for free* — it's always re-readable,
and its storage doesn't count as memory. That is precisely the wrong assumption for learning. In
learning, once a sample goes by, it's gone unless I paid to store it. I can always get a *fresh*
sample "as good as the old one," but I can't re-examine the *same* one. That asymmetry — fresh
randomness is cheap, but memory of past randomness is expensive — is the entire phenomenon I'm trying
to exploit, and the computation literature throws it away.

OK so I need to think directly about a memory-bounded learner reading a stream, with no per-sample
constraint and no free input. Let me pick the most general, least-assuming model so that a lower
bound here binds everything. A learner with $s$ bits of memory is, at each time step, in one of at
most $2^s$ states; reading a sample $(a,b)$ deterministically (worst case; randomness only helps
me prove a lower bound if I allow it, but let me start deterministic and general) moves it to a new
state. Unroll this in time: layer $t$ holds the possible states at step $t$, an edge from a state in
layer $t-1$ to a state in layer $t$ is labeled by the sample $(a,b)$ that causes that transition.
That's a *branching program*: $m+1$ layers (one per sample, plus the start), width $d = 2^s$, every
non-leaf vertex has exactly one outgoing edge per possible $(a,b) \in \{0,1\}^n \times \{0,1\}$, a
single start vertex, leaves labeled by an output guess. A stream of samples traces a computation-path
from start to a leaf. Width $d$ is memory $\log_2 d$; length $m$ is the number of samples. The learner
gets infinite computation — only memory and samples are charged. Beautiful: a lower bound in this
model is a lower bound for every algorithm, uniform or not. So I want to show: if $\log d \ll n^2$
and $m$ is sub-exponential, the path lands on a leaf whose guess contains $x$ with only exponentially
small probability.

Now, what does a vertex *know* about $x$? Here's where the special structure of *parity* helps, and
where I should let the linear algebra lead. After the learner has seen $(a_1,b_1),\dots,(a_t,b_t)$,
the set of $x'$ consistent with everything is $\{x' : a_i \cdot x' = b_i \ \forall i\}$. That's an
intersection of hyperplanes — an *affine subspace* of $\{0,1\}^n$. Its dimension starts at $n$ and
drops by at most one each time a new equation is genuinely independent. So the honest "knowledge state"
is an affine subspace, and "learning $x$" means driving that subspace down to dimension $0$, i.e.
shrinking it to the single point $x$.

But wait — the learner's memory state is *not* the consistent subspace. A general branching program
vertex is some arbitrary function of the history; it might lump together many different histories
(that's the whole point of having bounded width — it must merge states). So I have two notions: the
true consistent affine subspace, which is the *ideal* knowledge, and the actual vertex, which is a
coarse, possibly weird summary. Let me first pretend the learner is forced to use affine subspaces as
its states — call that an *affine* branching program — get a clean lower bound there, and only
afterward worry about general programs. Divide and conquer.

So: an affine branching program. Every vertex $v$, not just leaves, carries a label $w(v) \in A(n)$,
an affine subspace, with the rule that the start vertex is labeled $\{0,1\}^n$ and along each edge
the label can only get tighter in the consistent sense: for an edge $e=(u,v)$ labeled $(a,b)$, define
$w(e) = w(u) \cap \{x' : a\cdot x' = b\}$, and demand $w(e) \subseteq w(v)$. Call that *soundness*.
What does soundness buy me? By induction along the path: if $x \in w(u)$ and the edge taken is the
one labeled $(a, a\cdot x)$ (the honest sample), then $x \in w(u) \cap \{x' : a\cdot x' = a\cdot x\}
= w(e) \subseteq w(v)$. The base case $x \in \{0,1\}^n = w(\text{start})$ is free. So along the honest
computation-path, *always* $x \in w(v)$. That means an affine branching program has success
probability exactly $1$ — it never excludes $x$. So the only way it can be a *good* learner, in the
sense of actually pinning down $x$, is to make the *dimension* of $w(\text{leaf})$ small. A leaf with
$w$ of dimension $0$ is "$x$ is this exact point"; a leaf with $w$ of dimension $k$ is "$x$ is one of
$2^k$ points." So for affine programs, "learn $x$" has become "drive $\dim w$ down," and *that* is a
purely geometric event I can try to bound the probability of.

Let me fix a target: I'll show that reaching any vertex whose label has small dimension is
exponentially unlikely. Concretely, fix some threshold $k$ (I'll tune it later, something like a
constant fraction of $n$), and show that the probability the path reaches a vertex with $\dim w(v)
\le k$ is at most $2^{-\Omega(n^2)}$. Since the dimension drops by at most one per step, the path must
pass through a vertex of dimension *exactly* $k$ before it can reach anything smaller, so it's enough
to fix one vertex $v$ with $\dim w(v) = k$ and bound the probability the path reaches *that* $v$, then
union over all such $v$. (And I can throw away every vertex of dimension $< k$ — assume the program
only has vertices of dimension $\ge k$, redefining dimension-$k$ vertices as leaves.)

So: fix $v$, $\dim w(v) = k$. I need a *progress measure* — something that has to climb a lot for the
path to reach $v$, and that climbs only rarely. The natural geometric object: look at the directions
*orthogonal* to the subspaces. Let $s$ be the vector space orthogonal to $w(v)$, that is
$s = \{a \in \{0,1\}^n : \exists b, \ \forall x' \in w(v),\ a\cdot x' = b\}$ — the set of linear tests
that are *constant* on $w(v)$. Since $\dim w(v) = k$, the orthogonal space $s$ has dimension $n-k$.
Along the path, let $S_i$ be the space orthogonal to $w(V_i)$ at the $i$-th vertex; $\dim S_i =
n - \dim w(V_i)$, which is the number of independent linear constraints the learner has "locked in"
by step $i$. The measure I'll watch is the overlap between what the learner has locked in and the
target: $Z_i = \dim(S_i \cap s)$.

Why is this the right thing? Because to *reach* $v$, the learner's locked-in constraints must, at the
end, include *every* constraint defining $w(v)$ — that is, $s \subseteq S_{\text{last}}$, so
$Z = \dim(S \cap s)$ has to climb all the way to $\dim s = n - k$. It starts at $Z_0 = 0$ (the start
vertex's orthogonal space is trivial). And how fast can it climb? Here soundness pays off again. Along
edge $i$, labeled $(a_i, b_i)$, the new orthogonal space $S_i$ is contained in $\mathrm{span}(S_{i-1}
\cup \{a_i\})$ — because the only *new* linear test the edge adds is $a_i$ itself (the constraint
$a_i \cdot x' = b_i$), and $w(V_i) \supseteq w(e) = w(V_{i-1}) \cap \{a_i \cdot x' = b_i\}$ means
$S_i$ can't contain any direction not already in $\mathrm{span}(S_{i-1}, a_i)$. So $\dim S_i \le
\dim S_{i-1} + 1$, and therefore $Z_i \le Z_{i-1} + 1$: the overlap rises by at most one per step.

Good, so $Z$ goes $0 \to n-k$, in steps of at most one, over $m$ samples. For the path to reach $v$
it has to make $n-k$ *rises*. Now: how likely is a single rise? A rise at step $i$ means $\dim(S_i
\cap s) > \dim(S_{i-1}\cap s)$. Using $S_i \subseteq \mathrm{span}(S_{i-1} \cup \{a_i\})$, an increase
in the intersection with $s$ can only happen if $\mathrm{span}(S_{i-1}\cup\{a_i\}) \cap s$ is strictly
bigger than $S_{i-1}\cap s$, which forces: there is some $a \in S_{i-1}$ with $a \oplus a_i \in s$.
(If adding $a_i$ to $S_{i-1}$ creates a new direction inside $s$, that new direction is a combination
$a \oplus a_i$ with $a \in S_{i-1}$, and it lies in $s$.) Now $a_i$ is a *fresh uniform* sample,
independent of everything before. For a *fixed* $a \in S_{i-1}$, the event $a \oplus a_i \in s$ is
the event that the random $a_i$ lands in the coset $a \oplus s$, which has probability $2^{\dim s - n}
= 2^{(n-k) - n} = 2^{-k}$. So each fixed direction has only a $2^{-k}$ chance of triggering a rise.

I want a union bound over $a \in S_{i-1}$, but I have to be careful not to overcount: if $a \oplus a_i
\in s$ for one $a$, then it holds for every $a' \in a \oplus (S_{i-1}\cap s)$ too (adding an element of
$S_{i-1}\cap s$, which is in $s$, keeps $a' \oplus a_i \in s$). So the genuinely distinct possibilities
are cosets, and there are $2^{\dim S_{i-1} - \dim(S_{i-1}\cap s)} = 2^{\dim S_{i-1} - Z_{i-1}}$ of
them. And $\dim S_{i-1} \le n - k$ (the orthogonal space can't exceed codimension $n-k$ if we keep
vertices of dimension $\ge k$ — more carefully $\dim S_{i-1} = n - \dim w(V_{i-1}) \le n - k$). So,
conditioned on any event that fixes $x, a_1,\dots,a_{i-1}$ (hence fixes $Z_{i-1}$ and $S_{i-1}$), the
probability of a rise at step $i$ is at most
$$
2^{\,n - k - Z_{i-1}} \cdot 2^{-k} = 2^{\,n - 2k - Z_{i-1}}.
$$
That's the engine. Each rise is exponentially unlikely, and gets *more* unlikely as $Z$ climbs (the
$-Z_{i-1}$ in the exponent), which is exactly the kind of telescoping I want.

Now assemble. To reach $v$, there exist indices $i_1 < \dots < i_{n-k}$ at which the $j$-th rise
happens, $Z_{i_{j-1}} = j-1$ and $Z_{i_j} = j$. Fix such a tuple. Conditioning step by step, the
probability of the rise at $i_j$ given everything before is at most $2^{n - 2k - (j-1)}$ (because right
before the $j$-th rise, $Z = j-1$). So for a fixed tuple,
$$
\Pr[\text{rises at } i_1,\dots,i_{n-k}] \le \prod_{j=1}^{n-k} 2^{\,n - 2k - (j-1)}
= 2^{\sum_{j=0}^{n-k-1}(n - 2k - j)}.
$$
There are fewer than $m^{\,n-k}$ choices of the tuple $i_1 < \dots < i_{n-k} \in [m]$, so the union
bound gives the probability the path reaches $v$ as at most
$$
m^{\,n-k}\cdot 2^{\sum_{j=0}^{n-k-1}(n - 2k - j)}.
$$
That's the affine lower bound — call it the reach bound. Let me sanity-check the shape: with $k$ a
constant fraction of $n$ so that $n - 2k$ is *negative* (say $k > n/2$), each factor $2^{n-2k-j}$ is
exponentially small, and there are $n-k = \Theta(n)$ of them, so the product is $2^{-\Theta(n^2)}$;
the $m^{n-k}$ from the union bound is $2^{(n-k)\log_2 m}$, which I can afford as long as $\log_2 m$ —
the number-of-samples exponent — is small enough that it doesn't eat the $2^{-\Theta(n^2)}$. Exactly
the trade I want: many samples ($m$ large) fights the bound, but only polynomially in the exponent.
Hold onto the exact constants; I'll need them.

So affine branching programs are handled. But — and this is the hard part, the part SVW couldn't get
past — a *general* branching program's vertex is not an affine subspace. It's an arbitrary memory
state, and the conditional distribution of $x$ given "the path reached this vertex" can be some messy
distribution, not uniform on a subspace. I cannot run the dimension argument on a general vertex,
because "the orthogonal space to the vertex" isn't even defined. I need to *reduce* the general case
to the affine case.

Let me think about what the conditional distribution of $x$ at a general vertex actually looks like.
At the start, $x$ is uniform on $\{0,1\}^n$. Suppose, inductively, that at every vertex of the
*previous* layer the conditional distribution of $x$ is *close to* uniform on some affine subspace
(that's the property I'll maintain). Take a vertex $v$ in the current layer. To reach $v$ you came in
along some set of incoming edges, each from a previous-layer vertex $u$ with $x$ approximately
uniform on $w(u)$, via a sample $(a,b)$; conditioning on that edge restricts $x$ to (approximately)
uniform on $w(u) \cap \{a\cdot x = b\}$, which is again an affine subspace. So the conditional
distribution at $v$ is, approximately, a *convex combination of uniform distributions over affine
subspaces*. Now I want to *replace* that messy mixture at $v$ by a single affine subspace — to make
$v$ "affine." The naive move: split $v$ into one copy per affine subspace in the mixture. But there
are up to $|A(n)| = 2^{\Theta(n^2)}$ affine subspaces, and doing this at every vertex of every layer
would make the width astronomically — uncontrollably — large. I need to *group* many subspaces into a
few representatives, each representative an affine subspace containing its whole group, such that the
uniform distribution over the representative is *close* to the weighted average of the uniforms over
the group's members. If I can do that with few groups, the width stays controlled.

So now the question is purely about distributions over affine subspaces: given a random affine
subspace $W \in A(n)$ (drawn from the mixture), when is the mixture $\mathbb{E}_W[U_W]$ close to
uniform, and can I always carve off a big near-uniform chunk? This is where the *inner-product
extractor* shows up, and the right language is Fourier analysis over $\mathbb{F}_2^n$.

Take any affine subspace $w$. What are the Fourier coefficients of $U_w$? Recall $\widehat{U_w}(a) =
\mathbb{E}_{y\sim U_w}[(-1)^{a\cdot y}]$. If $a \cdot y$ is *constant* over $y \in w$ — i.e. $a$ is in
the orthogonal space of $w$ — then this expectation is $\pm 1$ times $\mathbb{E}_{y}[\cdot] = \pm$ a
constant; normalized as a function (not a distribution), the coefficient is $2^{-n}$ if $a\cdot y
\equiv 0$ on $w$, $-2^{-n}$ if $a\cdot y \equiv 1$ on $w$. And if $a \cdot y$ is *not* constant on
$w$ — i.e. $a$ is not orthogonal to $w$ — then $a\cdot y$ is balanced over the subspace and the
coefficient is $0$. So
$$
\widehat{U_w}(a) = \begin{cases} 2^{-n} & \text{if } \forall y\in w:\ a\cdot y = 0,\\
-2^{-n} & \text{if } \forall y\in w:\ a\cdot y = 1,\\ 0 & \text{otherwise.}\end{cases}
$$
Averaging over the random $W$, $\widehat{\mathbb{E}_W[U_W]}(a) = 2^{-n}\big(\Pr_W[\forall y\in W:
a\cdot y = 0] - \Pr_W[\forall y\in W: a\cdot y = 1]\big)$, and at $a=\vec 0$ this is $2^{-n}$ (every
$w$ contains... well, $a=0$ test is always $0$, so the coefficient is $2^{-n}$). The uniform
distribution $U_n$ has $\widehat{U_n}(a) = 2^{-n}$ at $a=0$ and $0$ elsewhere. So the *difference*
$\mathbb{E}_W[U_W] - U_n$ lives entirely on $a \neq 0$, and its coefficient there is the difference
of two "$a\cdot x$ is pinned to a constant on $W$" probabilities. So the mixture differs from uniform
*only* through linear tests $a\cdot x$ that the mixture pins to a constant. If no linear test is pinned
with non-negligible probability, the mixture is essentially uniform — which is exactly the inner
product acting as a strong extractor.

Make it quantitative. Suppose for every $a \neq 0$ and every $b \in \{0,1\}$,
$\Pr_W[\forall y \in W: a\cdot y = b] \le 2^{-r}$ for some $r$. Then for each $a \neq 0$, the
coefficient $|\widehat{\mathbb{E}_W[U_W]}(a)|$ is at most $2^{-n}\cdot 2^{-r}$ (the difference of two
things each $\le 2^{-r}$, but actually the bound I want is $\le 2^{-n}\cdot 2^{-r}$ from the
individual probabilities — taking the larger of the two and dropping the subtraction only helps).
Summing over the $2^n$ nonzero $a$:
$$
\sum_{a\neq 0}\big(\widehat{\mathbb{E}_W[U_W]}(a) - \widehat{U_n}(a)\big)^2
< 2^n\cdot\big(2^{-n}\cdot 2^{-r}\big)^2 = 2^{-n - 2r}.
$$
Now Cauchy–Schwarz to go from $\ell_2$ to $\ell_1$, and Parseval to evaluate the $\ell_2$. Writing
$P = \mathbb{E}_W[U_W]$, $Q = U_n$ as functions on $\{0,1\}^n$,
$$
\Big(\mathbb{E}_{x}\big|P(x)-Q(x)\big|\Big)^2 \le \mathbb{E}_x\big(P(x)-Q(x)\big)^2
= \sum_a \big(\widehat P(a) - \widehat Q(a)\big)^2 < 2^{-n-2r},
$$
where the $x$ is uniform over $\{0,1\}^n$ and Parseval gives the middle equality. So $\mathbb{E}_x|P-Q|
< 2^{-(n+2r)/2}$. The $\ell_1$ distance between the two distributions is $|P - Q|_1 = \sum_x |P(x) -
Q(x)| = 2^n\,\mathbb{E}_x|P(x)-Q(x)| < 2^n\cdot 2^{-(n+2r)/2} = 2^{-(r - n/2)}$. There it is:
$$
\big|\mathbb{E}_W[U_W] - U_n\big|_1 < 2^{-(r - n/2)}.
$$
If no linear test is pinned with probability above $2^{-r}$, the mixture is within $2^{-(r-n/2)}$ of
uniform. (And I'll want $r \ge n/2$ so this is genuinely small. The $n/2$ loss is the Cauchy–Schwarz
slack — going from squared-$\ell_2$ over $2^n$ coordinates to $\ell_1$ costs a $2^{n/2}$ — and it's
benign because I'll work with $r$ a constant fraction above $n/2$.)

But the hypothesis "*no* linear test is pinned" won't always hold for a given mixture $W$. So I want a
recursive version: even if some test *is* pinned, I can carve off a single affine subspace $s$ that
(a) captures a non-negligible chunk of the mixture and (b) on which the conditional mixture *is*
near-uniform. That's the next step, by induction on $n$. If the no-test-pinned hypothesis holds, take
$s = \{0,1\}^n$ and the previous lemma gives near-uniformity directly. Otherwise, there is some
$a \neq 0$, $b$ with $\Pr_W[\forall y \in W: a\cdot y = b] > 2^{-r}$; let $u = \{x : a\cdot x = b\}$,
an $(n-1)$-dimensional affine subspace, so $\Pr_W[W \subseteq u] > 2^{-r}$. Condition on $W \subseteq
u$: now $W' = W \mid (W \subseteq u)$ is a random affine subspace *inside* $u$, and $u$ is a copy of
$\{0,1\}^{n-1}$. Recurse, with dimension $n-1$ and parameter $r - 1/2$ (I'll see in a moment why
$r-1/2$ is exactly the right decrement). The induction hands me an affine $s \subseteq u$ with
$\Pr_{W'}[W' \subseteq s] \ge 2^{-\sum_{i=0}^{(n-1)-\dim s - 1}(r - 1/2 - i/2)}$ and the conditional
mixture on $s$ within $2^{-((n-1)/2)\ldots}$ — let me track the second property's bound by what the
recursion preserves; it stays $2^{-(r-n/2)}$ if I set the decrements right.

Let me nail down property (a), the probability of capture, since the bookkeeping is the whole point.
Since $s \subseteq u$,
$$
\Pr_W[W \subseteq s] = \Pr_W[W \subseteq u]\cdot \Pr_{W}[W \subseteq s \mid W \subseteq u]
= \Pr_W[W \subseteq u]\cdot \Pr_{W'}[W' \subseteq s].
$$
Plug in $\Pr_W[W\subseteq u] > 2^{-r}$ and the inductive lower bound on $\Pr_{W'}[W'\subseteq s]$:
$$
\Pr_W[W\subseteq s] > 2^{-r}\cdot 2^{-\sum_{i=0}^{n - 1 - \dim s - 1}(r - 1/2 - i/2)}
= 2^{-\sum_{i=0}^{n - \dim s - 1}(r - i/2)},
$$
where I used that the recursion's parameter at depth one is $r$ shifted so that the leading $2^{-r}$
exactly fills the $i=0$ term of the sum and the inner sum's $i$-range shifts up by one. (This is why
the per-level decrement of the $r$-parameter is $1/2$ and the index runs over $i = 0,\dots,n-\dim s
-1$: the recursion turns one factor of $2^{-r}$ into the $i=0$ term, and the inner sum supplies
$i = 1,\dots$. The arithmetic closes.) The base case $n=0$ is trivial: the only affine subspace is
$\{\vec 0\}$, and $s = \{\vec 0\}$ works. And property (b), near-uniformity on $s$, is inherited
verbatim, because conditioning $W$ on $W \subseteq s$ is the same as conditioning $W'$ on $W'
\subseteq s$ (since $s \subseteq u$), so $\mathbb{E}_{W\mid W\subseteq s}[U_W] = \mathbb{E}_{W'\mid
W'\subseteq s}[U_{W'}]$, which the recursion already certified within $2^{-(r-n/2)}$ of $U_s$. So I
have: there exists an affine $s$ with $\Pr_W[W\subseteq s] \ge 2^{-\sum_{i=0}^{n-\dim s-1}(r-i/2)}$
and $\big|\mathbb{E}_{W\mid W\subseteq s}[U_W] - U_s\big|_1 < 2^{-(r-n/2)}$.

Now grouping. I repeatedly carve. Start with $W_0 = W$, carve off $s_0$ (the subspace the last result
guarantees), define the grouping map $\sigma(w) = s_0$ for every $w \subseteq s_0$. Then look at what's
left: $W_1 = W \mid (W \not\subseteq s_0)$, carve off $s_1$, set $\sigma(w) = s_1$ for the remaining
$w \subseteq s_1$ not already mapped. Continue: at step $i$, $W_i = W \mid (W\not\subseteq s_0)\wedge
\dots\wedge(W\not\subseteq s_{i-1})$ is the part of $A(n)$ still unmapped; carve $s_i$, extend $\sigma$.
The $s_i$ are all distinct (each $W_i$ avoids the previous $s$'s). Stop when the unmapped mass
$\Pr_W[W \notin \mathrm{dom}\,\sigma] \le 2^{-2n}$. This $\sigma$ groups every (all-but-$2^{-2n}$) affine
subspace into a representative $s = \sigma(w) \supseteq w$, with the conditional mixture on each group
near-uniform on its representative (that's exactly the per-carve near-uniformity). The one thing I must
control is *how many groups* there are of each dimension, because that controls the width blowup.

Counting groups of dimension $\ge k$: each carve that produces a representative $s_i$ of dimension
$\ge k$ accounts for, by the capture bound, at least a $2^{-\sum_{i=0}^{n-\dim s_i - 1}(r - i/2)} \ge
2^{-\sum_{i=0}^{n-k-1}(r-i/2)}$ fraction of the *remaining* mass (larger dimension means fewer terms
in the sum, hence a *larger* captured fraction; so dimension exactly the threshold $k$ is the worst
case, smallest captured fraction). After at most $4n \cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}$ such carves
the remaining mass is below $2^{-2n}$ and we stop. (The factor $4n$ is slack to drive the geometric
sum of captured fractions down past $2^{-2n}$.) So the number of dimension-$\ge k$ groups is at most
$4n\cdot 2^{\sum_{i=0}^{n-k-1}(r - i/2)}$. That is the grouping lemma, and it's the only thing from
this whole distributions-over-subspaces detour that I'll use downstream.

Now I can do the reduction properly: turn any general branching program $B$ into an *accurate affine*
branching program $P$, layer by layer. "Accurate" meaning: at the vertex $V_t$ the path reaches in
layer $t$, the conditional distribution of $x$ is $\epsilon$-close in $\ell_1$ to uniform on the label
$w(V_t)$. I build $P$ inductively, one layer at a time, with the invariant that layers $0,\dots,j$
form a sound affine program and the accuracy holds at layer $j$ — but I have to state the inductive
hypothesis in a slightly stronger form to keep the error *additive* rather than multiplicative,
because I have $m = 2^{\Theta(n)}$ layers and a multiplicative $(1+\delta)$ per layer would explode to
$(1+\delta)^{2^{\Theta(n)}}$. Disaster averted only by additivity.

The inductive hypothesis I carry: there is a random variable $U_j$ ranging over layer-$j$ vertices of
$P$ such that, with $y_j$ uniform over $w(U_j)$,
$$
\big|P_{V_j, x} - P_{U_j, y_j}\big|_1 \le \tfrac{\epsilon_j}{2}, \qquad \epsilon_j = 4j\cdot
2^{-(r - n/2)},
$$
where $P_{V_j,x}$ is the joint distribution of (the reached vertex, $x$) and $P_{U_j,y_j}$ is the
joint of ($U_j$, a fresh uniform-on-$w(U_j)$ point). This says the true (vertex, $x$) pair is close to
a (vertex, uniform-on-its-label) pair — i.e. $x$ is, jointly with where you are, close to uniform on
the labels. Base case $j=0$: label the start vertex $\{0,1\}^n$; $x$ is exactly uniform there, so
$U_0 = V_0$ and the distance is $0$. Now the inductive step, which is where grouping enters.

Suppose layers $0,\dots,j-1$ are built. Take the layer-$(j-1)$ random vertex $U_{j-1}$ from the
hypothesis, $y_{j-1}$ uniform on $w(U_{j-1})$. Draw a fresh uniform $a$, set $b = a\cdot y_{j-1}$
(the *honest* label as seen from the near-uniform surrogate), follow the edge $E = (U_{j-1}, V)$
labeled $(a,b)$ to a layer-$j$ vertex $V$ of $B$. The honest restricted subspace is $W = w(E) =
w(U_{j-1}) \cap \{x' : a\cdot x' = b\}$. For each fixed layer-$j$ vertex $v$ of $B$, consider $W_v =
W \mid (V = v)$, a random affine subspace, and apply the grouping lemma to it to get $\sigma_v$ (extend
$\sigma_v$ to output a special symbol $*$ where it was undefined — the $\le 2^{-2n}$ leftover). In $P$,
*split* $v$ into one copy $(v,s)$ per $s$ in the image of $\sigma_v$ (including the $*$-copy, labeled
$\{0,1\}^n$). Label $(v,s)$ by the affine subspace $s$ (or $\{0,1\}^n$ for $*$). The outgoing edges of
every copy $(v,s)$ are copies of $v$'s outgoing edges in $B$ (so functionality is preserved). The
incoming edges: for an edge $e = (u, v)$ labeled $(a,b)$ in $B$, compute $w = w(e) = w(u) \cap \{a\cdot
x' = b\}$ and route it to $(v, \sigma_v(w))$. Soundness holds: $(v, \sigma_v(w))$ is labeled
$\sigma_v(w) \supseteq w$ (or $\{0,1\}^n \supseteq w$ in the $*$ case), so $w(e) \subseteq w((v,
\sigma_v(w)))$. And the number of vertices created is controlled: by the grouping count, for each $B$-
vertex we make at most $4n\cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}$ copies of each dimension $k$.

To verify the inductive hypothesis at layer $j$, define $U_j = (V, \sigma_V(W))$ and $y_j$ uniform on
$w(U_j)$. I need $\big|P_{V_j,x} - P_{U_j,y_j}\big|_1 \le 2j\cdot 2^{-(r-n/2)}$. Introduce an
intermediate: $y'_j$ uniform on $W$ itself (the *honest* restricted subspace, before grouping). Two
pieces and the triangle inequality. First piece — the cost of grouping:
$$
\big|P_{U_j, y'_j} - P_{U_j, y_j}\big|_1 \le 2\cdot 2^{-(r-n/2)}.
$$
This is exactly what the grouping lemma's near-uniformity gives: for each $(v,s)$ with $s \neq *$, the
conditional mixture of $W$ given $V = v$ and $\sigma_v(W) = s$ is within $2^{-(r-n/2)}$ of $U_s$ — and
$P_{y'_j \mid U_j = (v,s)}$ is that conditional mixture while $P_{y_j \mid U_j=(v,s)} = U_s$; taking
expectation over $U_j$ and adding the $\le 2^{-2n}$ leftover mass from the $*$-copies, the gap is at
most $2^{-(r-n/2)} + 2^{-2n} \le 2\cdot 2^{-(r-n/2)}$. Second piece — the inductive error carried
forward:
$$
\big|P_{V_j, x} - P_{U_j, y'_j}\big|_1 \le 2(j-1)\cdot 2^{-(r-n/2)}.
$$
For this I use that the *same* probabilistic transformation maps the layer-$(j-1)$ pairs to the
layer-$j$ pairs. Define $T$ on (layer-$(j-1)$ vertex, point) pairs: given $(u, z)$, draw a fresh
uniform $a$, set $b = a\cdot z$, follow the edge labeled $(a,b)$ out of $u$ to a layer-$j$ vertex, and
output (that vertex, $z$). By the definition of the computation-path, $T(V_{j-1}, x)$ has the same
distribution as $(V_j, x)$ — feeding the true $x$ through one honest step is exactly one step of the
real process. And by the definitions of $U_j$, $y_j$, $y'_j$, the surrogate $T(U_{j-1}, y_{j-1})$ has
the same distribution as $(U_j, y'_j)$ — the surrogate uses $y_{j-1}$ in place of $x$ to generate the
honest restricted subspace $W$ and a uniform point $y'_j$ on it. A probabilistic transformation never
increases $\ell_1$ distance (it's an averaging of channels). So
$$
\big|P_{V_j,x} - P_{U_j, y'_j}\big|_1 = \big|P_{T(V_{j-1},x)} - P_{T(U_{j-1},y_{j-1})}\big|_1
\le \big|P_{V_{j-1},x} - P_{U_{j-1}, y_{j-1}}\big|_1 \le 2(j-1)\cdot 2^{-(r-n/2)},
$$
the last step being the inductive hypothesis (in the form $\epsilon_{j-1}/2 = 2(j-1)\cdot
2^{-(r-n/2)}$). Adding the two pieces, $\big|P_{V_j,x} - P_{U_j,y_j}\big|_1 \le 2j\cdot 2^{-(r-n/2)} =
\epsilon_j/2$, which is the hypothesis at layer $j$. And it confirms why I needed the funny
"$U_j$-surrogate" form: the transformation step requires comparing a *transported* honest pair to a
*transported* surrogate pair, and only the surrogate form is preserved by $T$ without picking up a
multiplicative factor. So errors add, layer by layer, and after $m$ layers the total is $\epsilon =
4m\cdot 2^{-(r-n/2)}$. The whole program $P$ is $\epsilon$-accurate (the accuracy definition follows
from the layer-$t$ hypothesis: with $z_t$ uniform on $w(V_t)$, $|P_{V_t,x} - P_{V_t,z_t}|_1 \le
|P_{V_t,x} - P_{U_t,y_t}|_1 + |P_{V_t,z_t} - P_{U_t,y_t}|_1$, and the second term equals $|P_{V_t} -
P_{U_t}|_1 \le \epsilon/2$ because $z_t, y_t$ are both uniform on the same labels, so accuracy $\le
\epsilon$).

Last, the output property of $P$. Let $V$ be the layer-$m$ vertex of $B$ reached, and $V_m = (V, S)$
the corresponding vertex of $P$ (since $P$ simulates $B$). $B$ outputs $w(V)$, of dimension $\le k'$,
with success $\Pr[x \in w(V)] = \beta$. Since $P$ is $\epsilon$-accurate, $x$ at $V_m$ is $\epsilon$-
close to $y_m$ uniform on $w(V_m)$, so $\Pr[y_m \in w(V)] \ge \Pr[x \in w(V)] - \epsilon = \beta -
\epsilon$. But $w(V)$ has dimension $\le k'$; if the $P$-label $w(V_m)$ has dimension $\ge k > k'$,
then a uniform point of $w(V_m)$ lands in $w(V)$ with conditional probability at most $2^{k' - k}$. So
$$
\beta - \epsilon \le \Pr[y_m \in w(V)] \le \Pr[\dim w(V_m) < k] + 2^{k' - k},
$$
giving $\Pr[\dim w(V_m) < k] \ge \beta - \epsilon - 2^{-(k - k')}$. In words: $P$ outputs a subspace
of dimension below $k$ with probability at least the success of $B$ minus the accuracy loss minus the
slack $2^{-(k-k')}$. So if $B$ succeeds with non-negligible probability, $P$ must reach a low-
dimensional vertex with comparable probability — and the affine reach bound says reaching a low-
dimensional vertex is $2^{-\Omega(n^2)}$-rare. Squeeze.

Time to assemble the constants, and this is delicate — the two halves have to cancel. Let $B$ have
length $m = 2^{\alpha n}$ and width $d = 2^{c n^2}$, output dimension $\le k' = \tfrac{3n}{5}$,
success $\beta$. Choose the carving parameter $r = \big(\tfrac12 + 2\alpha\big)n$ and the dimension
threshold $k = \tfrac{4n}{5}$. Then $\epsilon = 4m\cdot 2^{-(r-n/2)} = 4\cdot 2^{\alpha n}\cdot
2^{-2\alpha n} = 4\cdot 2^{-\alpha n}$, and the slack $2^{-(k-k')} = 2^{-(4n/5 - 3n/5)} = 2^{-n/5}$.
So $P$ reaches a dimension-$<k$ vertex with probability at least $\beta - 4\cdot 2^{-\alpha n} -
2^{-n/5} \ge \beta - 5\cdot 2^{-\alpha n}$. Make all dimension-$k$ vertices of $P$ leaves and drop
the unreachable ones, so every vertex of $P$ has $\dim \ge k$ and the dimension-$<k$ event is the
reach-a-dimension-$k$-vertex event. The number of dimension-$k$ vertices in $P$ is, from the grouping
count times the number of $B$-vertices per layer times layers, at most $4n\cdot 2^{\sum_{i=0}^{n-k-1}
(r-i/2)}\cdot d m$. The reach bound per vertex is $m^{n-k}\cdot 2^{\sum_{i=0}^{n-k-1}(n-2k-i)}$. So
the probability $P$ reaches *some* dimension-$k$ vertex is at most the product:
$$
\Big(4n\cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}\cdot d m\Big)\cdot \Big(m^{n-k}\cdot
2^{\sum_{i=0}^{n-k-1}(n-2k-i)}\Big).
$$
Substitute $r = (\tfrac12 + 2\alpha)n$, $k = \tfrac{4n}{5}$ so $n-k = \tfrac n5$ and $n - 2k = n -
\tfrac{8n}{5} = -\tfrac{3n}{5}$, $d = 2^{cn^2}$, $m = 2^{\alpha n}$ (so $m^{n-k} = 2^{\alpha n
(n-k)}$). Pulling the $4nm$ poly factor and $d = 2^{cn^2}$ out front:
$$
= 4nm\cdot 2^{cn^2}\cdot 2^{\sum_{i=0}^{n-k-1}\big(\frac12 n + 2\alpha n - \frac i2\big)}\cdot
2^{\alpha n(n-k)}\cdot 2^{\sum_{i=0}^{n-k-1}\big(-\frac35 n - i\big)}.
$$
Combine the first sum and the $m^{n-k}$ factor: $\sum_{i=0}^{n-k-1}(\tfrac12 n + 2\alpha n) =
(n-k)(\tfrac12 n + 2\alpha n)$, and $\alpha n(n-k)$, together $(n-k)(\tfrac12 n + 2\alpha n + \alpha
n) = (n-k)(\tfrac12 n + 3\alpha n)$. The $-\tfrac35 n$ part of the second sum: $\sum_{i=0}^{n-k-1}
(-\tfrac35 n) = -(n-k)\tfrac35 n$. So those linear-in-$n$ pieces give $(n-k)\big(\tfrac12 n + 3\alpha
n - \tfrac35 n\big) = (n-k)\big(3\alpha n - \tfrac{1}{10}n\big)$ — the $\tfrac12 - \tfrac35 =
-\tfrac{1}{10}$ is the crucial cancellation. The remaining $-\tfrac i2$ from the first sum and $-i$
from the second combine to $\sum_{i=0}^{n-k-1}\big(-\tfrac32 i\big) = -\tfrac32\cdot\tfrac{(n-k-1)(n-k)}
{2} = -\tfrac34 (n-k)(n-k-1)$. So
$$
= 4nm\cdot 2^{cn^2}\cdot 2^{(n-k)(3\alpha n - \frac{1}{10}n)}\cdot 2^{-\frac34 (n-k)(n-k-1)}.
$$
Now $n - k = \tfrac n5$: $(n-k)(3\alpha n - \tfrac1{10}n) = \tfrac n5(3\alpha n - \tfrac1{10}n)$, and
$\tfrac34 (n-k)(n-k-1) = \tfrac34\cdot\tfrac n5\big(\tfrac n5 - 1\big) = \tfrac n5\cdot\big(\tfrac34
\cdot\tfrac n5 - \tfrac34\big) = \tfrac n5\big(\tfrac{3}{20}n - \tfrac34\big)$. Collecting the $\tfrac
n5(\cdots)$:
$$
= 4nm\cdot 2^{cn^2}\cdot 2^{\frac15 n\left(3\alpha n - \frac{1}{10}n - \frac{3}{20}n + \frac34\right)}
= 4nm\cdot 2^{\,n^2\left(c + \frac35\alpha - \frac{1}{20} + \frac{3}{20 n}\right)}.
$$
Let me double-check that last collapse: $\tfrac15 n\cdot 3\alpha n = \tfrac35 \alpha n^2$; $\tfrac15 n
\cdot(-\tfrac1{10}n) = -\tfrac{1}{50}n^2$; $\tfrac15 n\cdot(-\tfrac{3}{20}n) = -\tfrac{3}{100}n^2$;
$-\tfrac1{50} - \tfrac{3}{100} = -\tfrac{2}{100} - \tfrac{3}{100} = -\tfrac{5}{100} = -\tfrac{1}{20}$,
so the $n^2$ coefficient is $c + \tfrac35\alpha - \tfrac1{20}$; and $\tfrac15 n\cdot\tfrac34 = \tfrac
{3}{20}n$, which is the $+\tfrac{3}{20n}\cdot n^2$ lower-order term. Confirmed.

So the probability of reaching a dimension-$k$ vertex is $4nm\cdot 2^{n^2(c + \frac35\alpha -
\frac1{20} + \frac{3}{20n})}$. The $4nm = 4n\cdot 2^{\alpha n}$ is sub-$2^{n^2\cdot o(1)}$, absorbed.
So if the $n^2$ coefficient is strictly negative — $c + \tfrac35\alpha - \tfrac1{20} < 0$, i.e.
$$
\alpha < \tfrac53\Big(\tfrac1{20} - c\Big),
$$
this probability is $2^{-\Omega(n^2)}$. Combine with the output property: $\beta - 5\cdot 2^{-\alpha
n} \le 2^{-\Omega(n^2)}$, so $\beta \le O(2^{-\alpha n})$. Done. Any branching program of length
$\le 2^{\alpha n}$, width $\le 2^{cn^2}$, outputting a dimension-$\le \tfrac{3n}{5}$ affine subspace,
succeeds with probability at most $O(2^{-\alpha n})$, for any $c < \tfrac1{20}$ and a suitable
$\alpha > 0$.

That's the formal theorem, and it implies the clean statement: a parity learner that uses fewer than
$c n^2$ memory bits and at most $2^{\alpha n}$ samples — width $2^{cn^2}$, length $2^{\alpha n}$,
outputting the single guess $\tilde x$ (a dimension-$0$ subspace, certainly dimension $\le \tfrac{3n}
{5}$) — has $\Pr[\tilde x = x] \le O(2^{-\alpha n})$. Picking $c < \tfrac1{20}$ and rounding to a
clean constant, any parity learner with fewer than about $\tfrac{n^2}{25}$ memory bits needs an
exponential number of samples. The two extreme algorithms I started with are the two corners this
forbids you from improving: Gaussian elimination pays the quadratic memory, candidate enumeration
pays the exponential samples, and *nothing* lives in between below quadratic memory and below
exponential samples.

Let me also note the cryptographic payoff, because it falls right out and it's the reason this isn't
only a curiosity. Two parties sharing a secret $x \in \{0,1\}^n$ can encrypt one bit $M$ as $M \oplus
(a\cdot x)$ for a fresh public uniform $a$; each use costs $n$ operations and reveals one sample
$(a, a\cdot x)$ to an eavesdropper. The theorem says an eavesdropper with fewer than $\tfrac{n^2}{25}$
memory bits, watching up to an exponential number of such uses, cannot reconstruct $x$ with better
than exponentially small probability — and since the inner product is a strong extractor, not knowing
$x$ means the next pad bit $a_{m+1}\cdot x$ is unpredictable beyond exponentially small advantage. So
this is a bounded-storage encryption scheme with key length $n$, encryption time $n$ per bit, secure
against memory-$\tfrac{n^2}{25}$ adversaries for exponentially many uses — and unlike the older
bounded-storage template, the encryption time isn't forced to be linear in the adversary's memory.

To recap the causal chain in one breath: parity's two algorithms sit at opposite resource corners, and
I wanted to know if the gap between them is forced; SQ and communication lower bounds couldn't reach it
because they constrain information *per sample*, while memory is cumulative and adaptive, and the
computation-tradeoff literature couldn't reach it because it stores the input for free; so I modeled
the learner as a branching program where width is memory and length is samples, observed that the
honest knowledge state is an affine subspace and that under a soundness invariant "learning" becomes
"shrinking the subspace's dimension"; I tracked the overlap between the learner's locked-in constraints
and the target's orthogonal space, showed it rises by at most one per step and each rise needs a fresh
random sample to hit an exponentially small coset, so reaching low dimension is $2^{-\Omega(n^2)}$-rare
for *affine* programs; then to handle *general* programs I showed the conditional distribution of $x$
at any state is a near-uniform mixture over affine subspaces — by Fourier analysis the inner product
acts as a strong extractor, so the mixture is uniform unless some linear test is pinned — grouped those
subspaces into few near-uniform representatives so the affine simulation has only $2^{O(n^2)}$ width,
and propagated the accuracy *additively* across the $2^{\Theta(n)}$ layers; finally I balanced the
threshold $k = \tfrac{4n}5$, the parameter $r = (\tfrac12 + 2\alpha)n$, and the budgets so the
grouping's vertex count and the affine reach bound cancel against the width, leaving $c < \tfrac1{20}$
and exponentially small success.

Theorem (formal). For any $c < \tfrac{1}{20}$ there exists $\alpha > 0$ such that: let $x$ be uniform
over $\{0,1\}^n$, let $m \le 2^{\alpha n}$, and let a branching program of length $m$ and width at most
$2^{cn^2}$ for parity learning, whose output is always an affine subspace of dimension $\le \tfrac{3n}
{5}$, be run on $x$ and honest samples $(a_t, b_t = a_t\cdot x)$ with each $a_t$ uniform over $\{0,1\}
^n$. Then the probability that $x$ lies in the output subspace is at most $O(2^{-\alpha n})$.

Corollary (the headline). For any $c < \tfrac{1}{20}$ there exists $\alpha > 0$ such that any parity-
learning algorithm using at most $cn^2$ bits of memory and at most $2^{\alpha n}$ samples outputs
$\tilde x$ with $\Pr[\tilde x = x] \le O(2^{-\alpha n})$. Equivalently: learning parity with fewer than
$\sim n^2/25$ bits of memory requires an exponential number of samples.

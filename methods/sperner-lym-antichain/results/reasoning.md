Schreier drops a question on me that sounds almost too plain to be interesting: take the subsets of an $n$-element set, and suppose you pick a bunch of them so that none is contained in any other — how many can you grab? My first instinct is that this is a one-line problem, and my second instinct, a few minutes in, is that I have no idea how to bound it. Let me think this through from scratch.

So I have $[n] = \{1, \dots, n\}$ and I'm choosing a family $\mathcal{F}$ of subsets with the rule: no member sits inside another member. Incomparable under $\subseteq$. The easiest such families are the obvious ones — all subsets of a fixed size $k$. Two different $k$-sets are never nested, so the whole layer $\binom{[n]}{k}$ is a legal family, of size $\binom{n}{k}$. And among all the layers, the biggest is the middle one, because the binomial coefficients go up and then come down. Let me actually pin *where* the peak is rather than wave at it: compare neighbors,
$$
\frac{\binom{n}{k}}{\binom{n}{k-1}} = \frac{n!/(k!(n-k)!)}{n!/((k-1)!(n-k+1)!)} = \frac{(k-1)!(n-k+1)!}{k!(n-k)!} = \frac{n-k+1}{k}.
$$
That ratio exceeds $1$ exactly when $n-k+1 > k$, i.e. $k < \tfrac{n+1}{2}$. So $\binom{n}{k}$ strictly increases while $k < (n+1)/2$ and strictly decreases after — a single hump, peaking at $k = \lfloor n/2 \rfloor$ (and tied with $\lceil n/2\rceil$ when $n$ is odd, since $\binom{n}{k} = \binom{n}{n-k}$). Good. So I have a family of size $\binom{n}{\lfloor n/2\rfloor}$ for free, and the real question is whether anything mixed — drawing from several layers at once — can ever beat that.

That's the crux, and it's not obvious it can't. Mixing *is* allowed: $\{\{1\},\{2,3\}\}$ is a perfectly good antichain in $[3]$. It's just small. Why small? Because the moment I put $\{2,3\}$ in, I'm barred from $\{2\}$, $\{3\}$, and from $\{1,2,3\}$ — every subset and every superset of it is now forbidden. So each set I admit "spends" not just itself but a whole upward-and-downward cone of the lattice. The hard part is accounting for that spending across many different sizes simultaneously. If I just bound each layer separately — $a_k \le \binom{n}{k}$ where $a_k$ is how many size-$k$ sets I took — I get $|\mathcal{F}| \le \sum_k \binom{n}{k} = 2^n$, which is laughably weak. It throws away the entire interaction between layers. I need an argument that *couples* the layers.

Let me try the most concrete thing I can think of: take any antichain and try to *shove* it into the middle layer without losing members, then read off the bound there. If I can transform any $\mathcal{F}$ into an antichain of the same (or larger) size that lives entirely in one middle layer, I'm done, because a single layer is at most $\binom{n}{\lfloor n/2\rfloor}$.

How to shove? Take the members of $\mathcal{F}$ that are too high — say they have size $k > n/2$ — and replace each by something one size smaller. The natural candidate is the shadow: replace the family $\mathcal{A}$ of these $k$-sets by $\partial \mathcal{A}$, the collection of all $(k-1)$-subsets of members of $\mathcal{A}$. For this move to be safe I need two things: that I don't lose count (the shadow is at least as big as $\mathcal{A}$), and that the result is still an antichain. Let me check the count, because that's where it could fail. Count incidences between the $m := |\mathcal{A}|$ sets of size $k$ and their $(k-1)$-subsets. Each $k$-set has exactly $k$ subsets of size $k-1$, so there are $m \cdot k$ incidences. On the other side, a fixed $(k-1)$-set can be a subset of at most $n - (k-1) = n-k+1$ different $k$-sets (you complete it by adding one of the remaining $n-k+1$ elements). So if $r := |\partial \mathcal{A}|$ is the number of distinct shadow sets,
$$
m \cdot k \ \le\ r \cdot (n-k+1) \quad\Longrightarrow\quad r \ \ge\ \frac{m\,k}{\,n-k+1\,}.
$$
Now $k > n/2$ — more precisely if $k > (n+1)/2$ then $k > n-k+1$, so $\frac{k}{n-k+1} > 1$, and $r \ge \frac{mk}{n-k+1} > m$, i.e. $r \ge m+1$. The shadow is *strictly* bigger. So replacing the top layer by its shadow doesn't shrink the family. And it stays an antichain after I clean up overlaps with the layers below — I have to be a bit careful, because the new $(k-1)$-sets must not be subsets of anything already present, but since the original family was an antichain and I'm moving the *highest* sets down by one, with a little case-checking the incomparability survives.

This works. I can iterate: keep pulling the top layer down via shadows, and dually push the bottom layer up via the shade (the same incidence count by complementation: $m$ sets of size $k < n/2$ have at least $m+1$ supersets of size $k+1$), and after finitely many steps every set sits in the middle. Each step didn't decrease the size; the final family is inside one middle layer, hence $\le \binom{n}{\lfloor n/2\rfloor}$; therefore the original family was too.

But — let me actually try to write this down cleanly and I immediately feel the weight of it. The shadow inequality $r \ge mk/(n-k+1)$ gives strict growth only for $k > (n+1)/2$; right at the boundary $k = (n+1)/2$ (odd $n$) it gives $r \ge m$ with equality possible, so I need a separate, finer argument to rule out the bad equality case there — and that argument is genuinely fiddly: I have to show that if $m < \binom{n}{\lfloor n/2\rfloor}$ then even at the boundary the shadow strictly grows, which involves producing an extra $(k-1)$-set not already accounted for. Then the whole thing is an induction over the number of replacement steps, at each step re-verifying that the family is still an antichain. It all goes through, but it's a *lot* of moving parts for a single number, and at the end I have only the inequality $|\mathcal{F}| \le \binom{n}{\lfloor n/2\rfloor}$ — no extra structure handed to me, and the equality cases still need to be dug out by hand. I have a proof, but it feels like I'm carrying a piano up the stairs.

Let me back up and stare at what I actually want to bound. I want to control $|\mathcal{F}| = \sum_k a_k$ where $a_k$ is the number of size-$k$ members. The shadow approach moves mass between layers. But maybe I shouldn't move anything at all. Maybe I should *weight* the layers and bound a weighted sum directly. The trouble with $|\mathcal{F}| \le \sum a_k$ is that a set in the fat middle layer "costs the same" as a set in a thin outer layer, even though there are wildly different numbers of them available. What if I weight each set so that a full layer always has total weight $1$? Then admitting one $k$-set costs $1/\binom{n}{k}$, and a full layer costs exactly $1$. If I could show the total weight of any antichain is at most $1$, I'd have $\sum_k a_k / \binom{n}{k} \le 1$, and then since the middle layer has the largest $\binom{n}{k}$, every weight $1/\binom{n}{k} \ge 1/\binom{n}{\lfloor n/2\rfloor}$, so
$$
\frac{|\mathcal{F}|}{\binom{n}{\lfloor n/2\rfloor}} = \sum_k \frac{a_k}{\binom{n}{\lfloor n/2\rfloor}} \ \le\ \sum_k \frac{a_k}{\binom{n}{k}} \ \le\ 1,
$$
giving $|\mathcal{F}| \le \binom{n}{\lfloor n/2\rfloor}$ in one line. So the whole game reduces to proving the weighted inequality $\sum_k a_k / \binom{n}{k} \le 1$. That's a cleaner target — and notice it's *stronger* than what I need, which usually means it's secretly easier, because there's more structure to grab onto.

Now, $1/\binom{n}{k}$ — where does a quantity like that come from naturally? $\binom{n}{k} = n!/(k!(n-k)!)$, so $1/\binom{n}{k} = \frac{k!(n-k)!}{n!}$. That has the flavor of a *probability*: a count of favorable arrangements over $n!$ total arrangements. $n!$ is the number of permutations of $[n]$. So I should be thinking about permutations. And what is $k!(n-k)!$? It's "arrange the first $k$ things among themselves, arrange the last $n-k$ things among themselves." That's the count of orderings of $[n]$ in which some particular $k$-element block comes first, in any internal order, followed by the rest, in any internal order.

Here's the picture that pulls it together. Read a permutation of $[n]$ as an instruction for building up the whole set one element at a time: start from $\emptyset$, add the first element, then the second, and so on, ending at $[n]$. That traces out a *maximal tower* of subsets
$$
\emptyset = C_0 \subset C_1 \subset C_2 \subset \cdots \subset C_n = [n], \qquad |C_i| = i,
$$
where each $C_i$ is one element bigger than $C_{i-1}$. A saturated, top-to-bottom chain through the lattice. Each permutation gives exactly one such chain, and conversely each such chain corresponds to exactly one permutation (the order in which the elements got added). So there are exactly $n!$ maximal chains.

Now ask: how many of these maximal chains pass through a *fixed* set $A$ of size $k$? To pass through $A$, the chain must build $A$ first — adding the $k$ elements of $A$ one at a time, which can be done in $k!$ orders — and then continue from $A$ up to $[n]$ by adding the remaining $n-k$ elements, in $(n-k)!$ orders. The two halves are independent, so the number of maximal chains through $A$ is exactly $k!\,(n-k)!$. There it is — that's the numerator I wanted. The number of chains through $A$, divided by the total number of chains, is $\frac{k!(n-k)!}{n!} = 1/\binom{n}{k}$. The weight $1/\binom{n}{k}$ is *literally the fraction of maximal chains that pass through a $k$-set*.

And now the antichain condition does exactly one thing, but it's the whole ballgame: a maximal chain can pass through at most one member of $\mathcal{F}$, because any two sets that both lie on a single chain $C_0 \subset \cdots \subset C_n$ are nested — one is some $C_i$ and the other some $C_j$ with, say, $i < j$, so $C_i \subset C_j$ — comparable, hence forbidden in an antichain. So the sets of $\mathcal{F}$ carve up the maximal chains into disjoint groups (the chains through $A$, for each $A \in \mathcal{F}$), with no chain counted twice. The groups are disjoint subsets of all $n!$ chains, so their sizes can't add up to more than $n!$:
$$
\sum_{A \in \mathcal{F}} k_A!\,(n-k_A)! \ \le\ n!, \qquad k_A := |A|.
$$
Group by size and write $a_k$ for the number of members of size $k$:
$$
\sum_{k=0}^{n} a_k\, k!\,(n-k)! \ \le\ n!.
$$
Divide both sides by $n!$:
$$
\boxed{\ \sum_{k=0}^{n} \frac{a_k}{\binom{n}{k}} \ \le\ 1.\ }
$$
That's the weighted inequality I was after, and it fell out of a single double count — count maximal chains one way ($n!$ of them) and another way (group by which antichain member they hit). No shadow lemma, no boundary case, no induction over replacement steps. The entire weight of the shadow/compression argument collapses into "two sets on one chain are comparable." I should double-check the bookkeeping with the probabilistic reading, because it's a good sanity check: pick a maximal chain uniformly at random (equivalently a random permutation); the probability it passes through a given $k$-set $A$ is $k!(n-k)!/n! = 1/\binom{n}{k}$; let $X$ be the number of members of $\mathcal{F}$ the random chain hits. Then $\mathbb{E}[X] = \sum_{A} 1/\binom{n}{k_A} = \sum_k a_k/\binom{n}{k}$ by linearity. But $X \le 1$ always, since a chain hits at most one antichain member. An expectation of a quantity that never exceeds $1$ is itself at most $1$. Same inequality, and it confirms I haven't dropped a factor.

Now finish. Each $\binom{n}{k} \le \binom{n}{\lfloor n/2\rfloor}$ — that's exactly the unimodality I checked at the very start, the peak at the middle. So $1/\binom{n}{k} \ge 1/\binom{n}{\lfloor n/2\rfloor}$ for every $k$, and
$$
\frac{|\mathcal{F}|}{\binom{n}{\lfloor n/2\rfloor}} = \sum_{k} \frac{a_k}{\binom{n}{\lfloor n/2\rfloor}} \ \le\ \sum_{k}\frac{a_k}{\binom{n}{k}} \ \le\ 1
\quad\Longrightarrow\quad |\mathcal{F}| \ \le\ \binom{n}{\lfloor n/2\rfloor}.
$$
And the middle layer achieves it, so the maximum antichain size is *exactly* $\binom{n}{\lfloor n/2\rfloor}$.

The weighted inequality also hands me the equality cases for free — which the compression argument made me dig for. Trace where the two inequalities are tight. The last step, $\sum_k a_k/\binom{n}{k} \le \sum_k a_k/\binom{n}{\lfloor n/2\rfloor}$, is an equality only if $a_k = 0$ for every $k$ with $\binom{n}{k} < \binom{n}{\lfloor n/2\rfloor}$ — i.e. $\mathcal{F}$ uses *only* layers that achieve the maximum binomial coefficient. For even $n$ that's the single middle layer $k = n/2$; for odd $n$ it's the two middle layers $k = \tfrac{n-1}{2}$ and $k = \tfrac{n+1}{2}$. The first inequality, $\sum_A k_A!(n-k_A)! \le n!$, is an equality only if the chains-through-members groups *exhaust* all $n!$ maximal chains — every maximal chain hits a member of $\mathcal{F}$. For even $n$, a family living only in layer $n/2$ that meets every maximal chain must be the *entire* layer $\binom{[n]}{n/2}$ (if even one $(n/2)$-set were missing, the chains through it would miss $\mathcal{F}$). So for even $n$ the unique extremal antichain is the full middle layer. For odd $n$ it's more delicate: $\mathcal{F}$ lives in layers $\tfrac{n-1}{2}$ and $\tfrac{n+1}{2}$ and must hit every maximal chain exactly once; checking which such mixed families cover all chains without two members sharing a chain forces $\mathcal{F}$ to be one *entire* middle layer — either all of $\binom{[n]}{(n-1)/2}$ or all of $\binom{[n]}{(n+1)/2}$ — because a chain passes through exactly one set of each size, so covering every chain by sets drawn from two adjacent layers, with at most one per chain, leaves no room to split between the layers without either a gap or a collision. So the extremal families are: the full middle layer for even $n$; one of the two full middle layers for odd $n$. Exactly the construction I started with, now proven to be the *only* optima.

Let me make sure the small cases agree, since I claimed them as the yardstick. $n=2$: $\binom{2}{1} = 2$ — yes, $\{\{1\},\{2\}\}$, and nothing of size $2$ works. $n=3$: $\binom{3}{1} = 3$, either all singletons or all pairs, and the two layers $k=1,2$ are the two tied middle layers — matches the odd-$n$ "one of two middle layers." $n=4$: $\binom{4}{2} = 6$, the six $2$-sets. All consistent.

Stepping back at the whole chain of it: I wanted the largest antichain in the subset lattice; the obvious construction is the middle layer of size $\binom{n}{\lfloor n/2\rfloor}$; to prove nothing mixed beats it I first tried to compress every antichain into the middle by shadows, which works but drags a shadow-size lemma, a boundary case, and an induction; backing off, I reweighted so each full layer costs $1$, turning the goal into $\sum_k a_k/\binom{n}{k} \le 1$; recognizing $1/\binom{n}{k} = k!(n-k)!/n!$ as the fraction of maximal chains (= permutations) through a $k$-set, I counted maximal chains two ways, used the single fact that an antichain meets each chain at most once, and the inequality dropped out; bounding each denominator by the middle one gave Sperner's bound, and reading off where the inequalities are tight gave the exact extremal families. Here is the finished statement and proof, plus a tiny check.

```python
# The theorem and a brute-force sanity check on small n.
# Claim: the largest antichain in the subset lattice 2^[n] has size C(n, floor(n/2)),
#        and the LYM inequality sum_k a_k / C(n,k) <= 1 holds for every antichain.
from itertools import combinations
from math import comb, factorial

def chains_through_set_fraction(n, k):
    # fraction of maximal chains (= permutations of [n]) that pass through a fixed k-set:
    # k!*(n-k)! chains through it, out of n! total  ==  1 / C(n,k).
    return factorial(k) * factorial(n - k) / factorial(n)   # == 1 / comb(n, k)

def lym_lhs(n, antichain):
    # sum over members of 1 / C(n, |A|); the LYM inequality says this is <= 1.
    return sum(1 / comb(n, len(A)) for A in antichain)

def is_antichain(family):
    fam = list(map(set, family))
    return all(not (fam[i] <= fam[j]) for i in range(len(fam))
               for j in range(len(fam)) if i != j)

def brute_force_max_antichain(n):
    subs = [frozenset(s) for k in range(n + 1) for s in combinations(range(n), k)]
    from itertools import combinations as C
    best = 0
    for r in range(len(subs) + 1):
        for fam in C(subs, r):
            if is_antichain(fam):
                best = max(best, r)
    return best

if __name__ == "__main__":
    for n in range(5):                                   # exhaustive ground truth, small n
        assert brute_force_max_antichain(n) == comb(n, n // 2)
        # the full middle layer is an antichain whose LYM sum is exactly 1:
        middle = [frozenset(s) for s in combinations(range(n), n // 2)]
        assert abs(lym_lhs(n, middle) - 1.0) < 1e-9
        # weight of a k-set is exactly the fraction of maximal chains through it:
        for k in range(n + 1):
            assert abs(chains_through_set_fraction(n, k) - 1 / comb(n, k)) < 1e-12
    print("max antichain size = C(n, floor(n/2)); LYM weight of a full middle layer = 1.")
```

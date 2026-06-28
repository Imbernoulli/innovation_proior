I need to answer edge queries one at a time, with no future order visible, and I have to keep the final connectivity unresolved until the last pair is asked. The clean way to think about a prefix is not to guess the final graph. I have three sets of pairs: the pairs I have already answered yes to, the pairs I have already answered no to, and the pairs nobody has asked yet. Let $G$ be the graph containing only the confirmed yes edges. Let $H$ be the graph containing the confirmed yes edges plus every unasked pair, as if every remaining answer could still be yes.

Now the two ways I can lose are exact. If $G$ is connected, then the graph is certainly connected no matter how I answer later. If $H$ is disconnected, then even the most generous completion cannot connect the cities, so the graph is certainly disconnected. Therefore the whole interaction is asking me to maintain, for every non-final prefix, $G$ disconnected and $H$ connected. At the start this is true: $G$ has no edges, and $H$ is complete.

For one queried pair $e=(u,v)$, saying no removes $e$ from $H$ and leaves $G$ alone. So no is safe unless $e$ is the last remaining connection across some cut of $H$, a bridge. Saying yes keeps $e$ inside $H$ and adds it to $G$. So I am pushed toward a rule of the shape: say no whenever I can, and say yes only when no would disconnect $H$, that is, only when the queried edge is a bridge of the optimistic graph $H$.

That is conceptually clean but too slow if I test it literally. A bridge test asks whether $u$ can still reach $v$ in $H-e$, which means a graph search over a dense graph. Doing that for $\Theta(n^2)$ queries is far too much. I need the bridge condition in a form that can be updated.

Suppose I look at the connected components of $G$. If a queried edge goes between two current components $P$ and $Q$, then it is forced yes exactly when it is the last unasked pair crossing between $P$ and $Q$. If there is another unasked crossing pair, I can answer no and $H$ stays connected across that component cut. If this is the last one, answering no separates the two sides in $H$, so I must answer yes and merge the two components in $G$.

So a direct implementation keeps, for every pair of current $G$-components, a count $S(P,Q)$ of unasked crossing pairs. Initially every component is a singleton and $S(i,j)=1$. On a query between different components, if the count is bigger than one, I answer no and decrement it. If the count is one, I answer yes and merge the components; for any other component $X$, the new crossing count is the sum $S(P,X)+S(Q,X)$. There are at most $n-1$ yes answers because each yes merges two components, and each merge costs one pass over the components, so this already gives an $O(n^2)$ total strategy.

But the counts feel more rigid than this bookkeeping admits. Every yes edge only merges two components and never closes a cycle, so the accumulated yes graph is acyclic at all times, and the final yes graph will be a tree on all $n$ vertices. If the final tree is forced anyway, maybe I can fix it in advance and avoid the component matrix entirely. The cheapest spanning tree to describe is the one where every vertex $w>0$ has exactly one edge going to a strictly smaller label: following those edges strictly decreases the label until it reaches $0$, so all vertices reach $0$, and a cycle is impossible because its largest vertex would need two down-edges. So if I can arrange that each $w$ ends up with exactly one yes edge to a smaller label, the final yes graph is automatically a spanning tree.

The pairs split naturally by their larger endpoint. For each $w$, the owned set is

$$
E_w = \{(w,0),(w,1),\dots,(w,w-1)\},
$$

and it has exactly $w$ pairs. So "each $w$ gets exactly one down-edge" means: answer yes to exactly one pair of each $E_w$, no to the other $w-1$. That guarantees the final tree no matter which pair I pick. But which one? I have a free choice here, and it matters, because I still have to keep $G$ disconnected at every non-final prefix.

The obvious greedy choice is to say yes to the first pair of $E_w$ that gets queried, committing the edge as early as possible and saying no afterward. Let me see whether that keeps $G$ disconnected. Take $n=4$ and the query order $(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)$. With the first-pair rule, $(0,1)$ is the first of $E_1$ so yes; $(0,2)$ is the first of $E_2$ so yes; $(0,3)$ is the first of $E_3$ so yes. After only three of the six answers, $G=\{(0,1),(0,2),(0,3)\}$, which is a star on all four vertices and already connected. That violates the invariant: the questioner now knows the graph is connected with three queries still outstanding. So committing early is exactly wrong. The early yes edges pile up at vertex $0$ and connect everything before the run is over.

The failure points at the cure. To keep $G$ short, I want every yes edge to come as late as possible, so I should answer yes to the *last* queried pair of each $E_w$ instead of the first. Then the single yes of $E_w$ is deferred to the moment its set is exhausted. I cannot know in advance which pair of $E_w$ will be last, but I can recognize it when it arrives: keep a counter $c[w]$ of how many pairs from $E_w$ have been queried. The last one is precisely the moment after the increment when $c[w]=w$.

So the online rule collapses to this. For a query $(u,v)$, set $w=\max(u,v)$, increment $c[w]$, and answer yes exactly when the counter reaches $w$. As a single-file program I read the sample grader's input from stdin -- the first line is $n$, then $r=n(n-1)/2$ lines each give a queried pair $u\ v$ -- and print one answer per line, $1$ for a claimed flight and $0$ otherwise.

The last-query tree strategy, where the lazy spanning-tree adversary answers yes only on the last pair in each larger-endpoint owned set $E_w$, is the part I would most easily get wrong under time pressure; if I were not confident I could implement it correctly in the budget, I would fall back to the component-count bridge strategy with $S(P,Q)$ counts between current $G$-components that I have already traced as correct and ship that -- a plain correct submission beats an ambitious broken one.

```cpp
// IOI 2014 "Game": lazy spanning-tree adversary.
// Reads from stdin: line 1 is n; then r = n(n-1)/2 lines, each "u v".
// For each query prints one line, 1 if a direct flight is claimed, else 0.
#include <cstdio>

static int c[1500];

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    for (int i = 0; i < n; ++i) c[i] = 0;
    long long r = (long long)n * (n - 1) / 2;
    for (long long q = 0; q < r; ++q) {
        int u, v;
        if (scanf("%d %d", &u, &v) != 2) break;
        int w = u > v ? u : v;          // owner = larger endpoint
        int ans = (++c[w] == w) ? 1 : 0; // yes only on the last query owned by w
        printf("%d\n", ans);
    }
    return 0;
}
```

Before trusting the proof, I want to watch this rule run on the same $n=4$ instance, but with a friendlier order $(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)$, and track both graphs by hand. Query $(0,1)$: $w=1$, $c[1]\to1=w$, yes, $G=\{(0,1)\}$. Query $(0,2)$: $w=2$, $c[2]\to1\ne2$, no. Query $(1,2)$: $w=2$, $c[2]\to2=w$, yes, $G=\{(0,1),(1,2)\}$. Query $(0,3)$: $w=3$, $c[3]\to1$, no. Query $(1,3)$: $w=3$, $c[3]\to2$, no. Query $(2,3)$: $w=3$, $c[3]\to3=w$, yes, $G=\{(0,1),(1,2),(2,3)\}$. So the confirmed graph stays as the path components $\{0,1\},\{2\},\{3\}$, then $\{0,1,2\},\{3\}$, and only becomes connected at the final answer. Counting $G$-components across the six prefixes I get $3,3,2,2,2,1$: it stays at least $2$ until the very last query and drops to $1$ exactly there. Meanwhile $H$, the confirmed-yes plus unasked pairs, is connected at every prefix; for instance after the third answer the unasked pairs $(0,3),(1,3),(2,3)$ alone already span all four vertices. That is the invariant in front of me on a concrete run.

One example with a convenient order is not a proof, and I am worried that a hostile order could trip it. So I checked the rule against all $720$ query orders of the $n=4$ instance, recomputing the components of $G$ and of $H$ after every prefix. In every order, $G$ is disconnected after each of the first five answers and connected only after the sixth, and $H$ is connected after every prefix. No order breaks it. The first-pair variant, run through the same check, fails as soon as a star forms at $0$, which matches the hand example above. That is enough to make me trust the general argument and write it out.

So I check the invariant directly on the final rule. Over the full set of queries, each $w=1,\dots,n-1$ contributes exactly one yes edge, and that edge joins $w$ to a smaller label. Call the set of all these final yes edges $T$. Every vertex $w>0$ has a parent of smaller label, so repeatedly following parents reaches $0$; this gives connectivity. There is no cycle either: in any cycle, take the largest labelled vertex $w$ on it. Both of its cycle neighbors have smaller labels, so the two cycle edges incident to $w$ would both be owned by $w$, but $w$ contributes only one yes edge. Thus $T$ is a spanning tree, with exactly $n-1$ edges, matching the component count I watched: $n$ singletons reduced to $1$ by $n-1$ merges.

Now take any prefix before the last query. Let $G$ be the yes edges already answered, and let $H$ be those yes edges plus every unasked pair. The actual last query of the run belongs to some owned set $E_w$, and because it is the last remaining query in that set at the moment it is asked, the rule answers yes on it. Therefore every earlier prefix is missing at least that final yes edge of the final tree $T$. All confirmed yes edges are edges of $T$, so $G$ is a spanning graph on the same $n$ vertices with at most $n-2$ edges. A connected graph on $n$ vertices needs at least $n-1$ edges, so $G$ is disconnected. On the other hand, every edge of $T$ is present in $H$: an edge of $T$ that has already been asked is a confirmed yes edge, and an edge of $T$ that has not yet been asked is still an unasked pair. Since $H$ contains the spanning tree $T$, $H$ is connected. This is exactly the ambiguity invariant for every non-final prefix, and it is the same behavior the $n=4$ enumeration showed.

At the final query, the last missing tree edge is answered yes, so the confirmed graph becomes $T$, a connected tree. Just before that answer, the confirmed graph is $T$ minus one edge and is disconnected. The questioner gets the answer only at the last possible moment, independent of the query order. The cost is one counter increment and one comparison per question, plus resetting $n$ counters at the start, so the total time is $O(n^2)$ because there are $\binom{n}{2}$ questions, and the memory is $O(n)$.

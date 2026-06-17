I need to answer edge queries one at a time, with no future order visible, and I have to keep the final connectivity unresolved until the last pair is asked. The clean way to think about a prefix is not to guess the final graph. I have three sets of pairs: the pairs I have already answered yes to, the pairs I have already answered no to, and the pairs nobody has asked yet. Let $G$ be the graph containing only the confirmed yes edges. Let $H$ be the graph containing the confirmed yes edges plus every unasked pair, as if every remaining answer could still be yes.

Now the two ways I can lose are exact. If $G$ is connected, then the graph is certainly connected no matter how I answer later. If $H$ is disconnected, then even the most generous completion cannot connect the cities, so the graph is certainly disconnected. Therefore the whole interaction is asking me to maintain, for every non-final prefix, $G$ disconnected and $H$ connected. At the start this is true: $G$ has no edges, and $H$ is complete.

For one queried pair $e=(u,v)$, saying no removes $e$ from $H$ and leaves $G$ alone. So no is safe unless $e$ is the last remaining connection across some cut of $H$, a bridge. Saying yes keeps $e$ inside $H$ and adds it to $G$. So yes is what I should do only when no would disconnect $H$. This gives the first version of the strategy: say no unless the queried edge is a bridge of the optimistic graph $H$.

That is conceptually right but too slow if I test it literally. A bridge test asks whether $u$ can still reach $v$ in $H-e$, which means a graph search over a dense graph. Doing that for $\Theta(n^2)$ queries is far too much. I need the bridge condition in a form that can be updated.

Suppose I look at the connected components of $G$. If a queried edge goes between two current components $P$ and $Q$, then it is forced yes exactly when it is the last unasked pair crossing between $P$ and $Q$. If there is another unasked crossing pair, I can answer no and $H$ stays connected across that component cut. If this is the last one, answering no separates the two sides in $H$, so I must answer yes and merge the two components in $G$.

So a direct implementation keeps, for every pair of current $G$-components, a count $S(P,Q)$ of unasked crossing pairs. Initially every component is a singleton and $S(i,j)=1$. On a query between different components, if the count is bigger than one, I answer no and decrement it. If the count is one, I answer yes and merge the components; for any other component $X$, the new crossing count is the sum $S(P,X)+S(Q,X)$. There are at most $n-1$ yes answers because each yes merges two components, and each merge costs one pass over the components, so this already gives an $O(n^2)$ total strategy.

But the counts are doing something more rigid than they first seem. Every yes edge only merges components, so the final yes graph will be a tree. If I can decide the tree structure without dynamic component bookkeeping, I can avoid the whole matrix. A rooted tree appears for free if every vertex $w>0$ chooses exactly one edge to a smaller vertex: following those chosen edges strictly decreases the label until it reaches $0$, so all vertices connect to $0$ and no cycle is possible.

The question then becomes how to choose exactly one smaller-neighbor edge for each $w$ while still answering online. The pairs split naturally by their larger endpoint. For each $w$, the owned set is

$$
E_w = \{(w,0),(w,1),\dots,(w,w-1)\},
$$

and it has exactly $w$ pairs. If I answer yes on the last queried pair in each $E_w$, then every $w>0$ gets exactly one edge to a smaller label. I cannot know in advance which pair in $E_w$ will be last, but I can recognize it when it arrives: keep a counter $c[w]$ of how many pairs from $E_w$ have been queried. The last one is precisely the moment after the increment when $c[w]=w$.

So the online rule collapses to this. For a query $(u,v)$, set $w=\max(u,v)$, increment $c[w]$, and answer yes exactly when the counter reaches $w$.

```c
#include "game.h"

int c[1500];

void initialize(int n) {
    int i;
    for (i = 0; i < n; ++i) c[i] = 0;
}

int hasEdge(int u, int v) {
    int w = u > v ? u : v;
    return ++c[w] == w;
}
```

I should check the invariant directly on this final rule, because the strategy has become so small that the proof should not depend on the heavier component story. Over the full set of queries, each $w=1,\dots,n-1$ contributes exactly one yes edge, and that edge joins $w$ to a smaller label. Call the set of all these final yes edges $T$. Every vertex $w>0$ has a parent of smaller label, so repeatedly following parents reaches $0$; this gives connectivity. There is no cycle either: in any cycle, take the largest labelled vertex $w$ on it. Both of its cycle neighbors have smaller labels, so the two cycle edges incident to $w$ would both be owned by $w$, but $w$ contributes only one yes edge. Thus $T$ is a spanning tree.

Now take any prefix before the last query. Let $G$ be the yes edges already answered, and let $H$ be those yes edges plus every unasked pair. The actual last query of the run belongs to some owned set $E_w$, and because it is the last remaining query in that set at the moment it is asked, the rule answers yes on it. Therefore every earlier prefix is missing at least that final yes edge of the final tree $T$. All confirmed yes edges are edges of $T$, so $G$ is a spanning graph on the same $n$ vertices with at most $n-2$ edges. A connected graph on $n$ vertices needs at least $n-1$ edges, so $G$ is disconnected. On the other hand, every edge of $T$ is present in $H$: an edge of $T$ that has already been asked is a confirmed yes edge, and an edge of $T$ that has not yet been asked is still an unasked pair. Since $H$ contains the spanning tree $T$, $H$ is connected. This is exactly the ambiguity invariant for every non-final prefix.

At the final query, the last missing tree edge is answered yes, so the confirmed graph becomes $T$, a connected tree. Just before that answer, the confirmed graph is $T$ minus one edge and is disconnected. The questioner gets the answer only at the last possible moment, independent of the query order. The cost is one counter increment and one comparison per question, plus resetting $n$ counters at the start, so the total time is $O(n^2)$ because there are $\binom{n}{2}$ questions, and the memory is $O(n)$.

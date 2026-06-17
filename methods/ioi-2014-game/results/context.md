## Problem

There are $n$ cities numbered $0,1,\dots,n-1$. Some unordered pairs of cities are joined by undirected flights, and the question is whether the resulting graph is connected: whether every city can reach every other city through zero or more flights.

The graph is not fixed in advance. A questioner asks about each unordered pair of distinct cities exactly once, in an order chosen by the questioner. Each question has the form "is there a direct flight between cities $u$ and $v$?" The responder must answer immediately with $1$ for yes or $0$ for no, before seeing later questions. After all answers, the answers define one graph.

There are exactly

$$
r = \binom{n}{2} = \frac{n(n-1)}{2}
$$

questions. The questioner wins if, after the first $i$ answers for some $i < r$, the connectivity of every graph consistent with those answers is already the same. That is, either every completion is connected or every completion is disconnected. The responder wins if this never happens early: after every prefix of fewer than $r$ answers, at least one connected completion and at least one disconnected completion are still possible.

Implement the responder through two callbacks:

```c
void initialize(int n);
int hasEdge(int u, int v);
```

`initialize(n)` is called before the questions. Then `hasEdge(u, v)` is called exactly once for each unordered pair, with $u \ne v$, in the questioner's order. The return value of `hasEdge` is the responder's answer for that pair. The constraints include $4 \le n \le 1500$, so the responder should spend only constant or near-constant time per question and $O(n)$ or $O(n^2)$ total auxiliary work.

## Code framework

```c
#include "game.h"

int c[1500];

void initialize(int n) {
    /* TODO */
}

int hasEdge(int u, int v) {
    /* TODO */
}
```

# Grasshopper Mine-Avoidance

Let the current jump lengths be
$$0<a_1<a_2<\cdots<a_N,$$
with total $S=a_1+\cdots+a_N$. Call a mine active if it lies in $(0,S-a_1]$.

We prove the following strengthened claim by induction on $N$:

> If $S$ is not mined and at most $N-1$ mines are active, then the jumps can be ordered so that no landing point is mined.

This implies the stated result. In any order, every non-final landing point is at most $S-a_1$, since at least the smallest jump length remains unused before the final landing. Mines outside $(0,S-a_1]$ cannot be hit before the end, and $S$ itself is not mined.

For $N=1$, the active interval is empty and the single jump lands at $S$.

Assume the strengthened claim for smaller values of $N$. Set
$$x=S-a_N.$$
We split according to whether $x$ is mined and whether an active mine lies strictly to the right of $x$.

If $x$ is not mined and some active mine lies to the right of $x$, then at most $N-2$ active mines lie at or below $x$. In particular, at most $N-2$ mines lie in the recursive active interval $(0,x-a_1]$ for the lengths $a_1,\ldots,a_{N-1}$. By induction these lengths can be ordered to reach $x$ safely. Appending $a_N$ lands at $S$ and jumps over the active mine or mines to the right of $x$.

If $x$ is not mined and no active mine lies to the right of $x$, there is nothing to prove when there are no active mines. Otherwise let $m$ be the largest active mine, and temporarily remove it. The recursive problem on $a_1,\ldots,a_{N-1}$ now has at most $N-2$ relevant mines in $(0,x-a_1]$ and a safe endpoint $x$, so induction gives a path to $x$ avoiding every active mine except possibly $m$.

If this path misses $m$, append $a_N$. If it hits $m$ on a hop of length $a_k$ from a point $p$, so $p+a_k=m$, replace that hop by $a_N$, keep the subsequent hops in order, and put $a_k$ last. The bad landing becomes
$$p+a_N=m+(a_N-a_k)>m.$$
All subsequent pre-final landings are shifted right by $a_N-a_k$ and remain at most $S-a_k\le S-a_1$. Since $m$ is the largest active mine, those shifted landings are safe, and the final point is $S$.

If $x$ is mined and no active mine lies to the right of $x$, remove $x$ and use induction to reach $x$ with $a_1,\ldots,a_{N-1}$ while avoiding the other active mines. Let the last hop into $x$ have length $a_k$, from $p=x-a_k$. Replace this last hop by $a_N$ and put $a_k$ last. The intermediate landing is
$$p+a_N=x+(a_N-a_k)=S-a_k,$$
which lies strictly to the right of $x$ and no farther than $S-a_1$; no active mine lies there. The last jump lands at $S$.

It remains to handle the case in which $x$ is mined and some active mine $z$ satisfies $x<z\le S-a_1$. For each $i<N$, define
$$y_i=S-a_N-a_i=x-a_i,\qquad p_i=S-a_i.$$
The intended ending is
$$y_i \xrightarrow{a_N} p_i \xrightarrow{a_i} S.$$

The $y_i$ are distinct and lie below $x$, and the $p_i$ are distinct and lie above $x$, so all candidate points are distinct. Any active mine other than $x$ can equal at most one candidate point and therefore can block at most one index $i$. The point $x$ is not a candidate. There are $N-1$ candidate pairs and at most $N-2$ blocking mines other than $x$: the candidate count is the blocker count plus one, so at least one pair has both $y_i$ and $p_i$ unmined.

Choose such an $i$. The remaining $N-2$ lengths sum to $y_i$. Since $x$ and $z$ are two active mines to the right of $y_i$, at most $N-3$ active mines lie at or below $y_i$, and hence at most $N-3$ lie in the recursive active interval for those remaining lengths. By induction, the remaining lengths can be ordered to reach the safe point $y_i$. Then jump by $a_N$ to the safe point $p_i$, and by $a_i$ to $S$. The first final jump crosses $x$; the mine $z$ is crossed by the first final jump if $z<p_i$, and by the last final jump if $z>p_i$.

All four cases produce a safe ordering, and the strengthened induction is complete. The original grasshopper ordering follows.

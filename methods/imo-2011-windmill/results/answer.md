# Windmill construction: preserve an oriented side-count

## Statement

Let $S$ be a finite set of at least two points in the plane, no three collinear. A *windmill* starts with a line $\ell$ through a point $P \in S$; the line rotates clockwise about $P$ until it meets another point $Q \in S$, which becomes the new pivot, and so on indefinitely. Then one can choose the starting point $P$ and starting line $\ell$ so that the resulting windmill uses **every** point of $S$ as a pivot infinitely many times.

## Key idea

Give the windmill line an orientation (an arrow) and count the points of $S$ strictly on one chosen side. This count is a **conserved quantity** of the windmill: it does not change as the line rotates, and at a pivot swap the old pivot drops onto exactly the side the new pivot vacated. Starting from a line that splits $S$ as evenly as possible forces the rotating line to pass through every point in each full sweep.

## Proof

Choose an arrow on the starting line and let that arrow rotate continuously with the line. For an oriented line $\ell$, let $L(\ell)$ be the number of points of $S$ lying strictly to the **left** of $\ell$. The current pivot lies on $\ell$ and is counted on neither side.

**1. $L$ is invariant under the windmill.**
While the line rotates about a fixed pivot $P$ without meeting another point of $S$, no point of $S$ crosses the line, so $L$ is unchanged. Consider a pivot swap, where clockwise rotation about $P$ brings the line onto a new point $Q$, which becomes the pivot.

Place $Q$ at the origin with the line horizontal at the contact instant and $P=(a,0)$, $a>0$. First let the arrow point from $Q$ toward $P$. Just before contact, the line through $P$ has direction $d_-=(\cos\epsilon,\sin\epsilon)$, so
$$
\det(d_-,Q-P)=\det((\cos\epsilon,\sin\epsilon),(-a,0))=a\sin\epsilon>0.
$$
Thus $Q$ is on the left just before the swap. Just after the swap, the line through $Q$ has direction $d_+=(\cos\epsilon,-\sin\epsilon)$, so
$$
\det(d_+,P-Q)=\det((\cos\epsilon,-\sin\epsilon),(a,0))=a\sin\epsilon>0.
$$
Thus $P$ is on the left just after. If the arrow points the other way, both signed areas change sign, so $Q$ is on the right before and $P$ is on the right after. In all cases, the side that loses $Q$ gains $P$. Equivalently:

> When the pivot changes from $P$ to $Q$, the old pivot $P$ ends up on the same side that the new pivot $Q$ occupied just before.

So $L$ is unchanged at every swap. Therefore $L$ is constant throughout the windmill away from the swap instants.

**2. A balanced starting line exists through every point.**
For a point $T$, take an oriented line through $T$ containing no other point of $S$. Rotate it $180^\circ$ about $T$ and look at the open intervals between the finitely many moments when the line also contains another point of $S$. On each interval the line contains only $T$; crossing one of those moments changes the left-count by exactly $+1$ or $-1$, since no three points are collinear. After $180^\circ$ the arrow is reversed, so left and right have swapped roles.

- If $|S| = 2n+1$: the line through $T$ has $n+r$ on the left for some $r$; after $180^\circ$ it has $n-r$. The integer count starts at $n+r$, ends at $n-r$, and changes by unit steps, so some open interval has count $n$. Since the interval is open, choose the line's direction also avoiding the finitely many directions parallel to lines through two points of $S$. The line contains only $T$ and has $n$ points on **each** side.
- If $|S| = 2n$: the $2n-1$ points other than $T$ cannot split evenly. The left-count starts at some $a$ and ends at $2n-1-a$; these two values have average $n-\frac12$, so one is at most $n-1$ and the other is at least $n$. As the open-interval counts change by unit steps between them, some interval has count $n-1$. Choose the line's direction in that open interval and away from the finitely many directions parallel to lines through two points of $S$. The line contains only $T$, so it has $n-1$ on the left and $n$ on the right.

**3. Uniqueness of the balanced line in a fixed orientation.**
Fix an oriented direction and a balanced left-count $k$. Let $u$ be the left normal to that direction, so the parallel line $u \cdot x=t$ has left-count
$$
\#\{X \in S : u \cdot X>t\}.
$$
As $t$ increases through the finitely many values $u \cdot X$, this count strictly decreases, by the number of points at the crossed value. Thus among lines in this direction that contain at least one point of $S$, at most one can have left-count $k$.

**4. Conclusion.**
Choose $P$ arbitrarily and start the windmill on a balanced line through $P$: $n$ on each side if $|S|=2n+1$, or $(n-1,n)$ if $|S|=2n$. By Step 1, $L$ stays equal to this balanced value $k$ forever.

Fix any target $T \in S$, and let $\ell_T$ be a balanced oriented line through $T$ chosen as in Step 2, with direction not parallel to any line through two points of $S$. By Step 3, $\ell_T$ is the unique line of its orientation and left-count $k$ through a point of $S$. The windmill line always has left-count $k$ away from swap instants, and when it attains the orientation of $\ell_T$ it is not at a two-point swap. It must therefore coincide with $\ell_T$. Since the chosen direction is not parallel to a line through two points of $S$, this line contains no point of $S$ except $T$; the windmill line always contains its current pivot, so the pivot at that moment is $T$.

- If $|S| = 2n+1$ (split $n/n$): reversing the arrow preserves the balance, and the reversed direction is also not a two-point direction. The same uniqueness argument applies to either orientation, so it suffices for the windmill line to become **parallel** to $\ell_T$ — which happens within every $180^\circ$.
- If $|S| = 2n$ (split $(n-1,n)$): the split is not symmetric under reversing the arrow, so the windmill line must attain $\ell_T$'s exact oriented direction — which happens within every $360^\circ$.

After the first hit, each leg starts on a line through two points of $S$ and ends on another such line. The positive clockwise angles between successive pivot lines therefore come from a finite set, so they have a positive minimum. Because the process has infinitely many legs, the accumulated clockwise rotation is unbounded, and the line moves continuously through the intervening directions. Thus every completed block of $180^\circ$ of accumulated motion contains every undirected direction, and every completed block of $360^\circ$ contains every oriented direction. Hence every point $T \in S$ becomes a pivot again and again, and therefore **infinitely often**.

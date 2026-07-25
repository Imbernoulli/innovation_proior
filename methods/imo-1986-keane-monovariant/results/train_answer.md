I would call this approach Keane's monovariant, or more descriptively the partial-sum monovariant for the pentagon operation problem. The question is whether a certain discrete process on a regular pentagon must terminate after finitely many steps. We are given five integers assigned to the vertices, with the sole global condition that their sum is positive. The allowed operation acts on three consecutive vertices with values x, y, z where y is negative, and replaces them by x+y, -y, z+y. The middle vertex flips sign, while the negativity is pushed into the two neighbors. We must decide whether repeated application of this rule necessarily stops.

My first observation is that the total sum is invariant. Before the move the three relevant entries contribute x+y+z; after the move they contribute (x+y)+(-y)+(z+y), which is again x+y+z. So if the initial sum s is positive, it remains positive and unchanged forever. This invariant gives global control, but it cannot serve as a progress measure because it never changes. I therefore need a nonnegative integer-valued quantity that strictly decreases on every legal move.

A naive first attempt might be to count negative entries, but this fails immediately. When a negative middle vertex becomes positive, one or both of its neighbors may become negative, so the count of negative entries can rise. Measures based on individual coordinates also behave badly. The sum of absolute values can increase, and the sum of squares can increase as well. For example, operating at a negative entry y with neighbors a and c changes the sum of squares by 2y(a+c+y), whose sign depends on a+c+y, not just on y being negative. Coordinates can blow up during the process even though the process eventually terminates, so local size is the wrong lens.

The right lens turns out to be the collection of all consecutive-arc sums. Consider the five entries in cyclic order and look at the partial sums along each arc of length one, two, three, or four. The operation has a surprisingly simple effect on these partial sums. If p is the running sum just before the triple a, y, c, then before the move the four boundary partial sums are p, p+a, p+a+y, p+a+y+c. After the move the entries become a+y, -y, c+y, and the same four boundary partial sums become p, p+a+y, p+a, p+a+y+c. The two middle partial sums have simply swapped. Because y is negative, the smaller of the two moves earlier in the order. This suggests that some function of all partial sums should decrease.

Define the monovariant M as the sum of the absolute values of every proper consecutive-arc sum. Formally, if the entries are x_1 through x_5 with indices read modulo 5, then

M = sum over j from 1 to 5 and L from 1 to 4 of |x_j + x_{j+1} + ... + x_{j+L-1}|.

This is a nonnegative integer because all entries are integers and we are summing absolute values. It is bounded below by zero, so if I can show it strictly decreases at every legal operation, termination follows immediately.

To verify the decrease, write the cyclic order as (a, b, c, d, e) and suppose we operate at b, which is negative. After the operation the pentagon becomes (a+b, -b, c+b, d, e). I now compare all proper consecutive-arc sums before and after the move. Most of them appear with the same absolute value on both sides. The single entries a, b, c, d, e transform into a+b, -b, c+b, d, e, and the absolute values line up with the original list. The two-vertex arcs, three-vertex arcs, and most four-vertex arcs also pair off exactly. In fact, only one old arc sum fails to appear in the new list, and only one new arc sum fails to appear in the old list.

The old arc that disappears is the four-vertex arc omitting the operated middle vertex b, namely a+c+d+e. Since the total sum is s, this equals s-b. Its absolute value |s-b| is removed from M. The new arc that appears is a+c+d+e+2b, which equals s+b. Its absolute value |s+b| is added to M. Every other term cancels pairwise. Therefore the change in the monovariant is exactly

M_new - M_old = |s+b| - |s-b|.

Because b is negative, set t = |b| = -b, which is a positive integer. Then

M_new - M_old = |s-t| - |s+t|.

Since s is positive and t is positive, s+t is positive, so |s+t| = s+t. If s is at least t, then |s-t| = s-t, and the change is (s-t) - (s+t) = -2t. If s is smaller than t, then |s-t| = t-s, and the change is (t-s) - (s+t) = -2s. In either case,

M_new - M_old = -2 min(s, t) = -2 min(s, |b|).

Because s is a positive integer and |b| is a positive integer, min(s, |b|) is at least 1. Thus every legal operation decreases M by at least 2. A nonnegative integer cannot decrease by at least 2 infinitely many times, so only finitely many legal operations are possible.

When the process stops, no legal operation remains, which means no vertex can be negative; otherwise that vertex together with its two neighbors would form a legal triple. Hence the procedure necessarily terminates in a finite number of steps, ending in a configuration with all entries nonnegative. This completes the proof.

The key insight is that the operation, although destructive to individual coordinates, becomes a clean transposition when viewed through partial sums. The monovariant M captures all those partial sums at once, and the positive invariant total sum s is exactly what forces M to drop. The method is a classic example of a monovariant argument: find a quantity that is integer-valued, bounded below, and strictly decreasing under the allowed moves.

Written out in full, the deliverable is this pair: the monovariant itself and its exact decrease law. For entries $x_1,\dots,x_5$ in cyclic order, indices read modulo $5$, define
$$
M=\sum_{j=1}^{5}\sum_{L=1}^{4}\left|x_j+x_{j+1}+\cdots+x_{j+L-1}\right|,
$$
the sum of the absolute values of all twenty proper consecutive-arc sums. If the operation is applied at a negative vertex $b$, and $s=\sum_i x_i>0$ denotes the (invariant) total, then
$$
\Delta M = -2\min(s,|b|)\ \le\ -2 .
$$
Because $M$ is a nonnegative integer that drops by at least $2$ on every legal move, it can absorb only finitely many operations — at most $\lfloor M_0/2\rfloor$ of them, where $M_0$ is the monovariant of the starting configuration — before no negative vertex remains. That is the whole proof: the pentagon process on five integers with positive total sum always terminates, and it does so in a configuration where every entry is nonnegative, precisely because $M$ has nowhere left to fall.

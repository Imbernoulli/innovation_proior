Let me start with the obvious search, because it shows exactly what has to be escaped. A prisoner can open half the boxes. For that one prisoner, before seeing their own number, their number is equally likely to be in any one of the $100$ boxes, so opening $50$ boxes gives probability $1/2$ of success. If the prisoners each make independent random searches, then conditional on the hidden arrangement each prisoner's private random choice succeeds with probability $1/2$, independently of the other prisoners' private choices, so the chance that all $100$ succeed is $(1/2)^{100} \approx 7.9 \times 10^{-31}$. The bad part is not a single prisoner's marginal chance. The bad part is multiplying many almost unrelated success events. I need the prisoners' fates to be controlled by the same feature of the one hidden arrangement, so that the runs on which the group loses are concentrated into one structural obstruction instead of spread across one hundred independent misses.

The shared object is the placement itself. I can write it as a permutation $\sigma$ of $\{1,\ldots,100\}$, where box $j$ contains the number $\sigma(j)$. Every box has one outgoing pointer, from $j$ to $\sigma(j)$, and because every number appears exactly once, every box also has one incoming pointer. That means the directed graph of these pointers decomposes into disjoint cycles. A prisoner cannot see the whole permutation, but maybe the numbers they uncover can make them walk one of its cycles.

Suppose prisoner $p$ opens box $p$ first. If box $p$ contains $\sigma(p)$, the natural next box is the box labeled $\sigma(p)$. That box contains $\sigma^2(p)$, which points to box $\sigma^2(p)$, and so on. The boxes opened are

$$
p,\ \sigma(p),\ \sigma^2(p),\ \ldots
$$

This uses only information the prisoner actually sees. If the cycle of $\sigma$ containing $p$ has length $L$, then $L$ is the least positive integer with $\sigma^L(p)=p$. The box containing $p$ is $\sigma^{-1}(p)$, and on this cycle that predecessor is $\sigma^{L-1}(p)$, since $\sigma(\sigma^{L-1}(p))=p$. In the walk above, $\sigma^{L-1}(p)$ is the $L$-th box opened. So prisoner $p$ finds their own number exactly on the $L$-th opening, where $L$ is the length of the cycle containing $p$. Therefore prisoner $p$ succeeds within $50$ openings if and only if their cycle has length at most $50$.

Now the group event becomes exact. Every prisoner belongs to exactly one cycle of $\sigma$. All prisoners on a short cycle, length at most $50$, find their numbers; all prisoners on a long cycle, length at least $51$, fail. So all $100$ prisoners are freed if and only if every cycle of $\sigma$ has length at most $50$, equivalently if and only if the longest cycle of $\sigma$ is at most $50$. This is the correlation I was looking for. The rule has made the joint failure event a single permutation event: a cycle longer than $50$ exists.

I need the probability of that event. A permutation of $100$ elements cannot have two cycles longer than $50$, because two disjoint cycles of lengths at least $51$ would require at least $102$ elements. So the events "there is a cycle of length exactly $k$" are disjoint for $k=51,\ldots,100$.

For a fixed $k$ in that range, count permutations with a $k$-cycle. First choose the $k$ elements in the cycle, which gives $\binom{100}{k}$ choices. On that chosen set, the number of distinct directed cycles is $(k-1)!$, because a cyclic listing has $k!$ linear listings and each cycle is represented by its $k$ rotations. The remaining $100-k$ elements can be permuted arbitrarily, giving $(100-k)!$ choices. Thus the number of permutations with a cycle of length $k$ is

$$
\binom{100}{k}(k-1)!(100-k)!
= \frac{100!}{k!(100-k)!}(k-1)!(100-k)!
= \frac{100!}{k}.
$$

Dividing by the total number $100!$ of permutations gives

$$
\Pr(\sigma \text{ has a cycle of length } k)=\frac{1}{k}, \qquad 51 \le k \le 100.
$$

Because those long-cycle events are disjoint, the failure probability is just the sum:

$$
\Pr(\text{longest cycle}>50)
= \sum_{k=51}^{100}\frac{1}{k}
= H_{100}-H_{50}.
$$

So the success probability of the cycle-following rule is

$$
\Pr(\text{all freed})
= 1-(H_{100}-H_{50}).
$$

Numerically,

$$
H_{100}-H_{50}=0.6881721793101953\ldots,
$$

so

$$
\Pr(\text{all freed})=0.3118278206898047\ldots.
$$

That is about $31.18\%$, not $2^{-100}$. The whole improvement is in the dependence structure: if the longest cycle is at most $50$, everyone wins; if a cycle is longer than $50$, exactly the prisoners on that cycle lose.

I should also check the large-size behavior, because this number could have been a lucky finite accident. With $2n$ prisoners and a limit of $n$ boxes, the same argument gives

$$
\Pr(\text{failure})=\sum_{k=n+1}^{2n}\frac{1}{k}=H_{2n}-H_n.
$$

This tends to $\ln 2$, since

$$
\sum_{k=n+1}^{2n}\frac{1}{k}
= \sum_{j=1}^{n}\frac{1}{n+j}
= \frac{1}{n}\sum_{j=1}^{n}\frac{1}{1+j/n}
\to \int_0^1 \frac{dx}{1+x}
= \ln 2.
$$

The direction is important. The failure sums increase with $n$, because

$$
(H_{2n+2}-H_{n+1})-(H_{2n}-H_n)
= \frac{1}{2n+1}+\frac{1}{2n+2}-\frac{1}{n+1}
= \frac{1}{2(2n+1)(n+1)} > 0.
$$

So the survival probabilities decrease monotonically to $1-\ln 2 \approx 0.3068528194$. For $100$ prisoners, where $n=50$, the value $0.3118278207$ is slightly above that limiting floor.

It remains to ask whether a different strategy could beat this. I want an upper bound that does not assume cycle-following. Make the room more generous: opened boxes stay visible for later prisoners, and if a prisoner arrives with their own number already visible, they succeed immediately. I can also require a prisoner to stop once they find their number; in the original room, opening more boxes after success cannot help anyone because the boxes are reset and no one hears about it. Any original strategy can be played in this easier room, so the best original success probability is at most the best success probability there.

In this easier room, a fresh unopened box has no exploitable label information left. Conditional on the boxes and numbers already exposed and on the previous prisoners all having succeeded, the still-hidden numbers are uniformly arranged among the still-closed boxes. If the next prisoner must search, choosing one closed label rather than another cannot change the chance that the desired number is in the next fresh box; only the number of fresh openings matters. So the whole-group success probability in this easier room is independent of the rule used to choose fresh boxes. I can compute that common upper-bound value by using the cycle-following rule there.

Now compare cycle-following in the easier room to cycle-following in the original room. If prisoner $p$'s number is already visible when $p$ arrives, then some earlier successful walk has exposed the cycle containing $p$; for that earlier walk to be successful, that cycle has length at most $50$, so $p$ would also find their number within $50$ openings in the original room. If $p$'s number is not visible, then $p$ starts on a cycle no previous successful walk has exposed, and the walk uses fresh boxes until it either returns within $50$ steps or runs too long. Thus cycle-following succeeds in the easier room on exactly the same permutations on which it succeeds in the original room: precisely those with no cycle longer than $50$.

The easier room gives an upper bound for every original strategy, and cycle-following reaches that bound. The final rule is simple: prisoner $p$ opens box $p$, then repeatedly opens the box whose label is the number just read, stopping after at most $50$ boxes. The proof is carried by one equivalence and one count: all prisoners succeed exactly when the longest cycle of the hidden permutation is at most $50$, and the probability of a long cycle is $\sum_{k=51}^{100}1/k$. That gives the optimal success probability $0.3118278206898047\ldots$.

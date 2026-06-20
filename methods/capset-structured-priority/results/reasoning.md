Random multi-start clears the floor and even hits the optimum at `n = 4`, but the feedback made
its ceiling plain: it is a lottery over orders, and the right tail of the cap-size distribution
thins so fast that more restarts buy almost nothing as `n` grows. The diagnosis was that the orders
are *blind* — uniform noise, with no preference for points that sit in structured positions. So the
move is to replace the random order with a *priority*: score every vector by some deterministic
function of its coordinates, and feed the same greedy admission rule the vectors in
highest-priority-first order. If the priority reflects the actual symmetry of `F_3^n`, the greedy
fill should prefer points that pack the space efficiently, and I should beat the random lottery
without spending thousands of restarts. The admission rule is unchanged, so validity is still free;
only the ordering function changes.

The question is what structure to reward. `F_3^n` has a lot of symmetry to lean on, and the large
caps in the literature are known to be highly structured objects — they are not random, they have
regular weight profiles and reflection symmetries. Two structural features feel natural to encode.
First, *reflection symmetry*: pair coordinate `i` with coordinate `n−1−i`, and reward a vector when
those paired entries agree, `el[i] == el[n−1−i]`. A cap built from reflection-symmetric vectors
inherits a symmetry that tends to make its line structure regular, so I will add a bonus per matched
reflection pair. Second, a *weight profile*: the number of nonzero coordinates (the Hamming weight)
of a vector, taken mod 3, seems to matter for how vectors interact under the line condition `a + b +
c ≡ 0`, since that condition is itself a statement about coordinate sums mod 3. I will gently prefer
vectors whose weight mod 3 sits in a chosen residue class, to bias the greedy fill toward a coherent
weight layer rather than a mix. A tiny tie-break on the coordinate sum keeps the ordering total. The
priority is the sum of these terms — symmetric, deterministic, parameter-light — and it is exactly
the kind of hand-designed structured score one would write down before resorting to search.

Let me reason about what I expect this to do, and I want to be careful not to oversell it. The
symmetry rewards should beat lexicographic decisively, because they give the greedy process a
*reason* to prefer one point over another that is aligned with the geometry instead of with the
counting order. But I am genuinely unsure whether a hand-crafted symmetric priority will beat the
*best of thousands of random orders* — and that uncertainty is the point. A single structured order
is still a single order; if my chosen structure happens to be well-aligned with the optimal cap's
structure I will win, but if my guesses about which symmetries matter are even slightly off, a
deterministic structured order can easily underperform the maximum over a large random sample. The
reflection-and-weight priority is a *reasonable* guess, not a derived one. So my honest expectation
is: comfortably above the lexicographic floor at every `n`, in the same general band as random
multi-start, and quite possibly *below* it at some `n` — because best-of-thousands is a strong
baseline and one hand-tuned order is a single shot. At `n = 4` I would hope to clear the floor of
`16` but I would not bet on matching the `20` that random multi-start found by sheer sampling. At
larger `n` I expect the structured order to land somewhere in the same neighborhood as the random
best, not dramatically beyond it.

If that is how it comes out, it is not a disappointment — it is the crucial lesson of the ladder.
It would show that *having* a structured priority is not enough; the priority has to be the
*right* one, finely shaped to the dimension, and a human guessing at which symmetries to reward
cannot reliably out-do brute sampling. The greedy-priority skeleton is clearly the correct
machinery — it is exactly the skeleton the strong constructions use — but the function plugged into
it is everything, and hand-design plateaus because the space of useful priorities is large and
non-obvious. What reaches the records is not a cleverer human guess but a priority *discovered by
search* over the function space, tuned to the specific `n`, encoding regularities a person would not
write down. That is the endpoint: take the priority function that an evolutionary program search
(FunSearch) actually discovered for this exact skeleton, and run it. So this rung's job is to
establish that the skeleton is right and the hand-designed priority is not — to motivate handing the
priority itself over to search. That is the final rung.

One caution I will respect for the numbers: the structured priority can have ties (many vectors
share the same reflection/weight score), and how ties are broken affects the result, so I fix the
tie-break deterministically and run the verifier on every returned cap, never trusting a size I have
not checked.

The last rung settled the architecture and exposed the real bottleneck. The greedy-priority
skeleton is the right machinery — score every vector, add the highest-priority valid one, block the
closed lines, repeat — and the earlier rungs are all special cases of it: lexicographic is the
priority "index of the vector," random multi-start is "a random score, best of many," my structured
attempt is "reflection matches plus a weight layer." Every one of them plateaus below the optima,
and the structured priority, my best hand-designed guess, actually fell *below* best-of-thousands
random restarts. The lesson is sharp: the skeleton is not the limitation, the *priority function*
is, and a human cannot reliably write the right one. The space of useful priorities is large and
non-obvious — the function that reaches the records encodes regularities of `F_3^n` at a specific
`n` that nobody would think to hand-code. So the move is to stop guessing and hand the priority
itself to *search*. This is exactly what FunSearch did: an evolutionary loop pairing a pretrained
LLM with the cap-set evaluator, evolving the body of the `priority` function over millions of
samples, keeping the ones that build larger caps. The endpoint of this ladder is to take the
priority function that search actually discovered and run it through the same skeleton.

Let me reason about what the discovered function should look like, because it sharpens why hand-design
failed. My structured attempt rewarded reflection matches and a weight layer, but it did so *uniformly*
— a flat bonus, applied the same way regardless of how many zeros a vector has. The discovered
function is far more conditional than anything I would have written. It branches first on the *number
of zeros* in the vector: vectors with no zeros (full weight) are treated completely differently from
vectors with some zeros, getting a large additive boost and then reflection bonuses that *multiply*
the score rather than add to it. For vectors that do have zeros, it walks through the coordinates and
applies a *position-dependent multiplicative* adjustment to each zero — the first zero, the last zero,
and the interior zeros are each scaled differently, with factors that depend on `n` and on the zero's
ordinal position. And it stacks reflection-pair bonuses (`el[1]==el[-1]`, `el[2]==el[-2]`,
`el[3]==el[-3]`) as *multiplicative* `×1.5` factors, layered on top of the additive base. This is a
deeply non-linear, branch-heavy, position-and-count-sensitive score — nothing like the clean symmetric
sum I wrote down, and nothing a person would naturally propose, because its structure was *found*, by
selecting whatever happened to make the greedy fill reach `512`, not by reasoning from symmetry first.

Now the crucial point about *which* dimension this function is for. It was discovered while searching
specifically at `n = 8`, and its constants — the `×1.5` reflection factors, the `n·0.5^{in_el}` zero
weights, the `el[3]==el[-3]` term — are tuned to the structure of the `512`-cap in eight dimensions.
So I should *not* expect it to be good at small `n`. At `n = 4, 5, 6, 7` it is being run far outside
the regime it was evolved for, and I expect it to land back near the trivial `2^n` neighborhood — it
was never optimized to pack those spaces, and a function exquisitely tuned to one dimension carries no
guarantee elsewhere. The honest prediction is: at `n < 8` this discovered function is mediocre, quite
possibly *below* my structured priority and below random multi-start, because it is the wrong tool for
those dimensions. Its whole value is concentrated at `n = 8`, where I expect it to do the thing it was
discovered to do: build a cap of exactly `512`, the record, improving on the previous best
construction of `496`. That single number is the payoff of the entire ladder — it is the strong size
that no amount of random sampling or hand-designed symmetry reached, bought by *searching the function
space* for the priority.

I want to verify, not just assert, that I have reproduced the genuine discovery. Two checks. First,
run the discovered priority through the exact skeleton at `n = 8` and confirm the cap it builds has
size `512` and passes the validity verifier — a real cap, not a claimed one. Second, and stronger,
compare the *set of points* my run produces against the explicit `512`-cap stored in the FunSearch
repository: if the greedy fill with the discovered priority reproduces the same `512` points the
authors recorded, then I have not merely reached the number, I have reconstructed the exact object the
search found. If those match, the endpoint is genuine.

And here is where the ladder honestly ends. `512` at `n = 8` is the current frontier — it *is* the
record this construction set, improving the prior `496`, and I am reproducing it, not beating it.
There is no rung above this that I should claim, because going further would mean *running the
evolutionary search itself* (millions of LLM samples under evaluation) to discover a new priority,
which is a different kind of computation entirely, not a single constructor. The companion discovered
function for `n = 9` reaches the known-best `1082` in nine dimensions by the same mechanism, which I
can note as corroboration that the method generalizes, but `n = 8 → 512` is the headline. So the
ladder is: a fixed blind order gives `2^n`; many random orders give the best of a thin lottery; a
hand-designed structured priority lands in the same band and no higher; and only a priority
*discovered by program search* over the function space reaches the record `512`. The skeleton was
right all along; the discovery was the function — and that is the part that search, not derivation,
had to supply.

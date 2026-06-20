The structured-priority method settled the architecture and exposed the real bottleneck. The
greedy-priority skeleton is the right machinery — score every vector, add the highest-priority valid
one, block the closed lines, repeat — and every earlier method is a special case of it: lexicographic
is the priority "index of the vector," random multi-start is "a random score, best of many," my
structured attempt is "reflection matches plus a weight layer." All of them plateau below the optima,
and the structured priority, my best hand-designed guess, actually fell *below* best-of-thousands
random restarts. The lesson is sharp: the skeleton is not the limitation, the *priority function* is,
and a human cannot reliably write the right one. The space of useful priorities is large and
non-obvious — the function that reaches the records encodes regularities of `F_3^n` at a specific `n`
that nobody would think to hand-code. So the move is to stop guessing and hand the priority itself to
*search*. This is exactly what FunSearch did: an evolutionary loop pairing a pretrained LLM with the
cap-set evaluator, evolving the body of the `priority` function over millions of samples, keeping the
ones that build larger caps. The endpoint is to take the priority function search actually discovered
and run it through the same skeleton.

The discovered function sharpens why hand-design failed. My structured attempt rewarded reflection
matches and a weight layer *uniformly* — a flat bonus regardless of how many zeros a vector has. The
discovered function is far more conditional than anything I would write. It branches first on the
*number of zeros*: full-weight vectors (no zeros) get a large additive boost and reflection bonuses
that *multiply* the score; vectors with zeros get a different multiplicative treatment, walking through
the coordinates and applying a *position-dependent* factor to each zero — first zero, last zero, and
interior zeros each scaled differently, with factors depending on `n` and the zero's ordinal position.
And it stacks reflection-pair matches (`el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]`) as
multiplicative `×1.5` factors layered on the additive base. This is a deeply non-linear, branch-heavy,
position-and-count-sensitive score — nothing like the clean symmetric sum I wrote, and nothing a person
would naturally propose, because its structure was *found*, by selecting whatever made the greedy fill
reach `512`, not by reasoning from symmetry first.

The crucial point is *which* dimension this function is for. It was discovered while searching
specifically at `n = 8`, and its constants are tuned to the structure of the `512`-cap in eight
dimensions. So I should not expect it to be good at small `n`: at `n = 4..7` it runs far outside its
regime, and I expect it back near the trivial `2^n` neighborhood — it was never optimized to pack those
spaces. The running confirms it: `16, 32, 64, 128` for `n = 4..7`, at or below my structured priority
and below random multi-start. Its whole value is concentrated at `n = 8`, where it does the thing it was
discovered to do: build a cap of exactly `512`, the record, improving the previous best construction of
`496`. That single number is the payoff of the entire ladder — the strong size that no amount of random
sampling or hand-designed symmetry reached, bought by searching the function space for the priority.

I verify rather than assert. First, running the discovered priority through the skeleton at `n = 8`
yields a cap of size `512` that passes the validity verifier — a real cap, not a claimed one. Second,
and stronger, the set of `512` points my run produces equals, as a set, the explicit `512`-cap stored
in the FunSearch repository: all 512 points coincide, so I have reconstructed the exact object the
search found, not merely the number. So the method I present is the **FunSearch-evolved priority**, run
verbatim through the greedy skeleton. And here the ladder honestly ends: `512` at `n = 8` is the current
frontier — it *is* the record this construction set — and I am reproducing it, not beating it. There is
no rung above this I should claim, because going further would mean running the evolutionary search
itself (millions of LLM samples under evaluation), a different kind of computation, not a single
constructor. The companion discovered function for `n = 9` reaches the known-best `1082` by the same
mechanism, corroborating that the method generalizes, but `n = 8 → 512` is the headline. The skeleton
was right all along; the discovery was the function — the part that search, not derivation, had to
supply.

```python
import itertools
import numpy as np

def priority(el, n):
    """Priority function discovered by FunSearch (Romera-Paredes et al., Nature 2024);
    verbatim from github.com/google-deepmind/funsearch cap_set/cap_set.ipynb.
    Builds a 512-cap in n=8, improving the previous best construction of size 496."""
    score = n
    in_el = 0
    el_count = el.count(0)
    if el_count == 0:
        score += n ** 2
        if el[1] == el[-1]:
            score *= 1.5
        if el[2] == el[-2]:
            score *= 1.5
        if el[3] == el[-3]:
            score *= 1.5
    else:
        if el[1] == el[-1]:
            score *= 0.5
        if el[2] == el[-2]:
            score *= 0.5
    for e in el:
        if e == 0:
            if in_el == 0:
                score *= n * 0.5
            elif in_el == el_count - 1:
                score *= 0.5
            else:
                score *= n * 0.5 ** in_el
            in_el += 1
        else:
            score += 1
    if el[1] == el[-1]:
        score *= 1.5
    if el[2] == el[-2]:
        score *= 1.5
    return score

def construct(n):
    """Greedy with the FunSearch-discovered priority. Builds a 512-cap at n=8."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority(tuple(int(x) for x in v), n) for v in av], dtype=float)
    capset = np.empty((0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        k = int(np.argmax(priorities))
        v = av[None, k]
        blocking = np.einsum('cn,n->c', (-capset - v) % 3, powers).astype(np.int64)
        priorities[blocking] = -np.inf
        priorities[k] = -np.inf
        capset = np.concatenate([capset, v], axis=0)
    return capset
```

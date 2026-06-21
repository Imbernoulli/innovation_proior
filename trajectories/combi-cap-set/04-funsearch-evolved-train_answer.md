My hand-designed structured priority beat the lexicographic floor but landed *below* best-of-thousands random multi-start at every dimension shown — $18 < 20$, $36 < 39$, $64 < 77$, $138 < 147$ — which is the intended and decisive result. It settles the architecture and names the real bottleneck. The greedy-priority skeleton is the right machinery, and every earlier rung is a special case of it: lexicographic is the priority "index of the vector," random multi-start is "a random score, best of many," my structured attempt is "reflection matches plus a weight layer." All of them plateau, and my best hand-designed guess fell below a brute lottery. The skeleton is not the limitation — the *priority function* is, and a human cannot reliably write the right one, because the space of useful priorities is large and non-obvious and the function that reaches the records encodes regularities of $F_3^n$ at a specific $n$ that nobody would think to hand-code. So the move is to stop guessing and hand the priority itself to *search*.

The method I propose for this endpoint is the **FunSearch-evolved priority function** (Romera-Paredes et al., Nature 2024): keep the exact greedy skeleton — score every vector, add the highest-priority valid one, set the closing point of each line it forms to $-\infty$, repeat — but replace the hand-designed priority with the one an evolutionary loop pairing a pretrained LLM with the cap-set evaluator actually discovered, evolving the body of the `priority` function over millions of samples and keeping whatever built larger caps. The discovered function is far more conditional than anything I would write, and its structure is exactly what sharpens *why* hand-design failed. It branches first on the number of zeros in the vector, `el_count = el.count(0)`. Full-weight vectors (no zeros) are treated completely differently: they get a large additive boost $\mathtt{score} \mathrel{+}= n^2$, and then reflection-pair matches `el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]` *multiply* the score by $1.5$ rather than adding to it. Vectors that do have zeros instead get their reflection matches on the first two pairs penalized by a $\times 0.5$ factor, and then the function walks the coordinates applying a *position- and count-dependent multiplicative* adjustment to each zero: the first zero scales by $n \cdot 0.5$, the last zero by $0.5$, and an interior zero at ordinal position $\mathtt{in\_el}$ by $n \cdot 0.5^{\,\mathtt{in\_el}}$, while every nonzero coordinate adds $1$. Finally, on top of all of that, it stacks two more reflection-pair bonuses `el[1]==el[-1]` and `el[2]==el[-2]` as further $\times 1.5$ multiplicative factors. This is a deeply non-linear, branch-heavy, position-and-count-sensitive score — nothing like the clean symmetric *sum* I wrote down, and nothing a person would naturally propose, because its structure was *found* by selecting whatever made the greedy fill reach $512$, not derived from symmetry first.

The crucial point is *which* dimension this function is for, because it explains both its mediocrity below $n=8$ and its single great payoff at $n=8$. It was discovered while searching specifically at $n = 8$, and its constants — the $\times 1.5$ reflection factors, the $n \cdot 0.5^{\,\mathtt{in\_el}}$ zero weights, the very presence of the `el[3]==el[-3]` term — are tuned to the structure of the $512$-cap in eight dimensions. Run far outside that regime, at $n = 4, 5, 6, 7$, it was never optimized to pack those spaces and a function exquisitely tuned to one dimension carries no guarantee elsewhere, so I expect it to land back near the trivial $2^n$ neighborhood, quite possibly below my structured priority and below random multi-start — the wrong tool for those dimensions. Its entire value is concentrated at $n = 8$, where it does the thing it was discovered to do: build a cap of exactly $512$, the record, improving the previous best construction of $496$. That single number is the payoff of the whole ladder — the strong size that no amount of random sampling or hand-designed symmetry reached, bought by searching the *function space* for the priority. And I verify rather than assert it: running the discovered priority through the exact skeleton at $n = 8$ must yield a cap of size $512$ that passes the validity verifier, and — stronger — the set of $512$ points it produces must coincide with the explicit $512$-cap recorded in the FunSearch repository, so that I have reconstructed the exact object the search found, not merely matched its size. Here the ladder honestly ends. $512$ at $n=8$ is the current frontier and I am reproducing it, not beating it; there is no rung above this to claim, because going further would mean running the evolutionary search itself — millions of LLM samples under evaluation — to discover a *new* priority, a different kind of computation entirely, not a single constructor. (The companion discovered function for $n = 9$ reaches the known-best $1082$ by the same mechanism, corroborating that the method generalizes.) The skeleton was right all along; the discovery was the function, and that is the part search, not derivation, had to supply.

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

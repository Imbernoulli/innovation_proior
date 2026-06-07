# Aho-Corasick multi-keyword matching

## Problem

Given a finite set of nonempty keywords `K = {y1, ..., yk}` and a text `x` of
length `n`, report every occurrence of every keyword in `x`, overlaps included.
The straightforward method scans the text once per keyword and costs
`Theta(k*n)`; the goal is to read the text once, with non-output work independent
of `k`.

## Key idea

Build one finite-state pattern matching machine for the whole keyword set, then
run the text through it once. The machine has three pieces:

- **goto** `g(s, a)` is the trie of all keywords. The path from the root to a
  state spells a prefix of some keyword. Missing edges mean failure, while a
  missing edge at the root acts like a self-loop to the root.
- **failure** `f(s)` generalizes the Knuth-Morris-Pratt failure function from one
  pattern to a trie. If state `s` spells `u`, then `f(s)` is the state spelling
  the longest proper suffix of `u` that is also a prefix of some keyword.
- **output links** keep reporting separate from failure recovery. `output(s)`
  stores only keywords ending exactly at state `s`; `out_link(s)` points to the
  nearest failure ancestor with a nonempty local output.

Search keeps a current state. For each text symbol, follow failure links until a
goto edge is available, take that edge, emit local outputs of the new state, then
walk output links to emit terminal suffix states.

The optional deterministic next-move table over a chosen alphabet is filled in
breadth-first order: `delta(0, a) = g(0, a)`, and for a non-root state `r`,
`delta(r, a) = g(r, a)` when that goto exists, otherwise
`delta(r, a) = delta(f(r), a)`.

## Correctness

- The failure construction is correct by induction on trie depth: for a child
  `s = g(r, a)`, the candidates for a proper suffix of `s` are obtained by
  walking `f(r), f(f(r)), ...` and extending the first state that has a goto on
  `a`.
- The output-link chain from `s` visits exactly the terminal states on the
  failure chain of `s`, so local outputs plus output-link outputs are precisely
  the keywords whose strings are suffixes of the string for `s`.
- After processing text prefix `x[:j]`, the current state spells the longest
  suffix of that prefix that is also a keyword prefix. Therefore the emitted
  outputs at position `j` are exactly the keywords ending there.

## Complexity

Construction is linear in the total keyword length: trie insertion visits each
keyword character once, the failure-link construction has an amortized linear
number of failure steps charged over keyword paths, and assigning `out_link` is
constant time per trie edge because the failure ancestor has already been
processed in breadth-first order.

During search, each input symbol causes exactly one goto transition and the total
number of failure transitions over the whole scan is less than `n`, because every
failure lowers trie depth and gotos can raise it only by one. Non-output search
therefore takes fewer than `2n` state transitions, plus the unavoidable cost of
emitting the matches.

Using the deterministic next-move table replaces the goto/failure loop with one
transition per input symbol, at the cost of storing more transitions.

## Algorithm

```python
from collections import deque


class AhoCorasick:
    def __init__(self):
        self.goto = [{}]        # goto[state][symbol] -> state; missing == fail
        self.output = [[]]      # keywords ending exactly at this state
        self.fail = [0]         # failure link
        self.out_link = [0]     # nearest terminal state on the failure chain

    def add_keyword(self, word):
        state = 0
        for ch in word:
            nxt = self.goto[state].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto.append({})
                self.output.append([])
                self.fail.append(0)
                self.out_link.append(0)
                self.goto[state][ch] = nxt
            state = nxt
        if word not in self.output[state]:
            self.output[state].append(word)

    def build(self):
        queue = deque()
        for _, s in self.goto[0].items():
            self.fail[s] = 0
            queue.append(s)
        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while ch not in self.goto[state] and state != 0:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                self.out_link[s] = (
                    self.fail[s]
                    if self.output[self.fail[s]]
                    else self.out_link[self.fail[s]]
                )
        return self

    def search(self, text):
        state = 0
        for i, ch in enumerate(text):
            while ch not in self.goto[state] and state != 0:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for word in self.output[state]:
                yield (i - len(word) + 1, word)
            out = self.out_link[state]
            while out:
                for word in self.output[out]:
                    yield (i - len(word) + 1, word)
                out = self.out_link[out]


def build_matcher(keywords):
    ac = AhoCorasick()
    for w in keywords:
        ac.add_keyword(w)
    return ac.build()


if __name__ == "__main__":
    ac = build_matcher(["he", "she", "his", "hers"])
    print(sorted(ac.search("ushers")))
    # [(1, 'she'), (2, 'he'), (2, 'hers')]
```

On `{he, she, his, hers}` over `"ushers"` the machine reports `she` and `he`
ending at the `e` and `hers` ending at the final `s`: every keyword substring,
overlaps included, in one pass.

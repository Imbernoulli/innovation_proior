The problem is to find every occurrence of every keyword in a long text, including overlapping matches, without paying a cost proportional to the number of keywords. The obvious approach scans the text once per keyword, so a query with dozens of keywords reads the same corpus dozens of times. Knuth–Morris–Pratt solves the single-pattern case in linear time, but running it independently for each keyword simply rebuilds the same k-times-text cost. A trie matches all keywords that start at the same text position simultaneously, yet when the next character has no outgoing edge the trie offers no rule for recovering the partially matched prefix; one must restart at every position. Building a full deterministic automaton by determinizing a regular expression gives one transition per symbol, but the subset construction can blow up exponentially in the size of the pattern set and is complicated to implement. What is missing is a structure that has the trie's shared-prefix matching and KMP's failure recovery at the same time.

The method that supplies both is the Aho–Corasick algorithm. It builds a single finite-state machine for the entire keyword set and scans the text once. The first component is the goto function, which is just the trie of all keywords: shared prefixes share nodes, and each node reached by a complete keyword is tagged with that keyword. The second component is a failure link for every node. If a node represents the string u, its failure link points to the deepest node whose string is a proper suffix of u and is also a prefix of some keyword. This directly generalizes the KMP failure function from a single pattern to a set of patterns. The third component is an output link: from each node, an output link jumps to the nearest failure ancestor that actually ends a keyword, so suffix keywords can be reported without walking through empty failure states one by one.

Searching is then a single left-to-right pass. Maintain a current state. For each input character, follow failure links until the current state has a goto edge for that character, then take that edge. Because the root has an implicit self-loop on every missing character, this loop always terminates. After moving, emit every keyword stored at the new state, then follow output links and emit their keywords as well. Each text character causes exactly one goto transition, and the total number of failure transitions over the whole scan is bounded by the text length, so the non-output work is linear in the text length and does not depend on the number of keywords. Construction is also linear in the total keyword length: the trie is built by inserting each keyword once, and the failure links are computed in breadth-first order so that when a node's failure link is needed, its target has already been processed.

```python
from collections import deque


class AhoCorasick:
    def __init__(self):
        self.goto = [{}]      # trie edges: goto[state][char] -> state
        self.output = [[]]    # keywords ending exactly at this state
        self.fail = [0]       # failure link
        self.out_link = [0]   # nearest terminal state on failure chain

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

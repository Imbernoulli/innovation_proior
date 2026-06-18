# Link-Cut Tree (dynamic trees)

## Problem

Maintain a forest of $n$ valued nodes under online operations: `link(u, v)`
(add edge, $u$ and $v$ in different trees), `cut(u, v)` (remove an edge),
`connected(u, v)`, and the sum of node values on the path between $u$ and $v$.
The forest *shape* changes, so static path-decomposition structures do not
apply. $n$ and the number of operations are up to $\sim 10^5$.

## Key idea

A static heavy-light decomposition answers path queries in $O(\log^2 n)$ because a
root-to-node path crosses $O(\log n)$ light edges, but its chains are fixed -
`link`/`cut` reshape the tree and destroy the decomposition. Keep "a path is a few
ordered ranges," but make the chains **dynamic**:

- **Preferred paths.** Every node selects at most one child as its *preferred*
  child; the preferred edges partition the forest into *preferred paths*. The
  choice is mutable, unlike the static heavy chains.
- **Auxiliary (splay) trees keyed by depth.** Each preferred path is stored in a
  splay tree whose in-order order is depth along the path (shallow -> deep). A
  contiguous depth-range is a splay subtree, and its aggregate sits at the
  subtree root.
- **Path-parent pointers.** The splay-tree root of a path stores a parent pointer
  to the represented-tree parent of the path's top node; that parent does **not**
  point back (a node has one parent but unboundedly many virtual children). So
  $x$ is its own splay root iff its parent claims neither child slot - that test
  distinguishes a real splay edge from a virtual path-parent edge.

**`access(x)` is the one core operation.** It makes the path from the real root
down to $x$ a single preferred path / splay tree with $x$ on top, by climbing:
splay $x$, detach its deeper part (just overwrite the right child - the old child
keeps its upward pointer and so becomes virtual), then repeatedly move to the
path-parent $w$, splay $w$, graft the carried chain as $w$'s right (deeper)
subtree, and continue to the root; finish by splaying $x$.

Everything else is `access` plus $O(1)$ pointer surgery:

- **`make_root(x)`** = `access(x)`, then **reverse** the accessed chain. Re-rooting
  at $x$ flips the orientation of the $x$-to-old-root path, which is exactly a
  range-reverse of the depth-ordered splay tree (old root, the shallowest, becomes
  the deepest). The reversal is done lazily: a `rev` flag whose pushdown swaps a
  node's children and propagates. A plain `access` cannot re-root - it leaves depth
  order intact, so $x$ stays deepest.
- **`find_root(x)`** = `access(x)`, splay, walk to the leftmost (shallowest) node,
  splay it. **`connected(u, v)`** $=$ `find_root(u) == find_root(v)`.
- **`link(u, v)`** = `make_root(u)`, then set $u$'s parent to $v$ (a new virtual
  edge). Precondition: $u, v$ in different trees.
- **`cut(u, v)`** = `make_root(u)`, `access(v)`, splay $v$. The edge $(u, v)$ exists
  iff $u$ is $v$'s left child and has no right child of its own (they are adjacent
  on the path); if so, clear $v$'s left child and $u$'s parent.
- **`path_query(u, v)`** = `make_root(u)`, `access(v)`; then the path $u \to v$ is
  the splay tree at $v$, and the aggregate stored at $v$ is the answer.

## Code

```python
import sys


class Forest:
    """A dynamic forest with per-node values: link, cut, connected, and
    path-sum queries. Node 0 is a null sentinel; real nodes are 1..n."""

    def __init__(self, n, values):
        self.ch = [[0, 0] for _ in range(n + 1)]  # splay children: [left, right]
        self.fa = [0] * (n + 1)                   # splay parent OR path-parent
        self.val = [0] * (n + 1)                  # node value
        self.sm = [0] * (n + 1)                   # sum of values in this splay subtree
        self.rev = [0] * (n + 1)                  # lazy "reverse this subtree" flag
        for i in range(1, n + 1):
            self.val[i] = self.sm[i] = values[i - 1]

    def _is_root(self, x):
        f = self.fa[x]
        return self.ch[f][0] != x and self.ch[f][1] != x

    def _side(self, x):
        return 1 if self.ch[self.fa[x]][1] == x else 0

    def _pushup(self, x):
        l, r = self.ch[x]
        self.sm[x] = self.sm[l] + self.val[x] + self.sm[r]

    def _apply_rev(self, x):
        if x:
            self.ch[x][0], self.ch[x][1] = self.ch[x][1], self.ch[x][0]
            self.rev[x] ^= 1

    def _pushdown(self, x):
        if self.rev[x]:
            self._apply_rev(self.ch[x][0])
            self._apply_rev(self.ch[x][1])
            self.rev[x] = 0

    def _rotate(self, x):
        y = self.fa[x]
        z = self.fa[y]
        k = self._side(x)
        if not self._is_root(y):
            self.ch[z][self._side(y)] = x
        self.fa[x] = z
        w = self.ch[x][k ^ 1]
        self.ch[y][k] = w
        if w:
            self.fa[w] = y
        self.ch[x][k ^ 1] = y
        self.fa[y] = x
        self._pushup(y)
        self._pushup(x)

    def _splay(self, x):
        stack = [x]
        y = x
        while not self._is_root(y):
            y = self.fa[y]
            stack.append(y)
        while stack:
            self._pushdown(stack.pop())
        while not self._is_root(x):
            y = self.fa[x]
            if not self._is_root(y):
                if self._side(x) == self._side(y):
                    self._rotate(y)   # zig-zig: rotate the parent first
                else:
                    self._rotate(x)   # zig-zag
            self._rotate(x)

    def access(self, x):
        last = 0
        cur = x
        while cur:
            self._splay(cur)
            self.ch[cur][1] = last   # graft lower chain as the deeper subtree
            self._pushup(cur)
            last = cur
            cur = self.fa[cur]
        self._splay(x)
        return last

    def make_root(self, x):
        self.access(x)
        self._apply_rev(x)

    def find_root(self, x):
        self.access(x)
        while self.ch[x][0]:
            self._pushdown(x)
            x = self.ch[x][0]
        self._splay(x)
        return x

    def connected(self, u, v):
        if u == v:
            return True
        return self.find_root(u) == self.find_root(v)

    def link(self, u, v):
        self.make_root(u)
        if self.find_root(v) != u:
            self.fa[u] = v

    def cut(self, u, v):
        self.make_root(u)
        self.access(v)
        if self.ch[v][0] == u and self.fa[u] == v and self.ch[u][1] == 0:
            self.ch[v][0] = 0
            self.fa[u] = 0
            self._pushup(v)

    def path_query(self, u, v):
        self.make_root(u)
        self.access(v)
        return self.sm[v]

    def set_val(self, x, value):
        self._splay(x)
        self.val[x] = value
        self._pushup(x)


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    values = [int(next(it)) for _ in range(n)]
    forest = Forest(n, values)
    out = []
    for _ in range(q):
        op = next(it).decode()
        a = int(next(it))
        b = int(next(it))
        if op == "link":
            forest.link(a, b)
        elif op == "cut":
            forest.cut(a, b)
        elif op == "conn":
            out.append("1" if forest.connected(a, b) else "0")
        elif op == "path":
            out.append(str(forest.path_query(a, b)))
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
```

## Complexity - amortized $O(\log n)$ per operation

Every operation reduces to `access` plus $O(1)$ pointer surgery, so it suffices to
bound `access`. Two costs:

- **Splays.** With size $s(x)$ = number of nodes (across real *and* virtual edges)
  below $x$ and potential $\Phi = \sum_x \log s(x)$, the splay access lemma bounds
  one splay by $3(\log s(\text{root}) - \log s(x)) + 1$ amortized.
- **Preferred-child changes** (the length of the climb). Applying the heavy-light
  light-edge bound to the *represented* tree (which `access` does not reshape), a
  root-to-$x$ path makes at most $\log n$ *light* edges preferred; heavy edges
  becoming preferred are amortized against heavy edges becoming unpreferred, each
  charged to a light-edge change. So $O(\log n)$ changes per `access` amortized.

Folding the preferred-child changes into the same $\sum \log s(x)$ potential, the
per-level splay terms telescope to $3(\log s(\text{top}) - \log s(x)) + O(\#\text{changes})
= O(\log n)$ since $s(\text{top}) = n$. `cut` only lowers $\Phi$; `link` raises it
by at most $\log n$. Hence every operation is $O(\log n)$ amortized, and $m$
operations on $n$ nodes run in $O((n + m)\log n)$, with $O(n)$ memory.

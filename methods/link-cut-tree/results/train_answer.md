We are handed a forest of $n$ valued nodes and four operations that arrive online and must each be answered before the next is read: `link(u, v)` joins two nodes currently in different trees, `cut(u, v)` deletes an existing edge, `connected(u, v)` asks whether two nodes share a tree, and a path query returns the aggregate — say the sum — of node values along the unique simple path between $u$ and $v$. The thing that makes this hard is not the queries but the fact that the forest *shape* changes: `link` and `cut` insert and delete edges, so the set of trees and the paths inside them are not fixed in advance, and the path between two connected nodes can be anywhere from one to $n$ nodes long.

On a *fixed* tree I know how to do path aggregates well, and watching exactly where that method breaks is the whole problem. The standard move is a heavy-light decomposition: at every node call the child whose subtree is largest the heavy child and the edge to it heavy, every other child-edge light; following heavy edges gives maximal heavy chains that partition the vertices. Lay each chain out as a contiguous array, put a range-aggregate structure (a segment tree) over each, and any root-to-node path is covered by $O(\log n)$ chain-segments. The reason the count stays small is the halving argument: each time a path steps down a light edge, the subtree it descends into has at most half the size of the one it left — otherwise that child would have been heavy — so a root-to-node path crosses at most $\log n$ light edges, giving $O(\log^2 n)$ per query. It is beautiful, but only on a tree that never changes. The instant I turn on `link` and `cut`, subtree sizes shift and the heavy/light labelling the whole decomposition rests on can flip all over the tree; the array indexing, computed once from a frozen shape, has no meaning after the shape moves. Recomputing the decomposition after every edit is $O(n)$ and defeats the point. So the wall is not the chain idea — it is the *fixedness* of which chains we pick.

What the chains actually bought was this: a path is a handful of contiguous ranges, and a contiguous range in an ordered structure supports fast aggregation. Nothing about that required the chains to be the *heavy* chains specifically; heavy-light was just one static way to choose them. So I propose the Link-Cut Tree: keep the "a path is a few ordered ranges" idea, but make the chains **dynamic**. At every node I allow at most one child to be the *preferred* child, the one this node is chained together with, and I leave that choice mutable. Following preferred edges down from any node gives a *preferred path*, and these paths partition the vertices just as heavy chains did — except now the data structure controls them and can change them on the fly. Call the preferred edges real and the rest virtual.

Each preferred path is a top-to-bottom path in the real tree, and the range-aggregate tool wants it stored as an ordered sequence, so I store each preferred path in a balanced BST — a splay tree — whose in-order order is depth along the path, shallowest (closest to the chain's top) first, deepest last. Then a contiguous depth-range along a path is exactly a splay subtree, and its aggregate sits at that subtree's root, the same payoff a segment tree gave per heavy chain but now per *mutable* path. The pieces must be glued so the whole forest is recoverable: each path has a topmost node whose real-tree parent sits on some other path, so the root of each path's splay tree carries a *path-parent* pointer to that parent. This pointer crosses a virtual edge, and the asymmetry here is load-bearing — the splay root points *up* to its path-parent, but the parent does **not** point back down. The reason is that a node has exactly one parent but can have arbitrarily many virtual children hanging off it; a single up-pointer is $O(1)$ to store and follow, while listing all virtual children would be unbounded. The consequence I lean on constantly: a node $x$ roots its own splay tree exactly when its parent claims neither of its two splay-child slots — that parent is then the invisible path-parent. This is precisely what `_is_root(x)` tests, and it is what distinguishes a real splay edge from a virtual path-parent edge.

Everything reduces to one core primitive, `access(x)`: make the path from the real root down to $x$ a single preferred path / splay tree with $x$ on top, so that path's information lives in one ordered structure. I build it by climbing chain by chain. First splay $x$ to the root of its own splay tree; now its left part is everything above $x$ on this chain and its right part everything below. Since the path goes *up* from $x$, the part below should be cut off — and by the asymmetry rule, detaching it is free: I just overwrite $x$'s right child with the chain I am carrying up, and the old right child, still pointing up to $x$, automatically becomes a virtual child (parent forgot it). Then I move to the path-parent, splay it, graft the carried chain as its right (deeper) subtree because every node in that chain is deeper than the parent, and repeat until I reach a node with no path-parent, the real root. Concretely the loop carries a variable `last` for the chain being lifted: for the current node, splay it, set its right child to `last`, refresh its aggregate, set `last` to it, and move to its path-parent; finish by splaying $x$ back to the top. One pointer write per level performs both the graft and the detach.

With `access` in hand the connectivity query falls out: two nodes share a tree iff they have the same root, and `find_root(x)` is `access(x)` followed by walking to the leftmost (shallowest, since in-order is by depth) node of the splay tree and splaying it to keep the amortized cost honest; then `connected(u, v)` is `find_root(u) == find_root(v)`. But the path query between arbitrary $u$ and $v$ exposes the real subtlety. `access` only ever extracts root-to-node paths, whose depths are monotone; a $u$-to-$v$ path bends at the lowest common ancestor, going up then down, and a non-monotone sequence cannot live in one depth-keyed splay tree at all. The fix is `make_root(u)`: re-root the tree at $u$, so the LCA of $u$ and $v$ becomes $u$ itself, the bend disappears, and the $u$-to-$v$ path becomes a clean root-to-$v$ path that `access(v)` lays out as one chain. Re-rooting at $x$ means flipping the orientation of every edge on the old path from $x$ up to the old root — what was $x$'s parent becomes its child, and so on up the line — and leaving everything else alone. But `access(x)` already hands me exactly that path as a splay tree ordered by depth from the old root (shallowest, leftmost) to $x$ (deepest). Reversing the path's orientation is reversing the *order* of that sequence: the old root must become the deepest and $x$ the shallowest, i.e. the new root. So $\text{make\_root}(x) = \text{access}(x)$ then reverse the whole accessed chain. A plain `access` cannot re-root on its own — it leaves depth order intact, so $x$ stays deepest; the reversal is the extra ingredient.

Reversing a BST's order naively is $O(\text{size})$, which would wreck the bound, so it is done with the classic lazy-tag trick: reversing the in-order sequence of a subtree is the same as swapping every node's left and right child throughout it, deferred. Each node carries a boolean `rev` flag meaning "this subtree is logically reversed, not yet pushed down." To reverse a splay tree at $r$, swap $r$'s two children and toggle `rev[r]`. Whenever I am about to look at a node's children — in a rotation, while descending in `find_root`, or before splaying through it — I first `pushdown`: if `rev` is set, apply the swap one level down to each child, toggle their flags, and clear this node's. Crucially, before splaying a node I must push the pending flags down the entire path from its splay-root to it, top first, or a rotation will move a node whose orientation is unresolved and corrupt the order — which is why `_splay(x)` begins by collecting the root-to-$x$ chain and pushing down from the top.

The remaining operations are then a couple of pointer writes each. `link(u, v)` with $u$ and $v$ in different trees: `make_root(u)` so $u$ becomes a root, then set $u$'s parent pointer to $v$ — a new virtual edge with $v$ not pointing back. `cut(u, v)`: `make_root(u)`, then `access(v)` and splay $v$; if $(u, v)$ is genuinely an edge then $u$ and $v$ are adjacent on the depth-ordered path with $u$ shallower, so $u$ is $v$'s left child and, with nothing between them, $u$ has no right child of its own. That signature — $v$'s left child is $u$ and $u$'s right child is empty — both confirms the edge and lets a `cut` of a non-edge be rejected safely; when it holds, clear $v$'s left child and $u$'s parent to split the trees. `path_query(u, v)`: `make_root(u)`, `access(v)`, after which the $u$-to-$v$ path is the splay tree at $v$ with $v$ on top, so the aggregate stored at $v$ — the sum over its splay subtree — is the sum over the whole path; read it off. Updating a single value just splays the node to the top, changes its value, and refreshes.

Because everything is `access` plus $O(1)$ surgery, the cost analysis reduces to bounding `access`. Two costs appear. The splays obey the standard splay bound: with size $s(x)$ the number of nodes below $x$ across *both* real and virtual edges and potential $\Phi = \sum_x \log s(x)$, the access lemma bounds one splay by $3(\log s(\text{root}) - \log s(x)) + 1$ amortized. The second cost is the number of preferred-child changes per `access`, the length of the climb; reusing the heavy-light light-edge bound on the *represented* tree (which `access` does not reshape), a root-to-$x$ path makes at most $\log n$ light edges preferred, while heavy edges becoming preferred are amortized against heavy edges becoming unpreferred, each charged to a light-edge change — so $O(\log n)$ changes per `access` amortized. Folding the preferred-child changes into the same $\sum \log s(x)$ potential, the per-level splay terms telescope to $3(\log s(\text{top}) - \log s(x)) + O(\#\text{changes}) = O(\log n)$ since $s(\text{top}) = n$. A `cut` only lowers $\Phi$ and a `link` raises it by at most $\log n$, so every operation is $O(\log n)$ amortized, and $m$ operations on $n$ nodes run in $O((n + m)\log n)$ time with $O(n)$ memory — comfortably fast for $n$ and operation counts up to $10^5$. Node $0$ is a null sentinel so missing children never need special-casing, real nodes are $1..n$, the parent array doubles as splay-parent when claimed and path-parent when not, and the pushdown-before-splay loop is kept iterative so a long chain cannot overflow the call stack.

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

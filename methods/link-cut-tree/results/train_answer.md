We are handed a forest of $n$ valued nodes and four operations that arrive online and must each be answered before the next is read: `link(u, v)` joins two nodes currently in different trees, `cut(u, v)` deletes an existing edge, `connected(u, v)` asks whether two nodes share a tree, and a path query returns the aggregate — say the sum — of node values along the unique simple path between $u$ and $v$. The thing that makes this hard is not the queries but the fact that the forest *shape* changes: `link` and `cut` insert and delete edges, so the set of trees and the paths inside them are not fixed in advance, and the path between two connected nodes can be anywhere from one to $n$ nodes long.

On a *fixed* tree I know how to do path aggregates well, and watching exactly where that method breaks is the whole problem. The standard move is a heavy-light decomposition: at every node call the child whose subtree is largest the heavy child and the edge to it heavy, every other child-edge light; following heavy edges gives maximal heavy chains that partition the vertices. Lay each chain out as a contiguous array, put a range-aggregate structure (a segment tree) over each, and any root-to-node path is covered by $O(\log n)$ chain-segments. The reason the count stays small is the halving argument: each time a path steps down a light edge, the subtree it descends into has at most half the size of the one it left — otherwise that child would have been heavy — so a root-to-node path crosses at most $\log n$ light edges, giving $O(\log^2 n)$ per query. It is beautiful, but only on a tree that never changes. The instant I turn on `link` and `cut`, subtree sizes shift and the heavy/light labelling the whole decomposition rests on can flip all over the tree; the array indexing, computed once from a frozen shape, has no meaning after the shape moves. Recomputing the decomposition after every edit is $O(n)$ and defeats the point. So the wall is not the chain idea — it is the *fixedness* of which chains we pick.

What the chains actually bought was this: a path is a handful of contiguous ranges, and a contiguous range in an ordered structure supports fast aggregation. Nothing about that required the chains to be the *heavy* chains specifically; heavy-light was just one static way to choose them. So I propose the Link-Cut Tree: keep the "a path is a few ordered ranges" idea, but make the chains **dynamic**. At every node I allow at most one child to be the *preferred* child, the one this node is chained together with, and I leave that choice mutable. Following preferred edges down from any node gives a *preferred path*, and these paths partition the vertices just as heavy chains did — except now the data structure controls them and can change them on the fly. Call the preferred edges real and the rest virtual.

Each preferred path is a top-to-bottom path in the real tree, and the range-aggregate tool wants it stored as an ordered sequence, so I store each preferred path in a balanced BST — a splay tree — whose in-order order is depth along the path, shallowest (closest to the chain's top) first, deepest last. Then a contiguous depth-range along a path is exactly a splay subtree, and its aggregate sits at that subtree's root, the same payoff a segment tree gave per heavy chain but now per *mutable* path. The pieces must be glued so the whole forest is recoverable: each path has a topmost node whose real-tree parent sits on some other path, so the root of each path's splay tree carries a *path-parent* pointer to that parent. This pointer crosses a virtual edge, and the asymmetry here is load-bearing — the splay root points *up* to its path-parent, but the parent does **not** point back down. The reason is that a node has exactly one parent but can have arbitrarily many virtual children hanging off it; a single up-pointer is $O(1)$ to store and follow, while listing all virtual children would be unbounded. The consequence I lean on constantly: a node $x$ roots its own splay tree exactly when its parent claims neither of its two splay-child slots — that parent is then the invisible path-parent. This is precisely what `_is_root(x)` tests, and it is what distinguishes a real splay edge from a virtual path-parent edge.

Everything reduces to one core primitive, `access(x)`: make the path from the real root down to $x$ a single preferred path / splay tree with $x$ on top, so that path's information lives in one ordered structure. I build it by climbing chain by chain. First splay $x$ to the root of its own splay tree; now its left part is everything above $x$ on this chain and its right part everything below. Since the path goes *up* from $x$, the part below should be cut off — and by the asymmetry rule, detaching it is free: I just overwrite $x$'s right child with the chain I am carrying up, and the old right child, still pointing up to $x$, automatically becomes a virtual child (parent forgot it). Then I move to the path-parent, splay it, graft the carried chain as its right (deeper) subtree because every node in that chain is deeper than the parent, and repeat until I reach a node with no path-parent, the real root. Concretely the loop carries a variable `last` for the chain being lifted: for the current node, splay it, set its right child to `last`, refresh its aggregate, set `last` to it, and move to its path-parent; finish by splaying $x$ back to the top. One pointer write per level performs both the graft and the detach.

With `access` in hand the connectivity query falls out: two nodes share a tree iff they have the same root, and `find_root(x)` is `access(x)` followed by walking to the leftmost (shallowest, since in-order is by depth) node of the splay tree and splaying it to keep the amortized cost honest; then `connected(u, v)` is `find_root(u) == find_root(v)`. But the path query between arbitrary $u$ and $v$ exposes the real subtlety. `access` only ever extracts root-to-node paths, whose depths are monotone; a $u$-to-$v$ path bends at the lowest common ancestor, going up then down, and a non-monotone sequence cannot live in one depth-keyed splay tree at all. The fix is `make_root(u)`: re-root the tree at $u$, so the LCA of $u$ and $v$ becomes $u$ itself, the bend disappears, and the $u$-to-$v$ path becomes a clean root-to-$v$ path that `access(v)` lays out as one chain. Re-rooting at $x$ means flipping the orientation of every edge on the old path from $x$ up to the old root — what was $x$'s parent becomes its child, and so on up the line — and leaving everything else alone. But `access(x)` already hands me exactly that path as a splay tree ordered by depth from the old root (shallowest, leftmost) to $x$ (deepest). Reversing the path's orientation is reversing the *order* of that sequence: the old root must become the deepest and $x$ the shallowest, i.e. the new root. So $\text{make\_root}(x) = \text{access}(x)$ then reverse the whole accessed chain. A plain `access` cannot re-root on its own — it leaves depth order intact, so $x$ stays deepest; the reversal is the extra ingredient.

Reversing a BST's order naively is $O(\text{size})$, which would wreck the bound, so it is done with the classic lazy-tag trick: reversing the in-order sequence of a subtree is the same as swapping every node's left and right child throughout it, deferred. Each node carries a boolean `rev` flag meaning "this subtree is logically reversed, not yet pushed down." To reverse a splay tree at $r$, swap $r$'s two children and toggle `rev[r]`. Whenever I am about to look at a node's children — in a rotation, while descending in `find_root`, or before splaying through it — I first `pushdown`: if `rev` is set, apply the swap one level down to each child, toggle their flags, and clear this node's. Crucially, before splaying a node I must push the pending flags down the entire path from its splay-root to it, top first, or a rotation will move a node whose orientation is unresolved and corrupt the order — which is why `_splay(x)` begins by collecting the root-to-$x$ chain and pushing down from the top.

The remaining operations are then a couple of pointer writes each. `link(u, v)` with $u$ and $v$ in different trees: `make_root(u)` so $u$ becomes a root, then set $u$'s parent pointer to $v$ — a new virtual edge with $v$ not pointing back. `cut(u, v)`: `make_root(u)`, then `access(v)` and splay $v$; if $(u, v)$ is genuinely an edge then $u$ and $v$ are adjacent on the depth-ordered path with $u$ shallower, so $u$ is $v$'s left child and, with nothing between them, $u$ has no right child of its own. That signature — $v$'s left child is $u$ and $u$'s right child is empty — both confirms the edge and lets a `cut` of a non-edge be rejected safely; when it holds, clear $v$'s left child and $u$'s parent to split the trees. `path_query(u, v)`: `make_root(u)`, `access(v)`, after which the $u$-to-$v$ path is the splay tree at $v$ with $v$ on top, so the aggregate stored at $v$ — the sum over its splay subtree — is the sum over the whole path; read it off. Updating a single value just splays the node to the top, changes its value, and refreshes.

Because everything is `access` plus $O(1)$ surgery, the cost analysis reduces to bounding `access`. Two costs appear. The splays obey the standard splay bound: with size $s(x)$ the number of nodes below $x$ across *both* real and virtual edges and potential $\Phi = \sum_x \log s(x)$, the access lemma bounds one splay by $3(\log s(\text{root}) - \log s(x)) + 1$ amortized. The second cost is the number of preferred-child changes per `access`, the length of the climb; reusing the heavy-light light-edge bound on the *represented* tree (which `access` does not reshape), a root-to-$x$ path makes at most $\log n$ light edges preferred, while heavy edges becoming preferred are amortized against heavy edges becoming unpreferred, each charged to a light-edge change — so $O(\log n)$ changes per `access` amortized. Folding the preferred-child changes into the same $\sum \log s(x)$ potential, the per-level splay terms telescope to $3(\log s(\text{top}) - \log s(x)) + O(\#\text{changes}) = O(\log n)$ since $s(\text{top}) = n$. A `cut` only lowers $\Phi$ and a `link` raises it by at most $\log n$, so every operation is $O(\log n)$ amortized, and $m$ operations on $n$ nodes run in $O((n + m)\log n)$ time with $O(n)$ memory — comfortably fast for $n$ and operation counts up to $10^5$. Node $0$ is a null sentinel so missing children never need special-casing, real nodes are $1..n$, the parent array doubles as splay-parent when claimed and path-parent when not, and the pushdown-before-splay loop is kept iterative so a long chain cannot overflow the call stack.

The landing is a single-file C++17 program. It reads `n q`, then `n` node
values, then `q` operations `op a b` with `op` in {`link`, `cut`, `conn`,
`path`}, and prints `0`/`1` for each `conn` and the path-sum for each `path`,
one per line.

```cpp
// Dynamic forest of n valued nodes (link-cut tree).
// Reads: n q, then n node values, then q ops "op a b" with op in
// {link, cut, conn, path}; prints 0/1 for each conn and the path-sum
// for each path, one per line.
#include <array>
#include <cstdio>
#include <string>
#include <vector>
using namespace std;

// Node 0 is a null sentinel; real nodes are 1..n. Each preferred path is a
// splay tree keyed by depth (shallow -> deep); fa[x] doubles as splay-parent
// when claimed and path-parent when not.
struct Forest {
    vector<array<int, 2>> ch;  // splay children: [left, right]
    vector<int> fa;            // splay parent OR path-parent
    vector<long long> val;     // node value
    vector<long long> sm;      // sum of values in this splay subtree
    vector<char> rev;          // lazy "reverse this subtree" flag

    Forest(int n, const vector<long long>& values)
        : ch(n + 1, {0, 0}), fa(n + 1, 0), val(n + 1, 0), sm(n + 1, 0),
          rev(n + 1, 0) {
        for (int i = 1; i <= n; ++i) val[i] = sm[i] = values[i - 1];
    }

    // x roots its own splay tree iff its parent claims neither child slot
    // (then fa[x] is a one-way path-parent: child knows parent, not back).
    bool is_root(int x) const {
        int f = fa[x];
        return ch[f][0] != x && ch[f][1] != x;
    }

    int side(int x) const { return ch[fa[x]][1] == x ? 1 : 0; }

    void pushup(int x) {  // in-order = depth order
        sm[x] = sm[ch[x][0]] + val[x] + sm[ch[x][1]];
    }

    void apply_rev(int x) {  // reverse = swap children + toggle flag
        if (x) {
            int t = ch[x][0];
            ch[x][0] = ch[x][1];
            ch[x][1] = t;
            rev[x] ^= 1;
        }
    }

    void pushdown(int x) {
        if (rev[x]) {
            apply_rev(ch[x][0]);
            apply_rev(ch[x][1]);
            rev[x] = 0;
        }
    }

    void rotate(int x) {
        int y = fa[x], z = fa[y], k = side(x);
        if (!is_root(y)) ch[z][side(y)] = x;  // only relink z if y wasn't a splay root
        fa[x] = z;
        int w = ch[x][k ^ 1];
        ch[y][k] = w;
        if (w) fa[w] = y;
        ch[x][k ^ 1] = y;
        fa[y] = x;
        pushup(y);
        pushup(x);
    }

    void splay(int x) {
        // push pending reversals down the whole root..x chain first, top-down
        static vector<int> stk;
        stk.clear();
        int y = x;
        stk.push_back(y);
        while (!is_root(y)) {
            y = fa[y];
            stk.push_back(y);
        }
        while (!stk.empty()) {
            pushdown(stk.back());
            stk.pop_back();
        }
        while (!is_root(x)) {
            y = fa[x];
            if (!is_root(y)) {
                if (side(x) == side(y))
                    rotate(y);  // zig-zig: rotate the parent first
                else
                    rotate(x);  // zig-zag
            }
            rotate(x);
        }
    }

    // make the path from the real root down to x one splay tree, x on top
    int access(int x) {
        int last = 0, cur = x;
        while (cur) {
            splay(cur);
            ch[cur][1] = last;  // graft the lower chain as the deeper subtree
            pushup(cur);        // (old right child becomes virtual: parent forgets it)
            last = cur;
            cur = fa[cur];
        }
        splay(x);
        return last;
    }

    void make_root(int x) {
        access(x);      // path real-root..x as a chain
        apply_rev(x);   // reverse it: x becomes the new root
    }

    int find_root(int x) {
        access(x);
        while (ch[x][0]) {  // shallowest node = leftmost
            pushdown(x);
            x = ch[x][0];
        }
        splay(x);
        return x;
    }

    bool connected(int u, int v) {
        if (u == v) return true;
        return find_root(u) == find_root(v);
    }

    void link(int u, int v) {  // u, v in different trees
        make_root(u);
        if (find_root(v) != u) fa[u] = v;  // new virtual edge; v doesn't point back
    }

    void cut(int u, int v) {
        make_root(u);
        // after access(v)+splay(v): (u,v) is an edge iff u is v's left child
        // with no right child of its own (nothing between them on the path).
        access(v);
        if (ch[v][0] == u && fa[u] == v && ch[u][1] == 0) {
            ch[v][0] = 0;
            fa[u] = 0;
            pushup(v);
        }
    }

    long long path_query(int u, int v) {
        make_root(u);     // root at u: u..v is now a clean chain
        access(v);        // v on top, sm[v] = sum over the path
        return sm[v];
    }

    void set_val(int x, long long value) {
        splay(x);
        val[x] = value;
        pushup(x);
    }
};

static inline bool read_token(char* buf, int cap) {
    int c = getchar_unlocked();
    while (c == ' ' || c == '\n' || c == '\r' || c == '\t') c = getchar_unlocked();
    if (c == EOF) return false;
    int i = 0;
    while (c != EOF && c != ' ' && c != '\n' && c != '\r' && c != '\t') {
        if (i < cap - 1) buf[i++] = (char)c;
        c = getchar_unlocked();
    }
    buf[i] = '\0';
    return i > 0;
}

static inline bool read_ll(long long& out) {
    char buf[32];
    if (!read_token(buf, sizeof(buf))) return false;
    long long sign = 1, v = 0;
    const char* p = buf;
    if (*p == '-') { sign = -1; ++p; }
    while (*p) { v = v * 10 + (*p - '0'); ++p; }
    out = sign * v;
    return true;
}

int main() {
    long long n_ll, q_ll;
    if (!read_ll(n_ll)) return 0;
    if (!read_ll(q_ll)) return 0;
    int n = (int)n_ll, q = (int)q_ll;
    vector<long long> values(n);
    for (int i = 0; i < n; ++i) read_ll(values[i]);
    Forest forest(n, values);
    string out;
    char op[16];
    long long a, b;
    for (int i = 0; i < q; ++i) {
        if (!read_token(op, sizeof(op))) break;
        read_ll(a);
        read_ll(b);
        int ai = (int)a, bi = (int)b;
        if (op[0] == 'l') {  // link
            forest.link(ai, bi);
        } else if (op[0] == 'c' && op[1] == 'u') {  // cut
            forest.cut(ai, bi);
        } else if (op[0] == 'c' && op[1] == 'o') {  // conn
            out += forest.connected(ai, bi) ? '1' : '0';
            out += '\n';
        } else if (op[0] == 'p') {  // path
            out += to_string(forest.path_query(ai, bi));
            out += '\n';
        }
    }
    if (!out.empty() && out.back() == '\n') out.pop_back();
    fputs(out.c_str(), stdout);
    return 0;
}
```

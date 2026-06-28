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

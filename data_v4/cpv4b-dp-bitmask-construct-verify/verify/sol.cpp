#include <bits/stdc++.h>
using namespace std;

/*
  Lexicographically-smallest binary tape of length n that contains NO blacklisted
  window. Each of the m blacklisted patterns is a binary string of length L.
  '0' < '1'. Output the tape, or -1 if no safe tape of length n exists.

  WINDOW AUTOMATON.
  State = the last (L-1) emitted bits, packed as a mask in [0, S), S = 2^(L-1).
  Appending bit b to state s forms the full L-window  W = (s << 1) | b  (L bits).
  The edge (s --b--> ((s<<1)|b)&(S-1)) is allowed iff W is NOT blacklisted.

  Backward feasibility g[r][s] = "from state s, with r MORE window-completing
  positions to emit, can we legally finish?":
      g[0][s] = true                                   (nothing left)
      g[r][s] = OR_b ( edge(s,b) allowed AND g[r-1][edge(s,b)] )
  g_r is MONOTONE NON-INCREASING in r: any legal length-r walk has a legal
  length-(r-1) suffix walk, so surviving horizon r implies surviving r-1. The
  lattice of state-subsets has height S, so g_r is CONSTANT for r >= S. We compute
  r = 0..S and reuse g[S] for larger horizons.  With L <= 12, S <= 2048.

  PREFIX (first L-1 positions) carries no full window yet, but its bits fix the
  entry state, so feasibility there is NOT free. We fold the prefix into the same
  backward recursion via a tiny recursion over the (L-1) free bits.
*/

int L_, M_, S_;
long long N_;
vector<char> blacklisted;            // size 1<<L : is this L-bit pattern forbidden?
vector<array<int,2>> edge_;          // window-regime edges (-1 if blocked)
vector<vector<char>> g_;             // g[r][s], r = 0..S

static inline bool canFinishWindow(long long r, int s) {
    if (r < 0) return false;
    if (r >= (long long)S_) return g_[S_][s] != 0;
    return g_[(int)r][s] != 0;
}

// Feasibility from a prefix of j chosen bits (j < L-1), partial mask s of j bits.
// (L-1-j) free prefix bits remain, then the window regime. L tiny -> <=2^(L-1).
static bool feasFromPrefix(int j, int s) {
    if (j == L_ - 1) {
        long long windows = N_ - (L_ - 1);          // window-completing positions
        return canFinishWindow(windows, s & (S_ - 1));
    }
    for (int b = 0; b < 2; b++)
        if (feasFromPrefix(j + 1, (s << 1) | b)) return true;
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n; int L, m;
    if (!(cin >> n >> L >> m)) return 0;
    N_ = n; L_ = L; M_ = m;

    blacklisted.assign(1 << L, 0);
    for (int i = 0; i < m; i++) {
        string p; cin >> p;            // length-L binary string
        int v = 0;
        for (char c : p) v = (v << 1) | (c == '1' ? 1 : 0);
        blacklisted[v] = 1;
    }

    // Fewer than L bits: no window of length L -> all '0' is the smallest safe tape.
    if (n < L) {
        for (long long i = 0; i < n; i++) cout << '0';
        cout << "\n";
        return 0;
    }

    S_ = 1 << (L - 1);

    edge_.assign(S_, {-1, -1});
    for (int s = 0; s < S_; s++)
        for (int b = 0; b < 2; b++) {
            int window = (s << 1) | b;                 // L-bit window value
            if (blacklisted[window]) { edge_[s][b] = -1; continue; }
            edge_[s][b] = ((s << 1) | b) & (S_ - 1);
        }

    g_.assign(S_ + 1, vector<char>(S_, 0));
    for (int s = 0; s < S_; s++) g_[0][s] = 1;
    for (int r = 1; r <= S_; r++)
        for (int s = 0; s < S_; s++) {
            char ok = 0;
            for (int b = 0; b < 2 && !ok; b++) {
                int ns = edge_[s][b];
                if (ns >= 0 && g_[r - 1][ns]) ok = 1;
            }
            g_[r][s] = ok;
        }

    if (!feasFromPrefix(0, 0)) { cout << -1 << "\n"; return 0; }

    string out;
    out.reserve((size_t)n);
    int partial = 0;     // bits chosen so far (low bits = most recent)
    int placed = 0;

    for (long long i = 0; i < n; i++) {
        bool chose = false;
        for (int b = 0; b < 2; b++) {           // '0' first
            if (placed < L - 1) {
                int ns = (partial << 1) | b;     // (j+1)-bit partial mask
                if (feasFromPrefix(placed + 1, ns)) {
                    out.push_back(b ? '1' : '0');
                    partial = ns; placed++; chose = true; break;
                }
            } else {
                int s = partial & (S_ - 1);
                int e = edge_[s][b];
                if (e < 0) continue;
                long long remAfter = (n - i - 1);  // later positions are window bits
                if (canFinishWindow(remAfter, e)) {
                    out.push_back(b ? '1' : '0');
                    partial = e; placed++; chose = true; break;
                }
            }
        }
        if (!chose) { cout << -1 << "\n"; return 0; }  // unreachable after pre-check
    }

    cout << out << "\n";
    return 0;
}

// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// Naive cell-by-cell backward substitution: to go from layer t to layer t-1,
// for every cell independently pick the center bit that reproduces the
// desired value GIVEN THE NEIGHBORS AS THEY CURRENTLY STAND IN LAYER t (i.e.
// treats the not-yet-solved layer t-1 neighbors as if they equalled their own
// layer-t values). This ignores that neighbors' own predecessors are being
// solved simultaneously -- the classic "cell-by-cell reversal" trap: it never
// checks whether its independent per-cell guesses are mutually consistent,
// so on a rule that genuinely couples neighborhoods the error compounds every
// step. On a tie (both center bits reproduce the desired value) it keeps
// the cell's current value rather than exploiting the free bit to save cost.

int N, T;
char rule[32];

int main(){
    if (!(cin >> N >> T)) return 0;
    long long alpha, beta; cin >> alpha >> beta;
    string ruleTok; cin >> ruleTok;
    for (int v = 0; v < 32; v++) rule[v] = ruleTok[v];

    vector<string> layer(N);
    for (int i = 0; i < N; i++) cin >> layer[i];

    vector<string> nextLayer(N, string(N, '0'));
    for (int s = 0; s < T; s++){
        nextLayer = layer;
        for (int i = 0; i < N; i++){
            int up = (i - 1 + N) % N, down = (i + 1) % N;
            for (int j = 0; j < N; j++){
                int left = (j - 1 + N) % N, right = (j + 1) % N;
                int u = layer[up][j] - '0';
                int d = layer[down][j] - '0';
                int l = layer[i][left] - '0';
                int r = layer[i][right] - '0';
                int want = layer[i][j] - '0';
                int v0 = (0 << 4) | (u << 3) | (d << 2) | (l << 1) | r;
                int v1 = (1 << 4) | (u << 3) | (d << 2) | (l << 1) | r;
                bool ok0 = (rule[v0] - '0') == want;
                bool ok1 = (rule[v1] - '0') == want;
                char chosen;
                if (ok0 && ok1) chosen = layer[i][j];      // tie: no cost-saving insight
                else if (ok0) chosen = '0';
                else if (ok1) chosen = '1';
                else chosen = layer[i][j];                  // no local solution: best-effort fallback
                nextLayer[i][j] = chosen;
            }
        }
        layer = nextLayer;
    }

    for (int i = 0; i < N; i++) cout << layer[i] << "\n";
    return 0;
}

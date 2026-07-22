// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Insight: (1) the backward pass exploits the rule table's DON'T-CARE / local
// injectivity structure -- when both center values reproduce the wanted
// output, that bit is genuinely free, so we zero it (saves live-cell cost
// instead of blindly copying, unlike the naive greedy). (2) starting from
// that warm start, we run bounded local search that respects the T-step
// NEIGHBORHOOD COUPLING: each candidate flip is evaluated by re-simulating a
// small local window (the flip can only influence cells within a bounded
// distance after T steps), so the decision accounts for how the flip actually
// propagates, instead of a single myopic backward guess. This is a genuinely
// different algorithm from "greedy plus more iterations": it is a
// constraint-propagation-guided repair loop over the true multi-step map.

static int N, T;
static long long alphaW, betaW;
static char rule[32];
static vector<string> target;
static vector<string> initG;

static int WIN, HALF;
static vector<vector<char>> curW, nxtW;

void buildWindow(int i0, int j0, bool flipCenter){
    for (int a = 0; a < WIN; a++){
        int gi = ((i0 - HALF + a) % N + N) % N;
        for (int b = 0; b < WIN; b++){
            int gj = ((j0 - HALF + b) % N + N) % N;
            curW[a][b] = initG[gi][gj];
        }
    }
    if (flipCenter){
        char &c = curW[HALF][HALF];
        c = (c == '0') ? '1' : '0';
    }
}

// The window is simulated as its own small torus (an approximation: it
// ignores true grid data beyond the window). This is fine for a heuristic
// local-search decision -- the checker always scores the FULL global
// simulation, this local view only informs which flips we try.
void simulateWindow(){
    for (int s = 0; s < T; s++){
        for (int a = 0; a < WIN; a++){
            int up = (a - 1 + WIN) % WIN, down = (a + 1) % WIN;
            for (int b = 0; b < WIN; b++){
                int left = (b - 1 + WIN) % WIN, right = (b + 1) % WIN;
                int c = curW[a][b] - '0';
                int u = curW[up][b] - '0';
                int d = curW[down][b] - '0';
                int l = curW[a][left] - '0';
                int r = curW[a][right] - '0';
                int v = (c << 4) | (u << 3) | (d << 2) | (l << 1) | r;
                nxtW[a][b] = rule[v];
            }
        }
        swap(curW, nxtW);
    }
}

long long windowMismatch(int i0, int j0){
    long long m = 0;
    for (int a = 0; a < WIN; a++){
        int gi = ((i0 - HALF + a) % N + N) % N;
        for (int b = 0; b < WIN; b++){
            int gj = ((j0 - HALF + b) % N + N) % N;
            if (curW[a][b] != target[gi][gj]) m++;
        }
    }
    return m;
}

long long localMismatch(int i0, int j0, bool flip){
    buildWindow(i0, j0, flip);
    simulateWindow();
    return windowMismatch(i0, j0);
}

int main(){
    if (!(cin >> N >> T)) return 0;
    cin >> alphaW >> betaW;
    string ruleTok; cin >> ruleTok;
    for (int v = 0; v < 32; v++) rule[v] = ruleTok[v];

    target.resize(N);
    for (int i = 0; i < N; i++) cin >> target[i];

    // ---- warm start: cost-aware backward substitution ----
    vector<string> layer = target;
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
                if (ok0 && ok1) chosen = '0';               // don't-care: exploit it, save cost
                else if (ok0) chosen = '0';
                else if (ok1) chosen = '1';
                else chosen = layer[i][j];                   // no local solution: best-effort fallback
                nextLayer[i][j] = chosen;
            }
        }
        layer = nextLayer;
    }
    initG = layer;

    // ---- bounded local search with windowed re-simulation ----
    HALF = T + 2;
    WIN = 2 * HALF + 1;
    curW.assign(WIN, vector<char>(WIN, '0'));
    nxtW.assign(WIN, vector<char>(WIN, '0'));

    // cap total flip-evaluation work so this stays fast for any N,T in range
    long long budget = 6000000LL;
    long long perSweepCost = (long long)N * N;
    int sweeps = 3;
    if (perSweepCost > 0){
        long long maxSweeps = budget / max(1LL, perSweepCost);
        sweeps = (int)max(1LL, min((long long)sweeps, maxSweeps));
    }

    for (int sw = 0; sw < sweeps; sw++){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                long long before = localMismatch(i, j, false);
                long long after = localMismatch(i, j, true);
                long long dH = after - before;
                long long dL = (initG[i][j] == '1') ? -1 : 1;
                long long delta = alphaW * dH + betaW * dL;
                if (delta < 0) initG[i][j] = (initG[i][j] == '0') ? '1' : '0';
            }
        }
    }

    // final cost-pruning pass: drop any live cell whose removal doesn't hurt
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (initG[i][j] != '1') continue;
            long long before = localMismatch(i, j, false);
            long long after = localMismatch(i, j, true); // toggling a '1' turns it OFF
            long long dH = after - before;
            long long delta = alphaW * dH - betaW;
            if (delta <= 0) initG[i][j] = '0';
        }
    }

    for (int i = 0; i < N; i++) cout << initG[i] << "\n";
    return 0;
}

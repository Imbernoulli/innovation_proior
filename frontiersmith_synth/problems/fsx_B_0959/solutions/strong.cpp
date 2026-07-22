// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Simulation-guided local search. Instead of trusting an aggregate stat, we
// directly evaluate the true bracket objective F on candidate seedings and
// hill-climb via random slot swaps, with periodic random restarts to escape
// local optima. This is what actually exploits the innovation hook: it can
// discover, for each sponsor, a chain of SPECIFIC beatable opponents round by
// round (invisible to a win-count heuristic) and it directly arbitrates which
// sponsor gets the deep run when two sponsor paths would otherwise collide,
// because collisions show up as a real drop in F, not as an assumption.

int N, K;
vector<string> M;
vector<int> sid, sval;
vector<char> isSponsor;

static inline bool beats(int a, int b){ return M[a][b - 1] == '1'; }

static ll simulate(const vector<int>& start){
    vector<int> cur = start;
    vector<ll> roundsWon(N + 1, 0);
    int r = 0;
    while ((int)cur.size() > 1){
        r++;
        int half = (int)cur.size() / 2;
        vector<int> nxt(half);
        for (int j = 0; j < half; j++){
            int a = cur[2 * j], b = cur[2 * j + 1];
            int w = beats(a, b) ? a : b;
            nxt[j] = w;
            if (isSponsor[w]) roundsWon[w] = r;
        }
        cur.swap(nxt);
    }
    ll F = 0;
    for (int i = 0; i < K; i++){
        ll rw = roundsWon[sid[i]];
        F += (ll)sval[i] * (rw + 1);
    }
    return F;
}

int main(){
    cin >> N >> K;
    M.assign(N + 1, string());
    for (int i = 1; i <= N; i++) cin >> M[i];
    sid.assign(K, 0); for (auto &x : sid) cin >> x;
    sval.assign(K, 0); for (auto &x : sval) cin >> x;
    isSponsor.assign(N + 1, 0);
    for (int id : sid) isSponsor[id] = 1;

    // greedy warm start (win-count based classic seeding) -- cheap, gives the
    // local search a reasonable place to begin from among several starts
    vector<int> winCount(N + 1, 0);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            if (i != j && M[i][j - 1] == '1') winCount[i]++;
    vector<int> greedyPerm;
    {
        vector<int> sponsorsSorted = sid, others;
        vector<int> bountyOf(N + 1, 0);
        for (int i = 0; i < K; i++) bountyOf[sid[i]] = sval[i];
        sort(sponsorsSorted.begin(), sponsorsSorted.end(), [&](int a, int b){
            if (bountyOf[a] != bountyOf[b]) return bountyOf[a] > bountyOf[b];
            return a < b;
        });
        for (int i = 1; i <= N; i++) if (!isSponsor[i]) others.push_back(i);
        sort(others.begin(), others.end(), [&](int a, int b){
            if (winCount[a] != winCount[b]) return winCount[a] > winCount[b];
            return a < b;
        });
        vector<int> S;
        for (int x : sponsorsSorted) S.push_back(x);
        for (int x : others) S.push_back(x);
        function<vector<int>(int)> seedOrder = [&](int n) -> vector<int> {
            if (n == 1) return {1};
            vector<int> prev = seedOrder(n / 2), res;
            for (int x : prev){ res.push_back(x); res.push_back(n + 1 - x); }
            return res;
        };
        vector<int> order = seedOrder(N);
        greedyPerm.assign(N, 0);
        for (int i = 0; i < N; i++) greedyPerm[i] = S[order[i] - 1];
    }

    mt19937 rng(998244353u ^ (unsigned)(N * 2654435761u) ^ (unsigned)(K * 40503u));

    ll iters = min(150000LL, 30000000LL / max(1, N));
    ll restartEvery = max(200LL, iters / 6);

    vector<int> best = greedyPerm;
    ll bestF = simulate(best);

    vector<int> cur = greedyPerm;
    ll curF = bestF;
    for (ll t = 0; t < iters; t++){
        if (restartEvery > 0 && t > 0 && t % restartEvery == 0){
            // random restart: fresh random permutation
            cur.assign(N, 0);
            for (int i = 0; i < N; i++) cur[i] = i + 1;
            for (int i = N - 1; i > 0; i--){
                int j = rng() % (i + 1);
                swap(cur[i], cur[j]);
            }
            curF = simulate(cur);
            if (curF > bestF){ bestF = curF; best = cur; }
        }
        int i = rng() % N, j = rng() % N;
        if (i == j) continue;
        swap(cur[i], cur[j]);
        ll nf = simulate(cur);
        if (nf >= curF){
            curF = nf;
            if (curF > bestF){ bestF = curF; best = cur; }
        } else {
            swap(cur[i], cur[j]); // revert
        }
    }

    for (int i = 0; i < N; i++) cout << best[i] << (i + 1 == N ? '\n' : ' ');
    return 0;
}

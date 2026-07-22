// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious first attempt: make sure every sponsor at least SURVIVES round 1
// by hand-picking, for each sponsor (highest bounty first), a distinct
// not-yet-used opponent it directly beats (preferring the weakest-looking
// candidate by aggregate win count, which "sounds like" a sensible tie-break).
// Everything past round 1 is left to chance -- the remaining players (and the
// pairs already formed) are simply packed into the rest of the bracket by
// aggregate win count, with no thought to who a sponsor's winner-of-the-other-
// pair opponent will be in round 2, round 3, ... This is exactly a single
// greedy pass with no multi-round lookahead, so it is still blind to a chain
// of round-2+ specialists, to near-regular (circulant) tournaments where win
// count carries no signal at all, and to two sponsors whose "easy" round-1
// picks quietly set them on a collision course two rounds later.

int N, K;
vector<string> M;

static inline bool beats(int a, int b){ return M[a][b - 1] == '1'; }

int main(){
    cin >> N >> K;
    M.assign(N + 1, string());
    for (int i = 1; i <= N; i++) cin >> M[i];
    vector<int> sid(K); for (auto &x : sid) cin >> x;
    vector<int> sval(K); for (auto &x : sval) cin >> x;

    vector<char> isSponsor(N + 1, 0);
    vector<int> bountyOf(N + 1, 0);
    for (int i = 0; i < K; i++) { isSponsor[sid[i]] = 1; bountyOf[sid[i]] = sval[i]; }

    vector<int> winCount(N + 1, 0);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            if (i != j && M[i][j - 1] == '1') winCount[i]++;

    vector<int> sponsorsSorted = sid;
    sort(sponsorsSorted.begin(), sponsorsSorted.end(), [&](int a, int b){
        if (bountyOf[a] != bountyOf[b]) return bountyOf[a] > bountyOf[b];
        return a < b;
    });

    vector<char> used(N + 1, 0);
    // players ordered by ascending win count (weakest-looking first), the pool
    // greedy scans when looking for an "easy" round-1 opponent
    vector<int> byWeak(N);
    for (int i = 1; i <= N; i++) byWeak[i - 1] = i;
    sort(byWeak.begin(), byWeak.end(), [&](int a, int b){
        if (winCount[a] != winCount[b]) return winCount[a] < winCount[b];
        return a < b;
    });

    vector<pair<int,int>> pairs; // round-1 pairs, in construction order
    for (int s : sponsorsSorted) {
        if (used[s]) continue; // (can't happen: sponsors are distinct, unused so far)
        used[s] = 1;
        int chosen = -1;
        for (int cand : byWeak) {
            if (cand == s || used[cand]) continue;
            if (beats(s, cand)) { chosen = cand; break; }
        }
        if (chosen == -1) { // no beatable candidate found at all: fall back to weakest remaining
            for (int cand : byWeak) { if (cand != s && !used[cand]) { chosen = cand; break; } }
        }
        used[chosen] = 1;
        pairs.push_back({s, chosen});
    }
    // pack the remaining players into round-1 pairs by aggregate win count,
    // strongest-vs-weakest within the leftover pool (the textbook "spread
    // apart" instinct, applied with zero regard for what happens beyond
    // round 1 or for the pairs already fixed above)
    vector<int> rest;
    for (int i = 1; i <= N; i++) if (!used[i]) rest.push_back(i);
    sort(rest.begin(), rest.end(), [&](int a, int b){
        if (winCount[a] != winCount[b]) return winCount[a] > winCount[b];
        return a < b;
    });
    {
        int lo = 0, hi = (int)rest.size() - 1;
        while (lo < hi) { pairs.push_back({rest[lo], rest[hi]}); lo++; hi--; }
    }

    vector<int> perm;
    perm.reserve(N);
    for (auto &pr : pairs) { perm.push_back(pr.first); perm.push_back(pr.second); }

    for (int i = 0; i < N; i++) cout << perm[i] << (i + 1 == N ? '\n' : ' ');
    return 0;
}

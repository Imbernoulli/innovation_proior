// TIER: strong
// Insight #1 (exploits fold-order-commit + access-blocking-order): value is
// weight * commit-step, so the natural "take the biggest exposed weight now"
// instinct is backwards -- commit the SMALLER of the two currently-exposed
// frontier weights first, saving big weights for LATE (high-multiplier) steps.
// This single-pass frontier-min sweep is a genuine exchange-argument reversal
// of the greedy tier, not "greedy plus more iterations".
//
// Insight #2 (exploits layer-parity-invariant): folding everything with req_i
// only hits the exact M/V target T by luck; since the WHOLE score is halved on
// a miss, we actively repair it. Two candidate plans are evaluated and the
// better one is kept:
//   Plan A: commit all n-1 creases (the full sweep order), then flip the
//           minimum-cost set of directions (smallest weight*step first) to
//           land exactly on Mc-Vc=T if that parity is reachable.
//   Plan B: same, but first drop the LAST commit in the sweep (always safe --
//           nothing later in the sequence can depend on the very last step),
//           which flips k's parity and can unlock an exact-T match that Plan A
//           cannot reach.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static void bestFlips(const vector<ll>& order, const vector<char>& req,
                       const vector<ll>& w, ll T,
                       vector<char>& outDir, ll& outRaw, ll& outF){
    outDir.resize(order.size());
    for (size_t idx = 0; idx < order.size(); idx++) outDir[idx] = req[order[idx]];
    ll gap = 0;
    for (char d : outDir) gap += (d == 'M') ? 1 : -1;
    ll diff = T - gap;
    if (diff != 0 && (diff % 2 == 0)){
        ll numFlips = llabs(diff) / 2;
        char wantFrom = (diff > 0) ? 'V' : 'M';
        vector<pair<ll,int>> cand;
        for (size_t idx = 0; idx < order.size(); idx++)
            if (outDir[idx] == wantFrom)
                cand.push_back({w[order[idx]] * (ll)(idx + 1), (int)idx});
        sort(cand.begin(), cand.end());
        ll apply = min<ll>(numFlips, (ll)cand.size());
        for (ll f = 0; f < apply; f++){
            int idx = cand[f].second;
            outDir[idx] = (outDir[idx] == 'M') ? 'V' : 'M';
        }
        gap = 0;
        for (char d : outDir) gap += (d == 'M') ? 1 : -1;
    }
    ll raw = 0;
    for (size_t idx = 0; idx < order.size(); idx++){
        ll step = (ll)idx + 1;
        ll match = (outDir[idx] == req[order[idx]]) ? 2 : 1;
        raw += w[order[idx]] * step * match;
    }
    ll gf = (gap == T) ? 2 : 1;
    outRaw = raw;
    outF = raw * gf;
}

int main(){
    ll n, T;
    if (scanf("%lld %lld", &n, &T) != 2) return 0;
    ll km1 = n - 1;
    vector<char> req(km1 + 1);
    for (ll i = 1; i <= km1; i++){
        char buf[4]; scanf("%s", buf); req[i] = buf[0];
    }
    vector<ll> w(km1 + 1);
    for (ll i = 1; i <= km1; i++) scanf("%lld", &w[i]);

    // Insight #1: frontier-min sweep -> a chronological commit order.
    vector<ll> order;
    order.reserve(km1);
    ll L = 1, R = km1;
    while (L <= R){
        if (L == R){ order.push_back(L); break; }
        if (w[L] <= w[R]){ order.push_back(L); L++; }
        else { order.push_back(R); R--; }
    }

    // Plan A: full order.
    vector<char> dirA; ll rawA, FA;
    bestFlips(order, req, w, T, dirA, rawA, FA);

    // Plan B: drop the LAST commit (always safe -- nothing depends on it).
    vector<ll> orderB;
    if (!order.empty()) orderB.assign(order.begin(), order.end() - 1);
    vector<char> dirB; ll rawB = 0, FB = 0;
    bestFlips(orderB, req, w, T, dirB, rawB, FB);

    bool useB = FB > FA;
    const vector<ll>& finalOrder = useB ? orderB : order;
    const vector<char>& finalDir = useB ? dirB : dirA;

    printf("%zu\n", finalOrder.size());
    for (size_t idx = 0; idx < finalOrder.size(); idx++)
        printf("%lld %c\n", finalOrder[idx], finalDir[idx]);
    return 0;
}

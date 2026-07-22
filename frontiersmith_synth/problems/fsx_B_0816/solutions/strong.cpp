// TIER: strong
// Insight: build a small, capacity-aware PALETTE of shared intermediate wells
// (a coarse concentration grid plus extra points at the input's most frequent
// target concentrations), then route each target well through ONE exact-match
// palette instance or a two-point blend of the nearest bracketing instances
// (inverting the linear-mix equation to find the split volumes). A palette
// instance is only ever built once per Vcap's worth of downstream demand, so
// the number of DISTINCT wells that ever touch the stock reservoir (D) stays
// roughly proportional to the number of concentrations actually needed, not W.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Op { ll src, dst, vol; };

static void chunkFill(vector<Op>& ops, ll src, ll dst, ll vol, ll Vmax){
    while (vol > 0){
        ll chunk = min(Vmax, vol);
        ops.push_back({src, dst, chunk});
        vol -= chunk;
    }
}

int main(){
    ll W, M, Vmax, Vcap, stepCost, stockAccessCost, maxOps;
    cin >> W >> M >> Vmax >> Vcap >> stepCost >> stockAccessCost >> maxOps;
    ll N = W + M;
    vector<ll> c(W + 1), Vreq(W + 1);
    for (ll i = 1; i <= W; i++) cin >> c[i] >> Vreq[i];

    // ---- build palette: coarse grid in [20,980] + top frequent cluster buckets ----
    int NGRID = (int)max(2LL, min(19LL, W / 4));
    vector<ll> palette;
    if (NGRID == 1) palette.push_back(500);
    else {
        for (int k = 0; k < NGRID; k++){
            ll p = 20 + (ll)llround(k * 960.0 / (NGRID - 1));
            palette.push_back(p);
        }
    }
    // frequency of rounded (bucket width 20) target concentrations
    map<ll, int> freq;
    for (ll i = 1; i <= W; i++){
        ll b = ((c[i] + 10) / 20) * 20;
        if (b < 20) b = 20;
        if (b > 980) b = 980;
        freq[b]++;
    }
    vector<pair<int,ll>> byFreq;
    for (auto& kv : freq) byFreq.push_back({kv.second, kv.first});
    sort(byFreq.begin(), byFreq.end(), greater<pair<int,ll>>());
    int NCUST = (int)max(0LL, min(6LL, W / 10));
    int minFreq = max(2, (int)(W / 50));
    int added = 0;
    for (auto& pr : byFreq){
        if (added >= NCUST) break;
        if (pr.first < minFreq) break;
        ll val = pr.second;
        bool close = false;
        for (ll p : palette) if (llabs(p - val) <= 15) { close = true; break; }
        if (!close){ palette.push_back(val); added++; }
    }
    sort(palette.begin(), palette.end());
    palette.erase(unique(palette.begin(), palette.end()), palette.end());

    // candidate bracket points include the palette plus the free endpoints 0/1000
    vector<ll> allPts;
    allPts.push_back(0);
    for (ll p : palette) allPts.push_back(p);
    allPts.push_back(1000);

    const ll EPS = 5; // near-exact single-source tolerance (per-mille)

    // request: (paletteOrEndpointValue, volume) per well
    struct Req { ll val; ll vol; };
    vector<vector<Req>> wellReqs(W + 1);

    for (ll i = 1; i <= W; i++){
        ll t = c[i];
        // try near-exact single source among interior palette points
        ll best = -1, bestd = EPS + 1;
        for (ll p : palette){
            ll d = llabs(p - t);
            if (d < bestd){ bestd = d; best = p; }
        }
        if (best != -1 && bestd <= EPS){
            wellReqs[i].push_back({best, Vreq[i]});
            continue;
        }
        // bracket among allPts (sorted)
        ll lo = 0, hi = 1000;
        for (ll p : allPts){
            if (p <= t) lo = max(lo, p);
            if (p >= t) { hi = p; break; }
        }
        if (lo == hi){
            wellReqs[i].push_back({lo, Vreq[i]});
            continue;
        }
        double frac = (double)(t - lo) / (double)(hi - lo);
        ll vHi = (ll)llround(frac * (double)Vreq[i]);
        if (vHi < 0) vHi = 0;
        if (vHi > Vreq[i]) vHi = Vreq[i];
        ll vLo = Vreq[i] - vHi;
        if (vLo > 0) wellReqs[i].push_back({lo, vLo});
        if (vHi > 0) wellReqs[i].push_back({hi, vHi});
    }

    // ---- bin-pack demand for each interior palette value into capacity-Vcap instances ----
    ll nextScratch = W + 1;
    map<ll, vector<pair<ll,ll>>> instances; // paletteValue -> list of (wellId, usedSoFar)
    // assignment[i] = list of (instanceWellId or -1/-2, vol)
    vector<vector<pair<ll,ll>>> assign(W + 1);

    for (ll i = 1; i <= W; i++){
        for (auto& r : wellReqs[i]){
            if (r.val == 0){ assign[i].push_back({-2, r.vol}); continue; }
            if (r.val == 1000){ assign[i].push_back({-1, r.vol}); continue; }
            auto& lst = instances[r.val];
            if (lst.empty() || lst.back().second + r.vol > Vcap){
                if (nextScratch > N){
                    // ran out of scratch wells: fall back to direct stock+diluent for this draw
                    ll stockV = (r.vol * r.val) / 1000;
                    ll dilV = r.vol - stockV;
                    if (stockV > 0) assign[i].push_back({-1, stockV});
                    if (dilV > 0) assign[i].push_back({-2, dilV});
                    continue;
                }
                lst.push_back({nextScratch, 0});
                nextScratch++;
            }
            lst.back().second += r.vol;
            assign[i].push_back({lst.back().first, r.vol});
        }
    }

    // ---- phase 1: build every instance to its final required volume ----
    vector<Op> ops;
    for (auto& kv : instances){
        ll pval = kv.first;
        for (auto& inst : kv.second){
            ll wellId = inst.first, used = inst.second;
            if (used <= 0) continue;
            ll stockV = (used * pval) / 1000;
            ll dilV = used - stockV;
            if (stockV > 0) chunkFill(ops, -1, wellId, stockV, Vmax);
            if (dilV > 0) chunkFill(ops, -2, wellId, dilV, Vmax);
        }
    }

    // ---- phase 2: draw each well's requests from its assigned source ----
    for (ll i = 1; i <= W; i++){
        for (auto& a : assign[i]){
            if (a.second > 0) chunkFill(ops, a.first, i, a.second, Vmax);
        }
    }

    // safety: respect maxOps (should not trigger given generous generator bound)
    if ((ll)ops.size() > maxOps) ops.resize(maxOps);

    cout << ops.size() << "\n";
    for (auto& o : ops) cout << o.src << " " << o.dst << " " << o.vol << "\n";
    return 0;
}

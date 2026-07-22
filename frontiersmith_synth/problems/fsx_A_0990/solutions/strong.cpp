// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// The insight: instead of ranking substrings by pooled (summed) frequency, rank them by
// the MINIMUM across the 3 domains of (per-domain count * (length - token cost) / that
// domain's raw length) -- the worst-case, size-normalized contribution. This automatically
// deprioritizes any substring that is absent (or rare) in even one domain -- which is
// exactly what happens to a domain-exclusive whole word -- and surfaces the shorter
// cross-domain stems instead. As a further pruning step (and because the objective is
// itself a worst-corpus minimum), only substrings that actually occur in the SMALLEST
// corpus are worth considering at all: anything absent there scores 0.
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int K; long long Dmax;
    cin >> K >> Dmax;
    vector<string> corpus(K);
    vector<long long> raw(K);
    for(int d=0; d<K; d++) cin >> raw[d] >> corpus[d];

    const int LMIN = 3, LMAX = 12, TOKEN_COST = 2;

    int small = 0;
    for(int d=1; d<K; d++) if(raw[d] < raw[small]) small = d;

    vector<pair<double,string>> cands;
    cands.reserve(1<<18);
    for(int L=LMIN; L<=LMAX; L++){
        vector<unordered_map<string,long long>> cnt(K);
        for(int d=0; d<K; d++){
            cnt[d].reserve(1<<15);
            const string& T = corpus[d];
            int n = (int)T.size();
            for(int i=0; i+L<=n; i++) cnt[d][T.substr(i,L)]++;
        }
        for(auto& kv : cnt[small]){
            const string& s = kv.first;
            double worst = 1e18;
            for(int d=0; d<K; d++){
                long long c = 0;
                auto it = cnt[d].find(s);
                if(it != cnt[d].end()) c = it->second;
                double contrib = (double)c * (double)(L - TOKEN_COST) / (double)raw[d];
                worst = min(worst, contrib);
            }
            if(worst > 0.0) cands.push_back({worst, s});
        }
    }
    sort(cands.begin(), cands.end(), [](const pair<double,string>& a, const pair<double,string>& b){
        if(a.first != b.first) return a.first > b.first;
        if(a.second.size() != b.second.size()) return a.second.size() < b.second.size();
        return a.second < b.second;
    });
    vector<string> chosen;
    unordered_set<string> seen;
    seen.reserve((size_t)Dmax*2+16);
    for(auto& c : cands){
        if((long long)chosen.size() >= Dmax) break;
        if(seen.count(c.second)) continue;
        seen.insert(c.second);
        chosen.push_back(c.second);
    }
    cout << chosen.size() << "\n";
    for(auto& s : chosen) cout << s << "\n";
    return 0;
}

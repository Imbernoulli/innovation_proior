// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious "smarter" frequency-based dictionary trainer: scan every candidate length
// 3..12, score each candidate substring by its POOLED (summed across all 3 domains)
// occurrence count times (length - token cost), and take the top Dmax by score. This is
// domain-blind: it never checks whether a high scorer actually helps every domain.
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int K; long long Dmax;
    cin >> K >> Dmax;
    vector<string> corpus(K);
    vector<long long> raw(K);
    for(int d=0; d<K; d++) cin >> raw[d] >> corpus[d];

    const int LMIN = 3, LMAX = 12, TOKEN_COST = 2;
    vector<pair<double,string>> cands;
    cands.reserve(1<<18);
    for(int L=LMIN; L<=LMAX; L++){
        unordered_map<string,long long> cnt;
        cnt.reserve(1<<16);
        for(int d=0; d<K; d++){
            const string& T = corpus[d];
            int n = (int)T.size();
            for(int i=0; i+L<=n; i++) cnt[T.substr(i,L)]++;
        }
        for(auto& kv : cnt){
            double score = (double)kv.second * (double)(L - TOKEN_COST);
            cands.push_back({score, kv.first});
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

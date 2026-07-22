// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Reproduces the grader's own baseline exactly: the Dmax most frequent length-3
// substrings, pooled (summed) across all three corpora with no domain awareness and no
// other length considered.
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int K; long long Dmax;
    cin >> K >> Dmax;
    vector<string> corpus(K);
    vector<long long> raw(K);
    for(int d=0; d<K; d++) cin >> raw[d] >> corpus[d];

    const int LMIN = 3;
    unordered_map<string,long long> cnt;
    cnt.reserve(1<<16);
    for(int d=0; d<K; d++){
        const string& T = corpus[d];
        int n = (int)T.size();
        for(int i=0; i+LMIN<=n; i++) cnt[T.substr(i,LMIN)]++;
    }
    vector<pair<long long,string>> v;
    v.reserve(cnt.size());
    for(auto& kv : cnt) v.push_back({kv.second, kv.first});
    sort(v.begin(), v.end(), [](const pair<long long,string>& a, const pair<long long,string>& b){
        if(a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    long long D = min((long long)v.size(), Dmax);
    cout << D << "\n";
    for(long long i=0; i<D; i++) cout << v[i].second << "\n";
    return 0;
}

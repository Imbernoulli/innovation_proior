// TIER: greedy
// The obvious recipe: a UNIFORM B-tree. Pack size-B separator pages, balance the
// tree purely by tile COUNT (fanout B+1), fully demand-blind and scan-blind.
#include <bits/stdc++.h>
using namespace std;

int N, B, S;
vector<vector<int>> nodeKeys, nodeCh;   // index = id-1
int cnt = 0;

int newNode(){ nodeKeys.emplace_back(); nodeCh.emplace_back(); return ++cnt; }

int build(int lo, int hi){
    if (lo > hi) return 0;
    int size = hi - lo + 1;
    int id = newNode();
    if (size <= B){
        vector<int> ks; for (int k = lo; k <= hi; k++) ks.push_back(k);
        nodeKeys[id - 1] = ks;
        nodeCh[id - 1].assign((int)ks.size() + 1, 0);
        return id;
    }
    int rem = size - B, base = rem / (B + 1), extra = rem % (B + 1);
    int cur = lo;
    vector<int> ks;
    vector<pair<int,int>> childIv;
    for (int j = 0; j <= B; j++){
        int cc = base + (j < extra ? 1 : 0);
        childIv.push_back({cur, cur + cc - 1});   // empty if cc==0
        cur += cc;
        if (j < B){ ks.push_back(cur); cur++; }
    }
    vector<int> chIds;
    for (auto &iv : childIv) chIds.push_back(build(iv.first, iv.second));
    nodeKeys[id - 1] = ks;
    nodeCh[id - 1] = chIds;
    return id;
}

int main(){
    scanf("%d %d %d", &N, &B, &S);
    long long tmp;
    for (int i = 0; i < N; i++) scanf("%lld", &tmp);
    for (int i = 0; i < S; i++){ long long a,b,c; scanf("%lld %lld %lld", &a,&b,&c); }

    nodeKeys.reserve(2 * (N / max(1,B) + 4));
    nodeCh.reserve(2 * (N / max(1,B) + 4));
    int root = build(1, N);

    printf("%d %d\n", cnt, root);
    for (int id = 1; id <= cnt; id++){
        auto &ks = nodeKeys[id - 1];
        auto &cs = nodeCh[id - 1];
        printf("%d", (int)ks.size());
        for (int k : ks) printf(" %d", k);
        for (int c : cs) printf(" %d", c);
        printf("\n");
    }
    return 0;
}

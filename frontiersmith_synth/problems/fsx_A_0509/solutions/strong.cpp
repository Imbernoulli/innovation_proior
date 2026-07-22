// TIER: strong
// The insight: kerf turns feasibility into a property of the cut-tree shape, so the
// dying blade must be rationed by MARGINAL VALUE PER UNIT BUDGET (value / cost),
// i.e. per unit of thin cut-length -- not by raw value and not by cut order.
// Ration by density (a fractional-knapsack relaxation, then a final 0/1 top-up):
// this fills up on the many cheap efficient shelves and leaves the greedy's
// budget-hogging premium shelves behind.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Shelf { ll h, WB; int idB; ll WN; int idN; ll bonusVal, cost; bool up; };

ll W, H, T, t, L; int D;
vector<Shelf> shelves;

void emitShelf(const Shelf& s){
    if (s.up){
        printf("1 0 1 %lld\n", s.WB);
        printf("0 %d\n", s.idB);
        printf("1 0 1 %lld\n", s.WN);
        printf("0 %d\n", s.idN);
        printf("0 0\n");
    } else {
        printf("1 0 0 %lld\n", s.WB);
        printf("0 %d\n", s.idB);
        printf("0 0\n");
    }
}
void emit(int idx){
    if (idx == (int)shelves.size()){ printf("0 0\n"); return; }
    printf("1 1 0 %lld\n", shelves[idx].h);
    emitShelf(shelves[idx]);
    emit(idx + 1);
}

int main(){
    scanf("%lld %lld %lld %lld %lld %d", &W, &H, &T, &t, &L, &D);
    vector<ll> w(D+1), h(D+1), v(D+1);
    map<ll, vector<int>> g;
    for (int i = 1; i <= D; i++){ scanf("%lld %lld %lld", &w[i], &h[i], &v[i]); g[h[i]].push_back(i); }
    for (auto& kv : g){
        auto ids = kv.second;
        sort(ids.begin(), ids.end(), [&](int a, int b){ return w[a] > w[b]; });
        Shelf s; s.h = kv.first; s.up = false;
        s.idB = ids[0]; s.WB = w[ids[0]];
        if (ids.size() >= 2){ s.idN = ids[1]; s.WN = w[ids[1]]; s.bonusVal = v[ids[1]]; }
        else { s.idN = 0; s.WN = 0; s.bonusVal = 0; }
        s.cost = 2 * s.h;
        shelves.push_back(s);
    }
    // ration by DENSITY = bonusVal / cost (value per unit thin cut-length)
    vector<int> ord;
    for (size_t i = 0; i < shelves.size(); i++)
        if (shelves[i].idN != 0 && shelves[i].bonusVal > 0) ord.push_back(i);
    sort(ord.begin(), ord.end(), [&](int a, int b){
        // compare bonusVal_a/cost_a vs bonusVal_b/cost_b  (cross-multiply, avoid FP)
        return (__int128)shelves[a].bonusVal * shelves[b].cost
             > (__int128)shelves[b].bonusVal * shelves[a].cost;
    });
    ll budget = L;
    // greedy-by-density pass
    vector<int> skipped;
    for (int i : ord){
        if (shelves[i].cost <= budget){ shelves[i].up = true; budget -= shelves[i].cost; }
        else skipped.push_back(i);
    }
    // top-up pass: try to fit any skipped shelf into the leftover budget (0/1 refinement)
    sort(skipped.begin(), skipped.end(), [&](int a, int b){ return shelves[a].cost < shelves[b].cost; });
    for (int i : skipped){
        if (shelves[i].cost <= budget){ shelves[i].up = true; budget -= shelves[i].cost; }
    }
    emit(0);
    return 0;
}

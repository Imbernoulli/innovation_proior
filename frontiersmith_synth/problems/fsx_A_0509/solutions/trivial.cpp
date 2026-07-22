// TIER: trivial
// Reliable-blade baseline: cut every shelf's WIDE base piece with the thick blade,
// never touch the dying thin blade. This reproduces the checker's baseline B -> ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Shelf { ll h, WB; int idB; ll WN; int idN; ll bonusVal, cost; bool up; };

ll W, H, T, t, L; int D;
vector<Shelf> shelves;

void emitShelf(const Shelf& s){
    if (s.up){
        printf("1 0 1 %lld\n", s.WB);   // thin vertical, isolate base
        printf("0 %d\n", s.idB);
        printf("1 0 1 %lld\n", s.WN);   // thin vertical, isolate bonus
        printf("0 %d\n", s.idN);
        printf("0 0\n");                 // waste
    } else {
        printf("1 0 0 %lld\n", s.WB);   // thick vertical, isolate base
        printf("0 %d\n", s.idB);
        printf("0 0\n");                 // waste
    }
}
void emit(int idx){
    if (idx == (int)shelves.size()){ printf("0 0\n"); return; } // final waste leaf
    printf("1 1 0 %lld\n", shelves[idx].h);   // thick horizontal, carve this shelf
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
    emit(0);
    return 0;
}

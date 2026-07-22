// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

// The canonical first attempt: ONE full-range uniform random shuffle, then
// repair any leftover fixed points. No algebraic structure at all -- this is
// what "uniform random + repair" (the addendum's own archetype for the
// obvious greedy) looks like for this objective.

static u64 rngstate;
static u64 next_rand(){
    rngstate += 0x9E3779B97F4A7C15ULL;
    u64 z = rngstate;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z;
}

int main(){
    int p;
    scanf("%d", &p);

    vector<int> pi(p);
    for (int i = 0; i < p; i++) pi[i] = i;
    rngstate = (u64)p * 0x2545F4914F6CDD1DULL + 12345ULL;
    for (int i = p - 1; i > 0; i--){
        int j = (int)(next_rand() % (u64)(i + 1));
        swap(pi[i], pi[j]);
    }
    for (int pass = 0; pass < 12; pass++){
        bool changed = false;
        for (int x = 0; x < p; x++){
            if (pi[x] == x){
                int y = (x + 1) % p;
                swap(pi[x], pi[y]);
                changed = true;
            }
        }
        if (!changed) break;
    }

    for (int x = 0; x < p; x++) printf("%d%c", pi[x], x + 1 < p ? ' ' : '\n');
    return 0;
}

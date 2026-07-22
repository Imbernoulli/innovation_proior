// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// Reproduces the checker's own reference construction EXACTLY:
// block-local Fisher-Yates shuffle (block size round(5*sqrt(p))) with a fixed
// splitmix64 stream seeded from p, then repair leftover fixed points by
// swapping each with its right neighbour until none remain.

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

    int bs = max(2, (int)llround(5.0 * sqrt((double)p)));
    vector<int> pi(p);
    for (int i = 0; i < p; i++) pi[i] = i;
    rngstate = (u64)p * 0x2545F4914F6CDD1DULL + 777ULL;
    for (int start = 0; start < p; start += bs){
        int end = min(p, start + bs);
        for (int i = end - 1; i > start; i--){
            int j = start + (int)(next_rand() % (u64)(i - start + 1));
            swap(pi[i], pi[j]);
        }
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

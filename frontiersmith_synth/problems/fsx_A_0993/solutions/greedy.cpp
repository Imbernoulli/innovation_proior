// TIER: greedy
// The obvious first pass: give every block an independently pseudo-random
// pattern (maximize apparent variety) and color each block with its own
// FRESH round-robin counter -- the natural per-block coloring habit. It
// never looks at a neighbor's border, so seams only line up by chance
// (about half of Smax); and because the per-block counter restarts at 0
// for every block, whenever the palette size c does not evenly divide b*b,
// the small per-block remainder bias adds up B*B times and can blow
// straight through the color-balance tolerance (scoring 0 on those tests),
// while a global counter (see strong/trivial) never would.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int B,b,c,Q,tol; long long W1;
    if (scanf("%d %d %d %d %d %lld", &B, &b, &c, &Q, &tol, &W1) != 6) return 0;

    for (int br = 0; br < B; br++){
        for (int bc = 0; bc < B; bc++){
            unsigned long long seed = 0x9E3779B97F4A7C15ULL
                ^ ((unsigned long long)br * 1000003ULL)
                ^ ((unsigned long long)bc * 998244353ULL);
            int lk = 0;
            for (int i = 0; i < b; i++){
                for (int j = 0; j < b; j++){
                    unsigned long long h = seed + (unsigned long long)(i * b + j) * 2654435761ULL;
                    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
                    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
                    h ^= h >> 33;
                    int d  = (int)(h & 1ULL);
                    int c1 = (lk % c) + 1;
                    int c2 = ((lk + 1) % c) + 1;
                    printf("%d %d %d\n", d, c1, c2);
                    lk++;
                }
            }
        }
    }
    return 0;
}

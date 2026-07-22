// TIER: trivial
// The checker's own baseline construction: every cell d=0 (perfect seam
// continuity everywhere -> one dihedral class), colored with a single
// GLOBAL continuous round-robin counter (never resets per block) so the
// balance gate is always satisfied. Reproduces Bref exactly -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int B,b,c,Q,tol; long long W1;
    if (scanf("%d %d %d %d %d %lld", &B, &b, &c, &Q, &tol, &W1) != 6) return 0;
    long long K = 0;
    for (int br = 0; br < B; br++)
        for (int bc = 0; bc < B; bc++)
            for (int i = 0; i < b; i++)
                for (int j = 0; j < b; j++){
                    int c1 = (int)(K % c) + 1;
                    int c2 = (int)((K + 1) % c) + 1;
                    printf("0 %d %d\n", c1, c2);
                    K++;
                }
    return 0;
}

// TIER: trivial
// Alternating-domino baseline: fill every row with the matched pair A,B repeated
// (A,B,A,B,...). Reproduces the checker's baseline B = R*floor(C/2) -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int R, C, T;
    if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
    vector<int> W(T), E(T), N(T), S(T);
    int maxW = 0;
    for (int i = 0; i < T; i++){
        scanf("%d %d %d %d", &W[i], &E[i], &N[i], &S[i]);
        maxW = max(maxW, W[i]);
    }
    int P = maxW + 1;
    // A = ideal tile(0,0): W=0,E=1,N=P,S=P+1 ; B = ideal tile(1,0): W=1,E=2,N=P,S=P+1
    int A = -1, Bt = -1;
    for (int i = 0; i < T; i++){
        if (W[i] == 0 && E[i] == 1 && N[i] == P && S[i] == P + 1) A = i;
        if (W[i] == 1 && E[i] == 2 && N[i] == P && S[i] == P + 1) Bt = i;
    }
    if (A < 0) A = 0;
    if (Bt < 0) Bt = (T > 1 ? 1 : 0);

    for (int r = 0; r < R; r++){
        for (int c = 0; c < C; c++){
            printf("%d", (c % 2 == 0) ? A : Bt);
            putchar(c + 1 < C ? ' ' : '\n');
        }
    }
    return 0;
}

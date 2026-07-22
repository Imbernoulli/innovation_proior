// TIER: strong
// The insight: the tile set is a PHASE FIELD -- ideal tile(x,y) has E=(x+1)%P and
// S=P+((y+1)%Q). The perfect target is the periodic assignment tile(c%P, r%Q); the only
// unavoidable losses are the forced-defect cells whose ideal tile was removed. So keep
// the global periodic field everywhere and, at each defect cell, ROUTE the defect: place
// the present tile that matches the most of its four (ideal) neighbors -- never the
// poison decoy that greedy grabs. Defect routing, not local matching.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int R, C, T;
    if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
    vector<int> W(T), E(T), N(T), S(T);
    int maxW = 0, minN = INT_MAX, maxN = INT_MIN;
    for (int i = 0; i < T; i++){
        scanf("%d %d %d %d", &W[i], &E[i], &N[i], &S[i]);
        maxW = max(maxW, W[i]);
    }
    int P = maxW + 1;
    // vertical colors of ideal tiles live in [P, P+Q); decoys have S huge (poison).
    for (int i = 0; i < T; i++){
        if (N[i] < P + 1000) { minN = min(minN, N[i]); maxN = max(maxN, N[i]); }
    }
    int Q = maxN - P + 1;
    if (Q < 1) Q = 1;

    // map (x,y) -> ideal tile index; ideal iff E<P and S in [P,P+Q).
    vector<int> idx((long long)P * Q, -1);
    for (int i = 0; i < T; i++){
        if (E[i] >= 0 && E[i] < P && S[i] >= P && S[i] < P + Q){
            int x = W[i], y = N[i] - P;
            if (x >= 0 && x < P && y >= 0 && y < Q) idx[(long long)y * P + x] = i;
        }
    }

    vector<vector<int>> g(R, vector<int>(C, 0));
    for (int r = 0; r < R; r++){
        int yd = r % Q;
        for (int c = 0; c < C; c++){
            int xd = c % P;
            int id = idx[(long long)yd * P + xd];
            if (id >= 0){ g[r][c] = id; continue; }
            // forced defect: route it -- best present tile vs the four ideal neighbors.
            int wantW = xd;                    // left ideal's E
            int wantE = (xd + 1) % P;           // right ideal's W
            int wantN = P + yd;                 // top ideal's S
            int wantS = P + ((yd + 1) % Q);     // bottom ideal's N
            int best = -1, bestIdx = 0;
            for (int t = 0; t < T; t++){
                int s = (W[t] == wantW) + (E[t] == wantE) + (N[t] == wantN) + (S[t] == wantS);
                if (s > best){ best = s; bestIdx = t; }
            }
            g[r][c] = bestIdx;
        }
    }
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            printf("%d%c", g[r][c], c + 1 < C ? ' ' : '\n');
    return 0;
}

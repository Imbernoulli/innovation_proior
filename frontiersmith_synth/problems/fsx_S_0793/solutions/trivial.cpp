// TIER: trivial
// Do-nothing baseline: seed no cells. The all-zero state is a fixed point of any
// XOR rule, so it reproduces the checker's own baseline construction exactly.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, T, B, M; scanf("%d %d %d %d", &N, &T, &B, &M);
    char buf[300]; scanf("%299s", buf);
    printf("0\n\n");
    return 0;
}

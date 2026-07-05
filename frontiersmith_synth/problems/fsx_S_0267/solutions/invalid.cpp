// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int H, W, P;
    scanf("%d %d %d", &H, &W, &P);
    // deliberately infeasible: anchor row far out of the frame -> checker rejects -> score 0
    printf("1\n1 0 100000 100000\n");
    return 0;
}

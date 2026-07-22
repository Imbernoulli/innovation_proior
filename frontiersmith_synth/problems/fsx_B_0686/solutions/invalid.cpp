// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: '?' is not an allowed character in the output string S (S must
// use only A-F). This must score 0.
int main() {
    int n;
    scanf("%d", &n);
    char buf[64];
    for (int i = 0; i < n; i++) scanf("%s", buf);
    printf("1\n?\n");
    return 0;
}

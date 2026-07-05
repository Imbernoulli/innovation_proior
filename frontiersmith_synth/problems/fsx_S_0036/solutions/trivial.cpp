// TIER: trivial
// Run every generator for the whole horizon -> exactly the checker's baseline B.
// F == B  =>  ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, G;
    if (scanf("%d %d", &T, &G) != 2) return 0;
    for (int g = 0; g < G; g++) { long long p,b,r,k,u; scanf("%lld %lld %lld %lld %lld",&p,&b,&r,&k,&u); }
    for (int t = 0; t < T; t++) { long long d,w; scanf("%lld %lld",&d,&w); }
    string line;
    for (int g = 0; g < G; g++) { line += (g? " ":""); line += "1"; }
    for (int t = 0; t < T; t++) { fputs(line.c_str(), stdout); fputc('\n', stdout); }
    return 0;
}

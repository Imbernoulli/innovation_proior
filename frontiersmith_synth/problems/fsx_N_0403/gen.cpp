#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static bool isPrime(ll n){ if(n<2) return false; for(ll d=2; d*d<=n; d++) if(n%d==0) return false; return true; }
static ll nextPrime(ll n){ if(n<3) n=3; if(n%2==0) n++; while(!isPrime(n)) n+=2; return n; }
static ll powmod(ll a, ll e, ll m){ a%=m; if(a<0)a+=m; ll r=1%m; while(e){ if(e&1) r=(__int128)r*a%m; a=(__int128)a*a%m; e>>=1;} return r; }
static bool isNZQR(ll s, ll p){ s%=p; if(s<0)s+=p; if(s==0) return false; return powmod(s,(p-1)/2,p)==1; }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10) - 1;

    // Size ladder: tiny (example scale) -> fills the constraint envelope (N=200000).
    int Ns[]   = {6, 60, 700, 4000, 15000, 45000, 90000, 140000, 180000, 200000};
    ll  ptar[] = {23, 97, 1009, 10007, 100003, 200003, 500009, 999983, 1500007, 2000003};
    // background weight range and coolant range per test.
    int wbg[]  = {40, 40000, 60000, 90000, 100000, 100000, 100000, 100000, 100000, 100000};
    int cmax[] = {8, 500, 3000, 12000, 25000, 40000, 40000, 40000, 40000, 40000};

    int N = Ns[idx];
    ll p = nextPrime(ptar[idx]);
    int WBG = wbg[idx];
    int CM = cmax[idx];
    ll C = (ll)CM * 9 / 4;                 // ~2.25 * cmax -> a loop holds ~4 rigs
    if (C < 1) C = 1;
    int M = max(2, N / 10);                // loops are the binding resource (knapsack selection)

    vector<ll> w(N + 1), c(N + 1), k(N + 1);

    // ---- background: modest weights, uniform coolant, random keys (natural ~50% traps) ----
    for (int i = 1; i <= N; i++) {
        w[i] = rnd.next(1LL, (ll)WBG);
        c[i] = rnd.next(1LL, C);
        k[i] = rnd.next(0LL, p - 1);
    }

    // Guarantee at least one individually-stable rig (key = 1 is always a QR) so B > 0.
    if (N >= 1) { k[1] = 1; if (c[1] > C) c[1] = C; }

    // ---- NEEDLE / PLANTED groups: heavy rigs that are UNSTABLE alone but STABLE together ----
    // Each group of 4 heavy rigs has individually non-residue keys whose SUM is a nonzero QR,
    // and whose coolant fits one loop. A weight-greedy that keeps singletons misses them.
    int groups = (N >= 20000) ? 5 : (N >= 400 ? 2 : 1);
    ll needleC = max(1LL, C / 5);          // 4 rigs -> 4C/5 <= C
    int placed = 0;
    for (int g = 0; g < groups && placed + 4 <= N; g++) {
        int base = N - placed;             // fill from the tail: indices base-3..base
        int ids[4] = {base - 3, base - 2, base - 1, base};
        ll ks[4];
        // first three keys: individually non-residues
        for (int t = 0; t < 3; t++) {
            ll key;
            do { key = rnd.next(1LL, p - 1); } while (isNZQR(key, p));
            ks[t] = key;
        }
        // fourth key: non-residue AND makes the group sum a nonzero QR
        ll partial = (ks[0] + ks[1] + ks[2]) % p;
        ll k4 = -1;
        for (int tries = 0; tries < 4 * (int)p + 50; tries++) {
            ll key = rnd.next(1LL, p - 1);
            if (isNZQR(key, p)) continue;
            if (isNZQR((partial + key) % p, p)) { k4 = key; break; }
        }
        if (k4 < 0) break;                 // extremely unlikely; skip planting
        ks[3] = k4;
        for (int t = 0; t < 4; t++) {
            int id = ids[t];
            w[id] = rnd.next(900000LL, 1000000LL);  // ~10x background: the high-value structure
            c[id] = needleC;
            k[id] = ks[t];
        }
        placed += 4;
    }

    // ---- TRAP sprinkling: pairs of heavy-ish rigs one step off a stable residue ----
    // Nudge a few mid rigs so obvious pairings land on a non-residue although a tiny swap fixes it.
    if (N >= 50) {
        int traps = min(N / 50, 40);
        for (int t = 0; t < traps; t++) {
            int i = rnd.next(1, N);
            // pick a non-residue key so this rig is unstable alone (forces grouping decisions)
            ll key;
            int guard = 0;
            do { key = rnd.next(1LL, p - 1); } while (isNZQR(key, p) && ++guard < 40);
            k[i] = key;
            w[i] = max<ll>(w[i], rnd.next((ll)WBG / 2, (ll)WBG));
        }
    }

    // ---- emit ----
    printf("%d %d %lld %lld\n", N, M, C, p);
    for (int i = 1; i <= N; i++)
        printf("%lld %lld %lld\n", w[i], c[i], k[i]);
    return 0;
}

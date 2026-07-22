#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Cyclic Cup Seeding"  (generator)   family: cyclic-outcome-bracket-seeding
//
// Emits N players (N a power of two), a full deterministic pairwise-outcome
// tournament (no draws; may be intransitive), and K sponsor players with bounty
// values.
//
// Structure modes, chosen per testId, planting the traps the brief requires:
//   RANDOM     - i.i.d. random tournament (control: aggregate win-count is only
//                a weak, generic predictor here, so a win-count seeder does
//                reasonably but not exceptionally).
//   CIRCULANT  - a rotational (near-regular) tournament: every player's total
//                win count is (almost) IDENTICAL, so "spread the high win-count
//                players apart" carries ~zero information about any specific
//                pairwise result. TRAP for a win-count-based seeder.
//   SPECIALIST - after building the base table, a handful of otherwise
//                mid-table (by win count) non-sponsor players are overwritten to
//                beat EVERY sponsor player directly. A win-count seeder ranks
//                them unremarkable and never isolates them; they quietly
//                eliminate any sponsor routed near them. TRAP.
// Several tests combine CIRCULANT + SPECIALIST for a compounded trap, and the
// last two tests fill the constraint envelope at N=1024.
// -----------------------------------------------------------------------------

int N_, K_;
vector<string> M; // 1-indexed rows/cols, M[i][j-1] = '1' iff player i beats player j

static void setResult(int a, int b, bool aBeatsB){
    M[a][b-1] = aBeatsB ? '1' : '0';
    M[b][a-1] = aBeatsB ? '0' : '1';
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size / bounty-count ladder ----
    int expo[10]   = {2, 3, 4, 5, 6, 7, 8, 9, 10, 10};   // N = 2^expo
    int kCount[10] = {2, 3, 3, 4, 5, 6, 7, 8, 10, 12};
    // mode: 0=random control, 1=circulant, 2=random+specialist,
    //       3=circulant+specialist
    int mode[10]   = {0, 0, 1, 2, 3, 2, 1, 0, 3, 3};

    int N = 1 << expo[testId - 1];
    int K = min(kCount[testId - 1], N);
    N_ = N; K_ = K;
    int md = mode[testId - 1];

    M.assign(N + 1, string(N, '0'));

    if (md == 1 || md == 3) {
        // circulant / near-regular tournament: player a beats player b iff the
        // forward cyclic distance (b-a mod N) lies in the "first half"; the lone
        // antipodal distance N/2 (only when N even) is split by index parity so
        // every player's win count differs by at most 1.
        int half = N / 2 - 1;
        for (int a = 1; a <= N; a++) {
            for (int b = a + 1; b <= N; b++) {
                int d = (b - a) % N;                // 1..N-1, a<b so d=b-a here
                bool aBeats;
                if (d <= half) aBeats = true;
                else if (d >= N - half) aBeats = false;
                else aBeats = (a < b);               // only the d==N/2 case
                setResult(a, b, aBeats);
            }
        }
    } else {
        // i.i.d. random tournament
        for (int a = 1; a <= N; a++)
            for (int b = a + 1; b <= N; b++)
                setResult(a, b, rnd.next(2) == 1);
    }

    // ---- sponsor players + bounties ----
    vector<int> ids;
    {
        vector<int> pool(N);
        for (int i = 0; i < N; i++) pool[i] = i + 1;
        for (int i = 0; i < K; i++) {
            int j = rnd.next(i, N - 1);
            swap(pool[i], pool[j]);
        }
        ids.assign(pool.begin(), pool.begin() + K);
    }
    vector<int> vals(K);
    for (int i = 0; i < K; i++) vals[i] = rnd.next(50, 500);

    // ---- specialist trap: a few non-sponsor players forced to beat EVERY
    //      sponsor directly, chosen to not stand out on aggregate win count ----
    if (md == 2 || md == 3) {
        vector<char> isSponsor(N + 1, 0);
        for (int id : ids) isSponsor[id] = 1;
        vector<int> nonSponsor;
        for (int i = 1; i <= N; i++) if (!isSponsor[i]) nonSponsor.push_back(i);
        int S = max(2, min((int)nonSponsor.size(), N / 24 + 2));
        for (int i = 0; i < S && i < (int)nonSponsor.size(); i++) {
            int j = rnd.next(i, (int)nonSponsor.size() - 1);
            swap(nonSponsor[i], nonSponsor[j]);
        }
        for (int s = 0; s < S; s++) {
            int spec = nonSponsor[s];
            for (int id : ids) setResult(spec, id, true); // specialist beats sponsor
        }
    }

    printf("%d %d\n", N, K);
    for (int i = 1; i <= N; i++) printf("%s\n", M[i].c_str());
    for (int i = 0; i < K; i++) printf("%d%c", ids[i], i + 1 == K ? '\n' : ' ');
    for (int i = 0; i < K; i++) printf("%d%c", vals[i], i + 1 == K ? '\n' : ' ');
    return 0;
}

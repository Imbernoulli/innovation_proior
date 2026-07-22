#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "The Surfing Barista".  family: single-boiler-surfing-queue
//
// One boiler, deterministic bang-bang hysteresis thermal law (two-setpoint
// oscillation, mechanism #3): every tick T += ((target-T)*P)/Q toward T_HOT
// while heater ON, toward T_COLD while heater OFF; heater flips ON at T<=T_LO,
// OFF at T>=T_HI. This produces a perpetual limit cycle the solver cannot
// command directly -- it can only (a) WAIT for the cycle to sweep a pending
// order's window into reach (free), or (b) spend one of FMAX FLUSHES (mechanism
// #1, free-control-actions: zero serving time, but a scarce resource) to drop
// the temperature and reach a value the current leg has already passed. Each
// order has a quality window + deadline (mechanism #2, quality-window-scoring):
// Q_i = max(0, 1000 - 2*band - 12*max(0,wait-3)), weighted by w_i, summed = F.
//
// Internal baseline B: serve orders back-to-back in arrival order on the
// boiler's NATURAL (never-flushed) trajectory, ignoring deadlines and window
// accuracy entirely -- always produces a modest, always-positive number, since
// it only needs SOME order to land with low wait / rough temperature match.
// -----------------------------------------------------------------------------

static const ll BASE = 1000, TEMP_RATE = 2, WAIT_RATE = 12, GRACE = 3;

static inline ll stepT(ll T, int H, ll THOT, ll PH, ll QH, ll TCOLD, ll PC, ll QC) {
    ll diff = H ? (THOT - T) : (TCOLD - T);
    ll p = H ? PH : PC, q = H ? QH : QC;
    T += (diff * p) / q;
    return T;
}
static inline int nextH(ll T, int H, ll TLO, ll THI) {
    if (T <= TLO) return 1;
    if (T >= THI) return 0;
    return H;
}
static inline ll qualityOf(ll T, ll lo, ll hi, ll a, ll t) {
    ll band = 0;
    if (T < lo) band = lo - T; else if (T > hi) band = T - hi;
    ll temp_pen = TEMP_RATE * band;
    ll wait = max(0LL, (t - a) - GRACE);
    ll wait_pen = WAIT_RATE * wait;
    ll Q = BASE - temp_pen - wait_pen;
    return Q < 0 ? 0 : Q;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int Tmax = inf.readInt();
    ll TLO = inf.readLong();
    ll THI = inf.readLong();
    ll THOT = inf.readLong();
    ll PH = inf.readLong(), QH = inf.readLong();
    ll TCOLD = inf.readLong();
    ll PC = inf.readLong(), QC = inf.readLong();
    ll T0 = inf.readLong();
    int H0 = inf.readInt();
    ll FLUSH_DROP = inf.readLong();
    int FMAX = inf.readInt();

    vector<ll> a(N), d(N), lo(N), hi(N), w(N);
    for (int i = 0; i < N; i++) {
        a[i] = inf.readLong();
        d[i] = inf.readLong();
        lo[i] = inf.readLong();
        hi[i] = inf.readLong();
        w[i] = inf.readLong();
    }

    // ---- natural (never-flushed) trajectory, used only for the baseline ----
    vector<ll> nat(Tmax + 1);
    nat[0] = T0;
    {
        int H = H0;
        for (int t = 0; t < Tmax; t++) {
            ll nt = stepT(nat[t], H, THOT, PH, QH, TCOLD, PC, QC);
            nat[t + 1] = nt;
            H = nextH(nt, H, TLO, THI);
        }
    }

    // ---- internal baseline B: arrival-order, back-to-back, on nat[], no deadline/window logic ----
    ll Bsum = 0;
    {
        vector<int> ord(N);
        iota(ord.begin(), ord.end(), 0);
        stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return a[x] < a[y]; });
        ll prev = -1;
        for (int idx : ord) {
            ll t = max(a[idx], prev + 1);
            if (t > Tmax - 1) t = Tmax - 1;
            prev = t;
            Bsum += w[idx] * qualityOf(nat[t], lo[idx], hi[idx], a[idx], t);
        }
    }
    ll B = Bsum;
    if (B <= 0) B = 1;

    // ---- read + validate participant output ----
    int M = ouf.readInt(0, Tmax, "M");
    vector<ll> ticks(M);
    vector<int> kk(M);
    vector<char> served(N, 0);
    ll prevTick = -1;
    int flushCount = 0;
    for (int j = 0; j < M; j++) {
        ll t = ouf.readLong((ll) 0, (ll) (Tmax - 1), "tick");
        if (t <= prevTick) quitf(_wa, "ticks not strictly increasing at action %d (t=%lld <= prev=%lld)", j, t, prevTick);
        prevTick = t;
        int k = (int) ouf.readLong((ll) -1, (ll) (N - 1), "k");
        ticks[j] = t;
        kk[j] = k;
        if (k == -1) {
            flushCount++;
        } else {
            if (served[k]) quitf(_wa, "order %d pulled twice", k);
            if (!(t >= a[k] && t <= d[k])) quitf(_wa, "order %d pulled at t=%lld outside [%lld,%lld]", k, t, a[k], d[k]);
            served[k] = 1;
        }
    }
    if (flushCount > FMAX) quitf(_wa, "flush count %d exceeds FMAX=%d", flushCount, FMAX);
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d actions", M);

    // ---- simulate the ACTUAL trajectory under the participant's flush choices, score pulls ----
    vector<ll> T(Tmax + 1);
    T[0] = T0;
    int H = H0;
    int aj = 0;
    ll F = 0;
    for (int t = 0; t < Tmax; t++) {
        ll cur = T[t];
        if (aj < M && ticks[aj] == t) {
            int k = kk[aj];
            if (k == -1) {
                cur -= FLUSH_DROP;
                if (cur < 0) cur = 0;
            } else {
                F += w[k] * qualityOf(cur, lo[k], hi[k], a[k], t);
            }
            aj++;
        }
        ll nt = stepT(cur, H, THOT, PH, QH, TCOLD, PC, QC);
        T[t + 1] = nt;
        H = nextH(nt, H, TLO, THI);
    }

    double sc = min(1000.0, 100.0 * (double) F / (double) max((ll) 1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld M=%d flush=%d/%d Ratio: %.6f", F, B, M, flushCount, FMAX, sc / 1000.0);
    return 0;
}

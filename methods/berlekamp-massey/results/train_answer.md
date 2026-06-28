In front of me is the first $2k$ terms of a sequence over $\mathbb{Z}_p$ for a prime $p$, together with a promise that the sequence obeys some linear recurrence of order $\le k$, and a request for the $N$-th term with $N$ as large as $10^{18}$. A linear recurrence of order $m$ is a fixed coefficient list $c_0, \dots, c_{m-1}$ with $a_i = \sum_{j=0}^{m-1} c_j\, a_{i-j-1}$ for every $i \ge m$, and I want the *shortest* such list — its order is unknown, only the bound $m \le k$ is given. The obvious handle is linear algebra: once I fix a candidate order $m$, each supplied index $i \ge m$ gives one linear equation in the unknown coefficients, so I have a square system to solve by Gaussian elimination over $\mathbb{Z}_p$ (division is legal because $p$ is prime, so every nonzero element has a modular inverse). Guessing the worst-case order $m = k$ gives a $k \times k$ system — exactly why $2k$ terms are handed over, $k$ to seed the first window and $k$ more to write $k$ constraints — but solving it does not give the *shortest* recurrence: if the true order is smaller the system is rank-deficient, and I would have to detect the rank and dig the minimal recurrence out of the null structure. The honest alternative, looping $m = 1, 2, 3, \dots$ and solving an $m \times m$ system at each step until the coefficients predict the remaining terms, costs $\sum_{m=1}^{k} O(m^3) = O(k^4)$ and, worse, throws away all the elimination work each time $m$ bumps up by one, re-solving a system that shares almost everything with the previous one. The waste is the signal: the data has one shortest recurrence of one order, and the right question is not "for each candidate order, does a recurrence exist?" but "what is the shortest recurrence consistent with what I have seen so far, and how must it change when one more term arrives?"

I propose Berlekamp–Massey: build the recurrence incrementally in a single left-to-right scan, always holding the shortest recurrence $C = (c_0, \dots, c_{m-1})$ that fits everything read so far, and repairing it locally whenever a new term breaks it. When the next term $a_i$ arrives I do the one thing a recurrence claims it can do — predict it, $\hat a_i = \sum_{j} c_j a_{i-j-1}$ — and form the discrepancy $\delta_i = \hat a_i - a_i$, fixing the sign convention as predicted-minus-actual once and for all. If $\delta_i = 0$, $C$ still explains everything including this new term and I continue. If $\delta_i \ne 0$, $C$ fits everything up to $i-1$ but mispredicts $i$, and I must repair it so the new $C$ predicts $a_i$ correctly *without breaking any earlier term it already gets right*, growing the order as little as possible — ideally not at all. The key observation is what adding a vector $G = (g_0, g_1, \dots)$ to $C$ does to predictions: at any position $t$ the prediction changes by $G(t) = \sum_j g_j a_{t-j-1}$, the value of $G$ at $t$. I want a $G$ with $G(i) = a_i - \hat a_i = -\delta_i$ and $G(t) = 0$ at every earlier position $t < i$ where $C$ was already correct. Such a $G$ comes from the history of the scan itself. The current $C$ became what it is only because at some earlier step a previous recurrence $B = (b_0, b_1, \dots)$ was current until it failed at some position $f < i$ with its own known nonzero discrepancy $\mathrm{ld}$ there. The load-bearing object is not $B$'s prediction but its residual,
$$R_B(t) = a_t - \sum_j b_j a_{t-j-1},$$
which is $0$ wherever $B$ was correct and equals $-\mathrm{ld}$ at $f$ — silent everywhere, then a single known nonzero spike, exactly the shape I need. To turn it into a legal lookback coefficient vector at the later index $i$, I shift it by $i - f$: the residual's leading $+1$ coefficient on $a_f$ becomes a lookback coefficient landing at index $i - f - 1$, and the $-b_j$ coefficients trail below it, giving the unit repair vector $(0, \dots, 0, 1, -b_0, -b_1, \dots)$ with $i - f - 1$ leading zeros. This vector evaluates to $R_B(f) = -\mathrm{ld}$ at position $i$ and to earlier residuals where $B$ was already correct — hence $0$ — at every earlier position. I want a contribution of $-\delta_i$ and the unit contributes $-\mathrm{ld}$, so the scale is $\kappa = \delta_i\,\mathrm{ld}^{-1}$, the discrepancy-now over the discrepancy-then, and the repair is
$$C \leftarrow C + \kappa\,(0, \dots, 0, 1, -b_0, -b_1, \dots), \qquad \kappa = \delta_i\,\mathrm{ld}^{-1}.$$
The denominator $\mathrm{ld}$ is nonzero by construction, and because $p$ is prime I divide by it with a Fermat modular inverse — this $1/\mathrm{ld}$ is precisely why the modulus must be prime. The sign convention stays consistent: with $\hat a_i$ and $d = \hat a_i - a_i$, the unit shifted residual contributes $-\mathrm{ld}$ at $i$, I need $-d$, so $\kappa(-\mathrm{ld}) = -d$ forces $\kappa = d\,\mathrm{ld}^{-1}$ with the same convention top and bottom and no stray minus.

The order behavior falls out for free. The shifted fix has length $i - f + |B|$: if that exceeds the current $|C| = m$ the new $C$ is longer and the order grew, otherwise the order stays put, so the order increases only when the repair genuinely needs reach the current order cannot supply — I never decide to grow, the arithmetic grows only when forced. The first failure needs separate handling because no $B$ exists yet: scanning from the start, $C$ is empty (order $0$, effectively predicting $0$ for every term), and the first time a term is nonzero, at some position $i$, no recurrence on all-zero history can reproduce it, since a linear recurrence on an all-zero seed stays zero. The forced minimal move is to declare $C$ an order-$(i+1)$ all-zero recurrence, which slides $a_i$ into the seed window rather than asking the recurrence to predict it, and to record this empty-recurrence failure as the first reference: $f = i$, $\mathrm{ld} = \hat a_i - a_i = -a_i$. From there every failure has machinery to work with. The remaining subtlety is which past failing recurrence to keep as $B$. What I care about is the future grown order $i' - f + |B|$ at the next failure $i'$, whose controllable part now is $|B| - f$ because $i'$ is common to both candidates. The old reference scores $|B| - f$ and the just-overwritten $C$ scores $|C| - i$, so I switch the reference to the pre-repair $C$ exactly when $|C| - i \le |B| - f$, equivalently $i - f + |B| \ge |C|$, recording its failing position and discrepancy at that moment. This bookkeeping is what keeps the order minimal across the whole scan. A quick check on $1, 2, 4, 8, \dots$: at $i=0$ the empty recurrence predicts $0$ against actual $1$, so $\mathrm{ld} = -1$ and $C = (0)$; at $i=1$, $C = (0)$ predicts $0$ against $2$, $d = -2$, the shifted residual is a lone coefficient at index $0$ with unit value $a_0 = 1 = -\mathrm{ld}$, the scale is $d/\mathrm{ld} = (-2)/(-1) = 2$, so $C$ becomes $(2)$, the recurrence $a_i = 2 a_{i-1}$ that all later powers of two obey. Each term costs $O(m)$ to predict and $O(m)$ to repair with $m \le k$, and there are $O(k)$ terms, so the whole scan is $O(k^2)$ and discovers the order for free — a clean drop from $O(k^4)$ guess-and-resolve and even from a single $O(k^3)$ elimination. The promise that $2k$ terms suffice now makes sense: a true shortest order $m \le k$ means the scan sees at least the first $2m$ terms, enough for the minimal recurrence to be forced by its own prefix while the rest re-confirm the coefficients.

That leaves the $N$-th term for $N$ up to $10^{18}$, where running the recurrence forward in $O(Nm)$ is hopeless; I need to leap to index $N$ in roughly $\log N$ steps. The recurrence is linear, so a companion-matrix power $M^{N-m+1}$ on the seed state by binary exponentiation works in $O(m^3 \log N)$, but the polynomial structure gives something slicker. Associate to $C$ the characteristic polynomial $f(x) = x^m - \sum_{j=0}^{m-1} c_j x^{m-1-j}$, and define for any polynomial $g(x) = \sum_t g_t x^t$ the linear functional $\Lambda(g) = \sum_t g_t a_t$ that reads off the $a$-term of each power. The recurrence is exactly the statement $\Lambda(x^t f) = 0$ for all $t \ge 0$, because $\Lambda(x^t f) = a_{t+m} - \sum_j c_j a_{t+m-1-j}$ is the recurrence at index $t+m$. So $\Lambda$ annihilates every multiple of $f$, which means $\Lambda(g)$ depends only on $g \bmod f$. Therefore
$$a_N = \Lambda(x^N) = \Lambda(x^N \bmod f) = \sum_{i=0}^{m-1} (x^N \bmod f)_i\, a_i,$$
a length-$m$ dot product of the reduced polynomial's coefficients with the seed terms I already hold. The whole problem reduces to computing $x^N \bmod f$, which I get by square-and-multiply in $\mathbb{Z}_p[x]/(f)$: keep two degree-$<m$ polynomials, $s$ (accumulator, init $1$) and $t$ (running square of $x$, init $x$), walk the bits of $N$ from least significant, multiply $s$ by $t$ when the bit is set, square $t$ always. Each multiply is a polynomial product of degree $< 2m$ followed by a reduction back to degree $< m$, and the reduction is the load-bearing operation: rewrite each high power using $x^e = x^{e-m}\cdot x^m \equiv \sum_j c_j x^{e-1-j} \pmod f$, so a coefficient at index $e \ge m$ folds down as $r_{e-1-j} \mathrel{+}= r_e\,c_j$, swept from the top index down to $m$ until only indices $0, \dots, m-1$ remain. That is $O(m^2)$ per step and $O(m^2 \log N)$ overall, beating the matrix's $O(m^3 \log N)$. Two small wrinkles: at $m = 1$, $f(x) = x - c_0$ and $x$ already reduces to the constant $c_0$, so I seed $t$ as the constant $\mathrm{rec}[0]$ rather than the literal $x$ (which would have degree $1 = m$, out of range); and if $N$ is smaller than the number of seed terms I return $a_N$ directly, while an empty recurrence (an all-zero sequence) returns $0$.

The program reads `p`, `N`, the count of supplied terms, then the terms (whitespace-separated on stdin), and prints $a_N \bmod p$.

```cpp
// Reads: p, N, the count of supplied terms, then the terms (whitespace-separated).
// Prints: the N-th term of the sequence modulo p.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

ll power(ll a, ll b, ll p) {
    a %= p;
    if (a < 0) a += p;
    ll r = 1;
    while (b) {
        if (b & 1) r = (ll)((__int128)r * a % p);
        a = (ll)((__int128)a * a % p);
        b >>= 1;
    }
    return r;
}

ll inv(ll a, ll p) { return power(a, p - 2, p); } // Fermat; p prime

// Shortest C with a_i = sum_j C[j]*a[i-j-1], over Z_p.
vector<ll> find_recurrence(const vector<ll>& seq, ll p) {
    vector<ll> cur;        // current shortest recurrence so far
    vector<ll> last;       // the recurrence current right before the last failure
    ll lf = -1;            // failing index of `last`
    ll ld = 0;             // discrepancy (predicted - actual) of `last` at lf
    int n = (int)seq.size();
    for (int i = 0; i < n; i++) {
        // predict a_i from the current recurrence
        ll t = 0;
        for (int j = 0; j < (int)cur.size(); j++)
            t = (t + (__int128)seq[i - j - 1] * cur[j]) % p;
        ll d = ((t - seq[i]) % p + p) % p;     // discrepancy, predicted - actual
        if (d == 0) continue;                  // current recurrence still fits
        if (cur.empty()) {
            // first nonzero term: set the order floor, record this failure
            cur.assign(i + 1, 0);
            lf = i;
            ld = d;
            continue;
        }
        // build the scaled, shifted residual of the last failing recurrence
        ll k = (__int128)d * inv(ld, p) % p;   // discrepancy now / discrepancy then
        vector<ll> c(i - lf - 1, 0);
        c.push_back(k);                        // the +1-at-index-(i-lf-1), scaled
        for (int j = 0; j < (int)last.size(); j++)
            c.push_back(((-(__int128)last[j] * k) % p + p) % p);
        if ((int)c.size() < (int)cur.size())
            c.resize(cur.size(), 0);
        vector<ll> prev = cur;                 // remember pre-repair cur as a candidate ref
        for (int j = 0; j < (int)c.size(); j++) {
            ll cj = (j < (int)cur.size()) ? cur[j] : 0;
            c[j] = (c[j] + cj) % p;
        }
        // keep, as `last`, whichever reference grows the order least next time
        if (i - lf + (ll)last.size() >= (ll)cur.size()) {
            last = prev;
            lf = i;
            ld = d;
        }
        cur = c;
    }
    for (auto& x : cur) x = ((x % p) + p) % p;
    return cur;
}

// a_N mod p from rec = C[0..m-1] and the seed terms seq.
ll kth_term(const vector<ll>& rec, const vector<ll>& seq, ll N, ll p) {
    int m = (int)rec.size();
    if (N < (ll)seq.size()) return ((seq[N] % p) + p) % p;
    if (m == 0) return 0;

    // multiply two deg<m polys, reduce mod x^m - sum rec[j] x^{m-1-j}
    auto mulmod = [&](const vector<ll>& a, const vector<ll>& b) {
        vector<ll> r(2 * m, 0);
        for (int i = 0; i < m; i++) {
            if (a[i]) {
                ll ai = a[i];
                for (int j = 0; j < m; j++)
                    r[i + j] = (r[i + j] + (__int128)ai * b[j]) % p;
            }
        }
        for (int e = 2 * m - 1; e >= m; e--) {
            if (r[e]) {
                ll re = r[e];
                for (int j = 0; j < m; j++)
                    r[e - 1 - j] = (r[e - 1 - j] + (__int128)re * rec[j]) % p;
            }
        }
        r.resize(m);
        return r;
    };

    vector<ll> s(m, 0), t(m, 0);
    s[0] = 1;                          // accumulator = 1
    if (m == 1) t[0] = rec[0] % p;     // x reduces to the constant c_0
    else t[1] = 1;                     // running square = x
    ull K = (ull)N;
    while (K) {
        if (K & 1ULL) s = mulmod(s, t);
        t = mulmod(t, t);
        K >>= 1;
    }
    ll ans = 0;
    for (int i = 0; i < m; i++)
        ans = (ans + (__int128)s[i] * seq[i]) % p;
    return ((ans % p) + p) % p;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    ll p, N, cnt;
    if (!(cin >> p)) return 0;
    cin >> N >> cnt;
    vector<ll> seq(cnt);
    for (ll i = 0; i < cnt; i++) {
        ll v;
        cin >> v;
        seq[i] = ((v % p) + p) % p;
    }
    vector<ll> rec = find_recurrence(seq, p);
    cout << kth_term(rec, seq, N, p) << "\n";
    return 0;
}
```

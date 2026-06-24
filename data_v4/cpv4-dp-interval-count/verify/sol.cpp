#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {                 // empty input -> no expression
        cout << 0 << "\n";
        return 0;
    }

    const long long MOD = 1000000007LL;

    // The expression alternates literal, operator, literal, operator, ..., literal.
    // Extract the operands (T / F) into `val` and the operators (& | ^) into `op`.
    vector<int> val;                   // 1 for T, 0 for F
    vector<char> op;                   // '&', '|', '^'
    bool wellFormed = true;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (i % 2 == 0) {              // even position must be a literal
            if (c == 'T') val.push_back(1);
            else if (c == 'F') val.push_back(0);
            else { wellFormed = false; break; }
        } else {                       // odd position must be an operator
            if (c == '&' || c == '|' || c == '^') op.push_back(c);
            else { wellFormed = false; break; }
        }
    }
    if (s.size() % 2 == 0) wellFormed = false;  // must end on a literal
    if (!wellFormed) { cout << 0 << "\n"; return 0; }

    int m = (int)val.size();           // number of literals, m >= 1
    // dpT[i][j], dpF[i][j] = #ways to parenthesize literals i..j so the value is T / F.
    vector<vector<long long>> dpT(m, vector<long long>(m, 0));
    vector<vector<long long>> dpF(m, vector<long long>(m, 0));

    for (int i = 0; i < m; i++) {
        dpT[i][i] = (val[i] == 1) ? 1 : 0;
        dpF[i][i] = (val[i] == 0) ? 1 : 0;
    }

    // Interval DP over increasing length; split at operator k between literals i..k and k+1..j.
    for (int len = 2; len <= m; len++) {
        for (int i = 0; i + len - 1 < m; i++) {
            int j = i + len - 1;
            long long t = 0, f = 0;
            for (int k = i; k < j; k++) {       // operator op[k] joins [i..k] and [k+1..j]
                long long lt = dpT[i][k], lf = dpF[i][k];
                long long rt = dpT[k + 1][j], rf = dpF[k + 1][j];
                long long total = ((lt + lf) % MOD) * ((rt + rf) % MOD) % MOD;
                long long ways_t = 0;
                if (op[k] == '&') {
                    ways_t = lt * rt % MOD;
                } else if (op[k] == '|') {
                    // T unless both F
                    ways_t = (total - lf * rf % MOD + MOD) % MOD;
                } else { // '^'
                    ways_t = (lt * rf % MOD + lf * rt % MOD) % MOD;
                }
                long long ways_f = (total - ways_t % MOD + MOD) % MOD;
                t = (t + ways_t) % MOD;
                f = (f + ways_f) % MOD;
            }
            dpT[i][j] = t;
            dpF[i][j] = f;
        }
    }

    cout << dpT[0][m - 1] % MOD << "\n";
    return 0;
}

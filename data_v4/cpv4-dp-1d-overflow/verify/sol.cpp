#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> exactly one pattern (all lamps off, vacuously valid)
    string s;
    if (n > 0) cin >> s;                  // s has length n: '.' = working lamp, 'x' = broken (forced off)

    // off = number of valid patterns for the processed prefix whose LAST lamp is OFF
    // on  = number of valid patterns for the processed prefix whose LAST lamp is ON
    // A lamp may be ON only if it is working ('.') and the previous lamp is OFF.
    long long off = 1, on = 0;            // empty prefix: 1 pattern, ends "off" by convention
    for (int i = 0; i < n; i++) {
        long long noff = off + on;        // this lamp OFF: previous could be either
        long long non = (s[i] == '.') ? off : 0; // this lamp ON: needs working lamp + previous OFF
        off = noff;
        on = non;
    }

    cout << (off + on) << "\n";           // total valid patterns over the whole strip
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    long long k;
    if (!(cin >> s >> k)) return 0;        // empty input -> nothing to print

    long long n = (long long)s.size();
    if (k < 0) k = 0;                       // defensive: no deletions requested
    if (k >= n) {                           // delete everything (or more): empty result
        cout << "\n";
        return 0;
    }

    long long keep = n - k;                 // final length is fixed
    string st;                              // monotonic stack of kept characters
    st.reserve((size_t)n);
    long long budget = k;                   // remaining deletions allowed

    for (long long i = 0; i < n; i++) {
        char c = s[(size_t)i];
        // Pop a strictly larger top while we still have budget: a smaller char
        // arriving later at an earlier position lowers the result.
        while (budget > 0 && !st.empty() && st.back() > c) {
            st.pop_back();
            budget--;
        }
        st.push_back(c);
    }

    // If budget remains, the stack is non-decreasing; deleting from the tail is
    // optimal, so truncate to the required length.
    if ((long long)st.size() > keep) st.resize((size_t)keep);

    cout << st << "\n";
    return 0;
}

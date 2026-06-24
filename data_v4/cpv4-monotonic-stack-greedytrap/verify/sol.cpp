#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    string s;
    cin >> s;

    // Monotonic (non-decreasing) stack: while we still have removals left and the
    // top of the stack is strictly greater than the current character, pop it.
    // This greedily fixes the earliest position where a smaller character can
    // take a more significant slot.
    string st;
    st.reserve(n);
    long long budget = k;
    for (int i = 0; i < n; i++) {
        char c = s[i];
        while (budget > 0 && !st.empty() && st.back() > c) {
            st.pop_back();
            budget--;
        }
        st.push_back(c);
    }
    // If removals remain (string was non-decreasing), drop from the tail.
    while (budget > 0) {
        st.pop_back();
        budget--;
    }

    cout << st << "\n";
    return 0;
}

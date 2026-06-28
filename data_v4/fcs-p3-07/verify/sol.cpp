#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, k;
        cin >> n >> k;

        // Find the 0-indexed survivor among people labelled 0..n-1 standing in a
        // circle, eliminating every k-th person (counting starts at person 0).
        // Classic recurrence: r(1) = 0, r(m) = (r(m-1) + k) mod m.
        // We need the result for m = n. A plain loop over m = 2..n is O(n),
        // which is too slow for n up to 1e9. Because k is small we batch the
        // increments where no modular wrap occurs, giving O(k log n).
        long long r = 0;          // survivor (0-indexed) for current population
        long long cnt = 1;        // current population size

        if (k == 1) {
            // Every 1st person eliminated: eliminations go 0,1,2,...,n-1,
            // so the last person to die / survivor reasoning -> survivor is n-1.
            r = n - 1;
            cnt = n;
        }

        while (cnt < n) {
            // We hold r = survivor index for `cnt` people. Advancing one step:
            //   cnt -> cnt+1, r -> (r + k) % (cnt+1).
            // As long as r + k < (cnt+1) the mod is a no-op shift by k each step.
            // We may add multiple people at once. After adding `step` people the
            // population becomes cnt+step and r becomes r + k*step provided no
            // intermediate value reaches the (growing) modulus. Find the largest
            // safe `step`.
            //
            // After processing the j-th of these steps (j = 1..step) the
            // population is cnt+j and the candidate index is r + k*j. To avoid a
            // wrap we need r + k*j < cnt + j for every j in [1, step], i.e.
            //   r + k*j < cnt + j  =>  r + (k-1)*j < cnt  =>  j < (cnt - r)/(k-1).
            // The largest integer step with (k-1)*step <= cnt - r - 1 is
            //   step = (cnt - r - 1) / (k - 1).
            long long step = (cnt - r - 1) / (k - 1);
            if (step == 0) {
                // Cannot batch: do a single ordinary step.
                cnt += 1;
                r = (r + k) % cnt;
            } else {
                if (cnt + step > n) step = n - cnt;  // do not overshoot
                if (step == 0) {                      // safety: take one step
                    cnt += 1;
                    r = (r + k) % cnt;
                } else {
                    r += k * step;
                    cnt += step;
                    // After the batch no wrap was needed except possibly exactly
                    // hitting the boundary; reduce once to be safe.
                    if (r >= cnt) r %= cnt;
                }
            }
        }

        cout << (r + 1) << "\n";   // convert to 1-indexed survivor label
    }
    return 0;
}

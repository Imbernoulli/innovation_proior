#include <bits/stdc++.h>
using namespace std;
int main(){
    int n;
    if(!(cin>>n)) return 0;
    vector<long long> a(n);
    for(auto&x:a) cin>>x;
    // dp_take = best ending with i taken, dp_skip = best with i not taken; allow empty (>=0)
    long long take=LLONG_MIN/4, skip=0; // before any element, best-with-nothing = 0
    for(int i=0;i<n;i++){
        long long ntake = skip + a[i];      // take i: previous must be skipped
        long long nskip = max(skip, take);  // skip i: best of either before
        take=ntake; skip=nskip;
    }
    cout << max({take, skip, 0LL}) << "\n";
    return 0;
}

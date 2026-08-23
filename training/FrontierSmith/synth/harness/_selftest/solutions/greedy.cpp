// TIER: greedy
#include<bits/stdc++.h>
using namespace std;int main(){int n;scanf("%d",&n);vector<long long>p(n),c(n);vector<int>s;for(int i=0;i<n;i++){scanf("%lld %lld",&p[i],&c[i]);if(2*c[i]<p[i])s.push_back(i+1);}printf("%d\n",(int)s.size());for(int i=0;i<(int)s.size();i++)printf("%d%c",s[i]," \n"[i+1==(int)s.size()]);if(s.empty())printf("\n");}

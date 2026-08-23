#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc,char*argv[]){
  registerTestlibCmd(argc,argv);
  int n=inf.readInt();
  vector<long long> p(n),c(n); long long B=0;
  for(int i=0;i<n;i++){p[i]=inf.readInt();c[i]=inf.readInt();B+=p[i];}
  int k=ouf.readInt(0,n,"count");
  vector<char> cov(n,0);
  for(int i=0;i<k;i++){int id=ouf.readInt(1,n,"id");if(cov[id-1])quitf(_wa,"dup %d",id);cov[id-1]=1;}
  if(!ouf.seekEof())quitf(_wa,"trailing");
  long long F=0;
  for(int i=0;i<n;i++)F+= cov[i]?c[i]:p[i];
  double s=min(1000.0,100.0*(double)B/(double)max(1LL,F));
  quitp(s/1000.0,"OK F=%lld B=%lld Ratio: %.6f",F,B,s/1000.0);
  return 0;
}

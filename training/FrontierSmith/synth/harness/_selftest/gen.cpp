#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc,char*argv[]){
  registerGen(argc,argv,1);
  int t=atoi(argv[1]);
  int n = (t<=1)?3 : (t*40);
  printf("%d\n",n);
  for(int i=0;i<n;i++){int p=rnd.next(1,100),c=rnd.next(1,100);printf("%d %d\n",p,c);}
  return 0;
}

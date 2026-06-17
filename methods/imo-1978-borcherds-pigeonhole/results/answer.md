# Iterated Difference-Set Pigeonhole

**Theorem.** If the numbers $1,2,\ldots,1978$ are partitioned among six countries, then some country contains members numbered $x,y,z$, with $x$ and $y$ not necessarily distinct, such that $x+y=z$.

**Proof.** A country contains such a triple if and only if it contains two numbers $u<v$ whose positive difference $v-u$ is also in that country: then $u+(v-u)=v$. If $u=v-u$, this is exactly the equal-summand case $u+u=v$, so $v$ is twice a member of its own country. Suppose, for contradiction, that no country has this property.

Since $6\cdot329=1974<1978$, some country $C_1$ has at least $330$ members. Choose
$$a_1<a_2<\cdots<a_{330}$$
from $C_1$. The $329$ differences $a_i-a_1$ for $2\le i\le330$ are positive integers at most $1977$, hence are numbered members. None can lie in $C_1$, or $a_1+(a_i-a_1)=a_i$ would give the required sum. Thus all $329$ lie in the other five countries.

Since $5\cdot65=325<329$, some country $C_2$ contains at least $66$ of these differences. Write them as
$$b_1<b_2<\cdots<b_{66}.$$
The $65$ differences $b_j-b_1$ for $2\le j\le66$ cannot lie in $C_2$, or $b_1+(b_j-b_1)=b_j$. They also cannot lie in $C_1$, because if $b_j=a_{p(j)}-a_1$ and $b_1=a_{p(1)}-a_1$, then
$$b_j-b_1=a_{p(j)}-a_{p(1)},$$
so landing in $C_1$ would give a sum inside $C_1$. Hence all $65$ lie in the remaining four countries.

The same cancellation is the invariant. Whenever a selected list in a new country was made from previous-level differences over one common base, subtracting two entries cancels that base. Thus every fresh difference is not only a difference of two members of the current country, but also a difference of two members of every earlier country in the chain. If such a fresh difference lands in any country already touched, it supplies the missing third member of an internal difference relation there.

Since $4\cdot16=64<65$, some new country $C_3$ contains at least $17$ of the $65$ differences; taking differences from the smallest gives $16$ new positive differences. By the invariant, if any one lies in $C_1$, $C_2$, or $C_3$, it gives a within-country sum there. Therefore all $16$ lie in the remaining three countries.

Since $3\cdot5=15<16$, some new country $C_4$ contains at least $6$ of them; taking differences from the smallest gives $5$ differences. Unless one of these lands in $C_1,C_2,C_3$, or $C_4$ and gives the required sum, all $5$ lie in the two untouched countries. Since $2\cdot2=4<5$, some new country $C_5$ contains at least $3$ of those $5$; taking differences from the smallest gives $2$ differences. Unless one of these lands in one of $C_1,\ldots,C_5$ and gives the required sum, both are forced into the last country $C_6$. Call them $e_1<e_2$.

The count chain used above is
$$330\to329\to66\to65\to17\to16\to6\to5\to3\to2,$$
with strict pigeonhole inequalities
$$6\cdot329<1978,\quad 5\cdot65<329,\quad 4\cdot16<65,\quad 3\cdot5<16,\quad 2\cdot2<5.$$

Now $e_2-e_1$ is a positive numbered member, so it belongs to some country $C_k$. If $k=6$, then $e_1+(e_2-e_1)=e_2$ is a sum inside $C_6$. If $k<6$, trace the construction backward: $e_1$ and $e_2$ were formed from two members of $C_5$ by subtracting the same base, so $e_2-e_1$ is a difference of two members of $C_5$; those two members of $C_5$ were themselves formed from two members of $C_4$ by subtracting the same base, and the same cancellation continues back through $C_1$. Thus for the country $C_k$ containing $e_2-e_1$, there are members $u<v$ of $C_k$ with $v-u=e_2-e_1$. Since $e_2-e_1$ itself also lies in $C_k$, the relation $u+(e_2-e_1)=v$ is a within-country sum. In every case a country has the required within-country sum, contradicting the assumption.

Therefore some member's number is the sum of the numbers of two members from his own country, with the equal-summand case giving the "twice" alternative. $\blacksquare$

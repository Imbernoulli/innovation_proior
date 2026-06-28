import math

# Verify rerooting transition for product of subtree sizes.
# answer(r) = n! / prod_v size_r(v).
# Define P(r) = prod_v size_r(v).
# When moving root from u to child v (edge u-v), in the tree rooted at u:
#   size_u(v) = s  (subtree of v).  After rerooting at v:
#   size_v(u) = n - s, size_v(v) = n.  All other sizes unchanged.
# So P(v) = P(u) / (size_u(u) * size_u(v)) * (size_v(u)*size_v(v))
#         = P(u) / (n * s) * ((n-s) * n) = P(u) * (n-s) / s.
# Check on a tree.

def subtree_sizes(n, adj, root):
    size=[0]*n
    parent=[-1]*n
    order=[]
    visited=[False]*n
    stack=[root]; visited[root]=True
    while stack:
        u=stack.pop(); order.append(u)
        for w in adj[u]:
            if not visited[w]:
                visited[w]=True; parent[w]=u; stack.append(w)
    for u in reversed(order):
        size[u]=1
        for w in adj[u]:
            if parent[w]==u:
                size[u]+=size[w]
    return size

edges=[(0,1),(1,2),(1,3),(3,4),(3,5),(0,6)]
n=7
adj={i:[] for i in range(n)}
for a,b in edges:
    adj[a].append(b); adj[b].append(a)

P={}
for root in range(n):
    s=subtree_sizes(n,adj,root)
    p=1
    for x in s: p*=x
    P[root]=p

# verify transition along each edge
for (u,v) in edges:
    s_u = subtree_sizes(n,adj,u)
    # size of v's subtree when rooted at u
    sv = s_u[v]
    pred = P[u]*(n-sv)//sv
    print(f"edge {u}->{v}: P[u]={P[u]} sv={sv} predicted P[v]={pred} actual={P[v]} match={pred==P[v]}")

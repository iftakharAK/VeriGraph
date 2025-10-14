

from typing import List, Tuple, Dict
import math

def build_pair_indices(n: int):
    """All ordered pairs (i, j) with i != j."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]

def has_cycle(n: int, edges: List[Tuple[int,int]]) -> bool:
    """Detect cycles with DFS."""
    g = {i: [] for i in range(n)}
    for u, v in edges:
        g[u].append(v)
    visited, stack = [0]*n, [0]*n

    def dfs(u):
        visited[u] = 1
        stack[u] = 1
        for v in g[u]:
            if not visited[v] and dfs(v): return True
            if stack[v]: return True
        stack[u]=0
        return False

    return any(dfs(i) for i in range(n) if not visited[i])

def construct_graph(scores: Dict[Tuple[int,int], float],
                    n_nodes: int,
                    threshold: float = 0.5,
                    keep_best_acyclic: bool = True):
    """
    Create directed contradiction graph: edge i->j if score(i,j) >= threshold.
    If keep_best_acyclic=True, greedily remove lowest-score edges to break cycles.
    """
    # initial edges
    edges = [(i,j) for (i,j), s in scores.items() if s >= threshold and i != j]
    if not keep_best_acyclic:
        return edges

    # break cycles by removing weakest edges first
    edges_sorted = sorted(edges, key=lambda e: scores[e], reverse=True)
    kept = []
    for e in edges_sorted:
        kept.append(e)
        if has_cycle(n_nodes, kept):
            kept.pop()  # drop this edge to preserve acyclicity
    return kept
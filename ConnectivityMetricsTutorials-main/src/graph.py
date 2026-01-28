"""
Graph theory functions for connectivity analysis.

This module provides functions for analyzing brain connectivity matrices
as graphs, including thresholding, node metrics, clustering, path lengths,
small-world analysis, and hub detection.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional


# =============================================================================
# THRESHOLDING FUNCTIONS
# =============================================================================

def threshold_matrix_absolute(
    matrix: NDArray[np.float64],
    threshold: float
) -> NDArray[np.float64]:
    """
    Convert weighted matrix to binary using absolute threshold.
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Weighted connectivity matrix.
    threshold : float
        Minimum absolute value to keep edge.
        
    Returns
    -------
    NDArray[np.float64]
        Binary adjacency matrix.
    """
    binary = (np.abs(matrix) > threshold).astype(np.float64)
    np.fill_diagonal(binary, 0)  # No self-connections
    return binary


def threshold_matrix_proportional(
    matrix: NDArray[np.float64],
    density: float
) -> NDArray[np.float64]:
    """
    Convert weighted matrix to binary keeping top proportion of edges.
    
    Parameters
    ----------
    matrix : NDArray[np.float64]
        Weighted connectivity matrix.
    density : float
        Proportion of edges to keep (0-1).
        
    Returns
    -------
    NDArray[np.float64]
        Binary adjacency matrix with specified density.
    """
    n = matrix.shape[0]
    # Get upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    values = np.abs(matrix[triu_indices])
    
    # Find threshold for desired density
    n_edges_to_keep = int(np.ceil(density * len(values)))
    if n_edges_to_keep == 0:
        return np.zeros_like(matrix)
    
    sorted_values = np.sort(values)[::-1]  # Descending
    threshold = sorted_values[min(n_edges_to_keep - 1, len(sorted_values) - 1)]
    
    # Apply threshold
    binary = (np.abs(matrix) >= threshold).astype(np.float64)
    np.fill_diagonal(binary, 0)
    return binary


def get_graph_density(adjacency: NDArray[np.float64]) -> float:
    """
    Compute graph density (proportion of possible edges that exist).
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    float
        Density in range [0, 1].
    """
    n = adjacency.shape[0]
    n_possible = n * (n - 1)  # Exclude diagonal
    n_actual = np.sum(adjacency > 0) - np.trace(adjacency > 0)  # Exclude diagonal
    return n_actual / n_possible if n_possible > 0 else 0.0


# =============================================================================
# NODE METRICS
# =============================================================================

def compute_degree(adjacency: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute degree (number of connections) for each node.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Degree of each node.
    """
    binary = (adjacency > 0).astype(np.float64)
    np.fill_diagonal(binary, 0)
    return np.sum(binary, axis=1)


def compute_strength(
    weighted_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute strength (sum of edge weights) for each node.
    
    Parameters
    ----------
    weighted_matrix : NDArray[np.float64]
        Weighted adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Strength of each node.
    """
    matrix = weighted_matrix.copy()
    np.fill_diagonal(matrix, 0)
    return np.sum(np.abs(matrix), axis=1)


def compute_in_out_degree(
    adjacency: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute in-degree and out-degree for directed graphs.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Adjacency matrix (may be asymmetric).
        
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        (in_degree, out_degree) arrays.
    """
    binary = (adjacency > 0).astype(np.float64)
    np.fill_diagonal(binary, 0)
    in_degree = np.sum(binary, axis=0)   # Sum columns
    out_degree = np.sum(binary, axis=1)  # Sum rows
    return in_degree, out_degree


# =============================================================================
# CLUSTERING COEFFICIENT
# =============================================================================

def compute_clustering_coefficient(
    adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute local clustering coefficient for each node.
    
    The clustering coefficient measures how connected a node's
    neighbors are to each other.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Clustering coefficient for each node (0-1).
    """
    binary = (adjacency > 0).astype(np.float64)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    
    clustering = np.zeros(n)
    
    for i in range(n):
        # Get neighbors of node i
        neighbors = np.where(binary[i, :] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            # Need at least 2 neighbors to form triangles
            clustering[i] = 0.0
            continue
        
        # Count edges between neighbors
        n_triangles = 0
        for j in range(len(neighbors)):
            for k_idx in range(j + 1, len(neighbors)):
                if binary[neighbors[j], neighbors[k_idx]] > 0:
                    n_triangles += 1
        
        # Maximum possible triangles
        max_triangles = k * (k - 1) / 2
        clustering[i] = n_triangles / max_triangles
    
    return clustering


def compute_global_clustering(adjacency: NDArray[np.float64]) -> float:
    """
    Compute global clustering coefficient (average of local coefficients).
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    float
        Global clustering coefficient.
    """
    local_cc = compute_clustering_coefficient(adjacency)
    return float(np.mean(local_cc))


# =============================================================================
# PATH LENGTH AND EFFICIENCY
# =============================================================================

def compute_shortest_paths(
    adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute shortest path lengths between all pairs of nodes.
    
    Uses Floyd-Warshall algorithm.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Distance matrix (inf for disconnected pairs).
    """
    binary = (adjacency > 0).astype(np.float64)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    
    # Initialize distance matrix
    dist = np.full((n, n), np.inf)
    dist[binary > 0] = 1  # Direct connections have distance 1
    np.fill_diagonal(dist, 0)
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    return dist


def compute_characteristic_path_length(
    adjacency: NDArray[np.float64]
) -> Tuple[float, bool]:
    """
    Compute characteristic path length (average shortest path).
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    Tuple[float, bool]
        (path_length, is_connected). Path length is inf if disconnected.
    """
    dist = compute_shortest_paths(adjacency)
    n = dist.shape[0]
    
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    path_lengths = dist[triu_indices]
    
    # Check connectivity
    is_connected = not np.any(np.isinf(path_lengths))
    
    if is_connected:
        return float(np.mean(path_lengths)), True
    else:
        return float('inf'), False


def compute_global_efficiency(adjacency: NDArray[np.float64]) -> float:
    """
    Compute global efficiency (average inverse path length).
    
    Unlike path length, efficiency handles disconnected graphs gracefully.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    float
        Global efficiency in range [0, 1].
    """
    dist = compute_shortest_paths(adjacency)
    n = dist.shape[0]
    
    # Compute inverse distances (0 for infinite)
    with np.errstate(divide='ignore'):
        inv_dist = 1.0 / dist
    inv_dist[np.isinf(inv_dist)] = 0
    np.fill_diagonal(inv_dist, 0)
    
    # Average over all pairs
    n_pairs = n * (n - 1)
    return float(np.sum(inv_dist) / n_pairs) if n_pairs > 0 else 0.0


# =============================================================================
# GRAPH GENERATORS (for comparison)
# =============================================================================

def generate_random_graph(
    n_nodes: int,
    density: float,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generate Erdős-Rényi random graph.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    density : float
        Probability of edge between any two nodes.
    seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    NDArray[np.float64]
        Binary adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random symmetric matrix
    adj = np.random.rand(n_nodes, n_nodes)
    adj = (adj + adj.T) / 2  # Symmetrize
    adj = (adj < density).astype(np.float64)
    np.fill_diagonal(adj, 0)
    
    return adj


def generate_lattice_graph(
    n_nodes: int,
    k_neighbors: int = 2
) -> NDArray[np.float64]:
    """
    Generate regular lattice (ring) graph.
    
    Each node is connected to k nearest neighbors on each side.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    k_neighbors : int
        Number of neighbors on each side.
        
    Returns
    -------
    NDArray[np.float64]
        Binary adjacency matrix.
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for k in range(1, k_neighbors + 1):
            # Connect to k neighbors on each side (with wraparound)
            j_right = (i + k) % n_nodes
            j_left = (i - k) % n_nodes
            adj[i, j_right] = 1
            adj[i, j_left] = 1
    
    return adj


def generate_small_world_graph(
    n_nodes: int,
    k_neighbors: int = 2,
    rewire_prob: float = 0.1,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generate Watts-Strogatz small-world graph.
    
    Starts with a lattice and rewires edges with given probability.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    k_neighbors : int
        Initial neighbors on each side (before rewiring).
    rewire_prob : float
        Probability of rewiring each edge.
    seed : Optional[int]
        Random seed for reproducibility.
        
    Returns
    -------
    NDArray[np.float64]
        Binary adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with lattice
    adj = generate_lattice_graph(n_nodes, k_neighbors)
    
    # Rewire edges
    for i in range(n_nodes):
        for k in range(1, k_neighbors + 1):
            j = (i + k) % n_nodes
            if np.random.rand() < rewire_prob:
                # Remove edge (i, j)
                adj[i, j] = 0
                adj[j, i] = 0
                
                # Add edge to random node (not self, not already connected)
                candidates = np.where((adj[i, :] == 0) & (np.arange(n_nodes) != i))[0]
                if len(candidates) > 0:
                    new_j = np.random.choice(candidates)
                    adj[i, new_j] = 1
                    adj[new_j, i] = 1
    
    return adj


# =============================================================================
# HUB DETECTION
# =============================================================================

def compute_betweenness_centrality(
    adjacency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute betweenness centrality for each node.
    
    Betweenness measures how often a node lies on shortest paths
    between other nodes.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Betweenness centrality for each node.
    """
    binary = (adjacency > 0).astype(np.float64)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    
    betweenness = np.zeros(n)
    
    # For each source node
    for s in range(n):
        # BFS to find shortest paths
        dist = np.full(n, np.inf)
        dist[s] = 0
        n_paths = np.zeros(n)  # Number of shortest paths
        n_paths[s] = 1
        predecessors = [[] for _ in range(n)]
        
        # BFS queue
        queue = [s]
        order = []  # Nodes in order of discovery
        
        while queue:
            v = queue.pop(0)
            order.append(v)
            
            neighbors = np.where(binary[v, :] > 0)[0]
            for w in neighbors:
                # First time reaching w
                if np.isinf(dist[w]):
                    dist[w] = dist[v] + 1
                    queue.append(w)
                
                # Shortest path to w through v
                if dist[w] == dist[v] + 1:
                    n_paths[w] += n_paths[v]
                    predecessors[w].append(v)
        
        # Back-propagate dependencies
        dependency = np.zeros(n)
        for w in reversed(order):
            for v in predecessors[w]:
                if n_paths[w] > 0:
                    dependency[v] += (n_paths[v] / n_paths[w]) * (1 + dependency[w])
            if w != s:
                betweenness[w] += dependency[w]
    
    # Normalize
    if n > 2:
        betweenness /= ((n - 1) * (n - 2))
    
    return betweenness


def identify_hubs(
    adjacency: NDArray[np.float64],
    method: str = "degree",
    threshold_percentile: float = 80.0
) -> Dict[str, Any]:
    """
    Identify hub nodes in the network.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Binary or weighted adjacency matrix.
    method : str
        Method for hub identification: "degree", "strength", or "betweenness".
    threshold_percentile : float
        Percentile threshold for hub classification.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'scores', 'hub_indices', and 'threshold'.
    """
    if method == "degree":
        scores = compute_degree(adjacency)
    elif method == "strength":
        scores = compute_strength(adjacency)
    elif method == "betweenness":
        scores = compute_betweenness_centrality(adjacency)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    threshold = np.percentile(scores, threshold_percentile)
    hub_indices = np.where(scores >= threshold)[0]
    
    return {
        "scores": scores,
        "hub_indices": hub_indices,
        "threshold": threshold,
        "method": method
    }


# =============================================================================
# LAYOUT ALGORITHMS
# =============================================================================

def spring_layout(
    adjacency: NDArray[np.float64],
    n_iterations: int = 50,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Compute spring (force-directed) layout for graph visualization.
    
    Connected nodes attract each other, all nodes repel.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Adjacency matrix.
    n_iterations : int
        Number of iterations.
    seed : Optional[int]
        Random seed for initial positions.
        
    Returns
    -------
    NDArray[np.float64]
        Node positions (n_nodes, 2).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = adjacency.shape[0]
    binary = (adjacency > 0).astype(np.float64)
    
    # Initialize random positions
    pos = np.random.rand(n, 2) * 2 - 1
    
    k_attract = 0.1  # Spring constant for edges
    k_repel = 0.5    # Repulsion constant
    
    for iteration in range(n_iterations):
        forces = np.zeros((n, 2))
        
        # Repulsion between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff) + 0.01
                force = k_repel * diff / (dist ** 2)
                forces[i] += force
                forces[j] -= force
        
        # Attraction along edges
        for i in range(n):
            for j in range(i + 1, n):
                if binary[i, j] > 0:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    force = k_attract * diff * dist
                    forces[i] += force
                    forces[j] -= force
        
        # Update positions with decreasing step size
        step = 0.1 * (1 - iteration / n_iterations)
        pos += step * forces
        
        # Keep in bounds
        pos = np.clip(pos, -2, 2)
    
    # Center
    pos -= pos.mean(axis=0)
    
    # Normalize to unit circle
    max_dist = np.max(np.linalg.norm(pos, axis=1))
    if max_dist > 0:
        pos /= max_dist
    
    return pos


def spectral_layout(adjacency: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute spectral layout using eigenvectors of Laplacian.
    
    Parameters
    ----------
    adjacency : NDArray[np.float64]
        Adjacency matrix.
        
    Returns
    -------
    NDArray[np.float64]
        Node positions (n_nodes, 2).
    """
    binary = (adjacency > 0).astype(np.float64)
    n = binary.shape[0]
    
    # Compute Laplacian: L = D - A
    degrees = np.sum(binary, axis=1)
    laplacian = np.diag(degrees) - binary
    
    # Get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    # Use 2nd and 3rd eigenvectors (1st is constant)
    if n >= 3:
        pos = eigenvectors[:, 1:3]
    else:
        pos = np.random.rand(n, 2)
    
    # Normalize
    pos -= pos.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pos, axis=1))
    if max_dist > 0:
        pos /= max_dist
    
    return pos


def circular_layout(n_nodes: int) -> NDArray[np.float64]:
    """
    Compute circular layout for graph visualization.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
        
    Returns
    -------
    NDArray[np.float64]
        Node positions (n_nodes, 2).
    """
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) - np.pi / 2
    return np.column_stack([np.cos(angles), np.sin(angles)])

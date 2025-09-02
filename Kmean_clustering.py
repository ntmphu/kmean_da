import numpy as np
from collections import defaultdict

def kmeans(X, k, max_iters=100, tol=1e-4):
    n = X.shape[0]
    rng = np.random.RandomState(42)  # Create a separate random generator
    initial_indices = rng.choice(n, k, replace=False) 
    initial_centroids = X[initial_indices]
    centroids = initial_centroids.copy()
    cluster_labels_all = []
    cluster_members_all = []

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        cluster_labels_all.append(labels.copy())
        cluster_members = defaultdict(set)
        for idx, label in enumerate(labels):
            cluster_members[label].add(idx)
        cluster_members_all.append(cluster_members)
        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            instances_in_cluster = X[labels == cluster]
            if len(instances_in_cluster) > 0:
                new_centroids[cluster] = instances_in_cluster.mean(axis=0)
            else:
                new_centroids[cluster] = centroids[cluster]
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return initial_indices, cluster_labels_all, cluster_members_all

def Kmean_selected_uv(X, K, u, v, n_s, n_t):
    initial_centroids, labels_all, members_all = kmeans(X, K)
    idx_cluster_u = []
    idx_cluster_v = []
    
    for i in range(n_t): 
        if labels_all[-1][n_s + i] == u:
            idx_cluster_u.append(i)
        elif labels_all[-1][n_s + i] == v:
            idx_cluster_v.append(i)
    #if len(idx_cluster_u) == 0 or len(idx_cluster_v) == 0:
    #    return None
            
    idx_cluster_u = np.array(idx_cluster_u)
    idx_cluster_v = np.array(idx_cluster_v)
    return initial_centroids, labels_all, members_all, idx_cluster_u, idx_cluster_v 
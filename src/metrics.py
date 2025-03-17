import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import warnings

def calculate_davis_bouldin_index(X, labels, centroids):
    """
    Calculate the Davis-Bouldin Index for clustering evaluation.
    Lower values indicate better clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
        
    Returns:
    --------
    db_index : float
        Davis-Bouldin Index value
    """
    n_clusters = len(centroids)
    
    if n_clusters <= 1:
        return 0.0
    
    # Calculate cluster dispersion (average distance to centroid)
    cluster_dispersion = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = np.sqrt(np.sum((cluster_points - centroids[i])**2, axis=1))
            cluster_dispersion[i] = np.mean(distances)
    
    # Calculate the Davis-Bouldin Index
    db_sum = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # Calculate distance between centroids
                centroid_distance = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
                if centroid_distance > 0:  # Avoid division by zero
                    # Calculate the ratio of cluster dispersions to centroid distance
                    ratio = (cluster_dispersion[i] + cluster_dispersion[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio
    
    return db_sum / n_clusters

def calculate_dunn_index(X, labels, centroids):
    """
    Calculate the Dunn Index for clustering evaluation.
    Higher values indicate better clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
        
    Returns:
    --------
    dunn_index : float
        Dunn Index value
    """
    n_clusters = len(centroids)
    
    if n_clusters <= 1:
        return 0.0
    
    # Calculate minimum inter-cluster distance
    min_inter_cluster_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            # For efficiency, we'll use the distance between centroids as the inter-cluster distance
            dist = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            min_inter_cluster_dist = min(min_inter_cluster_dist, dist)
    
    # Calculate maximum intra-cluster distance
    max_intra_cluster_dist = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            # For efficiency, we'll use the maximum distance to centroid as the intra-cluster distance
            distances = np.sqrt(np.sum((cluster_points - centroids[i])**2, axis=1))
            max_dist = np.max(distances)
            max_intra_cluster_dist = max(max_intra_cluster_dist, max_dist)
    
    # Calculate Dunn Index
    if max_intra_cluster_dist > 0:  # Avoid division by zero
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
    else:
        dunn_index = 0.0
    
    return dunn_index

def calculate_c_index(X, labels):
    """
    Calculate the C-Index for clustering evaluation.
    Lower values indicate better clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
        
    Returns:
    --------
    c_index : float
        C-Index value
    """
    n_clusters = len(np.unique(labels))
    n_samples = len(X)
    
    if n_clusters <= 1 or n_samples <= 1:
        return 0.0
    
    # Calculate all pairwise distances
    distances = squareform(pdist(X, metric='euclidean'))
    
    # Collect within-cluster distances
    within_cluster_distances = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        
        if len(cluster_indices) > 1:
            for idx1 in range(len(cluster_indices)):
                for idx2 in range(idx1+1, len(cluster_indices)):
                    within_cluster_distances.append(
                        distances[cluster_indices[idx1], cluster_indices[idx2]]
                    )
    
    if not within_cluster_distances:
        return 0.0  # If no within-cluster distances exist
    
    # Sort all pairwise distances
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warning about upper triangular matrix
        sorted_distances = np.sort(distances[np.triu_indices(distances.shape[0], k=1)])
    
    # Get the sum of the smallest and largest distances
    nw = len(within_cluster_distances)
    sum_w = sum(within_cluster_distances)
    
    # Sum of nw smallest distances
    min_sum = sum(sorted_distances[:nw])
    
    # Sum of nw largest distances
    max_sum = sum(sorted_distances[-nw:])
    
    # Calculate C-Index
    if max_sum != min_sum:  # Avoid division by zero
        c_index = (sum_w - min_sum) / (max_sum - min_sum)
    else:
        c_index = 0.0
    
    return c_index

def calculate_goodman_kruskal_index(labels, true_labels):
    """
    Calculate the Goodman-Kruskal Index for clustering evaluation.
    Higher values indicate better clustering.
    
    Parameters:
    -----------
    labels : array-like, shape (n_samples,)
        Predicted cluster labels
    true_labels : array-like, shape (n_samples,)
        Ground truth labels
        
    Returns:
    --------
    gk_index : float
        Goodman-Kruskal Index value
    """
    n = len(labels)
    
    if n <= 1:
        return 0.0
    
    # For large datasets, use sampling to reduce computation
    if n > 10000:
        print("Large dataset detected, using sampling for Goodman-Kruskal calculation...")
        idx = np.random.choice(n, size=10000, replace=False)
        labels = labels[idx]
        true_labels = true_labels[idx]
        n = 10000
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    
    # Optimization: calculate concordant and discordant pairs more efficiently
    for i in range(n):
        # Compare each pair only once
        for j in range(i+1, n):
            # Check if pair is concordant or discordant
            if (labels[i] == labels[j] and true_labels[i] == true_labels[j]) or \
               (labels[i] != labels[j] and true_labels[i] != true_labels[j]):
                concordant += 1
            else:
                discordant += 1
    
    # Calculate Goodman-Kruskal Index
    if concordant + discordant > 0:  # Avoid division by zero
        gk_index = (concordant - discordant) / (concordant + discordant)
    else:
        gk_index = 0.0
    
    return gk_index

def calculate_silhouette_score(X, labels):
    """
    Calculate the mean Silhouette Coefficient for all samples.
    Higher values indicate better clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
        
    Returns:
    --------
    silhouette_score : float
        Mean Silhouette Coefficient
    """
    # Import here to avoid circular imports if other modules import metrics
    from sklearn.metrics import silhouette_score
    
    if len(np.unique(labels)) <= 1:
        return 0.0
    
    return silhouette_score(X, labels, metric='euclidean')

def calculate_all_metrics(X, labels, centroids, true_labels=None):
    """
    Calculate all clustering validation metrics.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
    true_labels : array-like, shape (n_samples,), optional
        Ground truth labels for supervised evaluation
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all evaluation metrics
    """
    metrics = {}
    
    # Calculate Davis-Bouldin Index
    metrics['davis_bouldin_index'] = calculate_davis_bouldin_index(X, labels, centroids)
    
    # Calculate Dunn Index
    metrics['dunn_index'] = calculate_dunn_index(X, labels, centroids)
    
    # Calculate C-Index
    metrics['c_index'] = calculate_c_index(X, labels)
    
    # Calculate Silhouette Score
    metrics['silhouette_score'] = calculate_silhouette_score(X, labels)
    
    # Calculate Goodman-Kruskal Index if true labels are available
    if true_labels is not None:
        metrics['goodman_kruskal_index'] = calculate_goodman_kruskal_index(labels, true_labels)
    
    return metrics

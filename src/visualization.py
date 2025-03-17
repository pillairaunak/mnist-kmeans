import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def visualize_centroids(kmeans, digit_size=28, n_cols=5):
    """
    Visualize the centroids (cluster centers) as images.
    
    Parameters:
    -----------
    kmeans : KMeans object
        Fitted KMeans estimator
    digit_size : int, default=28
        Size of digit images
    n_cols : int, default=5
        Number of columns in the plot
    """
    n_clusters = kmeans.n_clusters
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, centroid in enumerate(kmeans.centroids):
        if i < len(axes):
            axes[i].imshow(centroid.reshape(digit_size, digit_size), cmap='gray')
            axes[i].set_title(f'Centroid {i}')
            axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'K-Means Centroids (K={n_clusters})', y=1.02, fontsize=16)
    plt.show()

def plot_digit_clusters(X, labels, centroids, sample_size=1000, digit_size=28, n_per_cluster=2):
    """
    Plot sample digits from each cluster.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
    sample_size : int, default=1000
        Number of samples to use for visualization
    digit_size : int, default=28
        Size of digit images
    n_per_cluster : int, default=2
        Number of sample digits to show per cluster
    """
    n_clusters = len(centroids)
    
    # Use a subset of data if it's too large
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_subset = X[indices]
        labels_subset = labels[indices]
    else:
        X_subset = X
        labels_subset = labels
    
    # Create a figure
    fig, axes = plt.subplots(n_clusters, n_per_cluster+1, figsize=(3*(n_per_cluster+1), 3*n_clusters))
    
    for i in range(n_clusters):
        # Plot centroid
        axes[i, 0].imshow(centroids[i].reshape(digit_size, digit_size), cmap='gray')
        axes[i, 0].set_title(f'Centroid {i}')
        axes[i, 0].axis('off')
        
        # Get digits from this cluster
        cluster_indices = np.where(labels_subset == i)[0]
        
        # Plot sample digits
        for j in range(n_per_cluster):
            if j < len(cluster_indices):
                idx = cluster_indices[j]
                axes[i, j+1].imshow(X_subset[idx].reshape(digit_size, digit_size), cmap='gray')
                axes[i, j+1].set_title(f'Sample {j+1}')
                axes[i, j+1].axis('off')
            else:
                axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Cluster Centroids and Sample Digits (K={n_clusters})', y=1.02, fontsize=16)
    plt.show()

def plot_cluster_distribution(labels, true_labels=None):
    """
    Plot the distribution of clusters.
    
    Parameters:
    -----------
    labels : array-like, shape (n_samples,)
        Cluster labels
    true_labels : array-like, shape (n_samples,), optional
        Ground truth labels
    """
    n_clusters = len(np.unique(labels))
    
    plt.figure(figsize=(12, 5))
    
    # Plot cluster distribution
    plt.subplot(1, 2, 1)
    cluster_counts = np.bincount(labels)
    plt.bar(range(n_clusters), cluster_counts)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of samples')
    plt.xticks(range(n_clusters))
    
    # Plot true label distribution within clusters if available
    if true_labels is not None:
        plt.subplot(1, 2, 2)
        
        # Create a cross-tabulation of clusters vs true labels
        df = pd.DataFrame({'Cluster': labels, 'Digit': true_labels})
        ct = pd.crosstab(df['Cluster'], df['Digit'])
        
        # Plot as a heatmap
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
        plt.title('True Label Distribution within Clusters')
        plt.xlabel('Digit')
        plt.ylabel('Cluster')
    
    plt.tight_layout()
    plt.show()

def plot_clustering_metrics(k_values, metrics_dict, metric_names=None):
    """
    Plot clustering metrics for different K values.
    
    Parameters:
    -----------
    k_values : list
        List of K values
    metrics_dict : dict
        Dictionary with K values as keys and dictionaries of metrics as values
    metric_names : list, optional
        List of metric names to plot. If None, all metrics are plotted.
    """
    if metric_names is None:
        # Get all metric names from the first entry
        metric_names = list(metrics_dict[k_values[0]].keys())
    
    # Calculate the number of rows and columns for subplots
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, metric in enumerate(metric_names):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Extract metric values for each K
        values = [metrics_dict[k][metric] if metric in metrics_dict[k] else np.nan for k in k_values]
        
        plt.plot(k_values, values, marker='o', linestyle='-')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Metric Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(k_values)
    
    plt.tight_layout()
    plt.show()

def visualize_pca_clusters(X, labels, centroids, n_components=2):
    """
    Visualize clusters in PCA-reduced space.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
    n_components : int, default=2
        Number of PCA components
    """
    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Scatter plot of data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                alpha=0.5, s=10, edgecolors='none')
    
    # Plot centroids
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, 
                marker='X', edgecolors='black', linewidths=2)
    
    # Add cluster labels to centroids
    for i, (x, y) in enumerate(centroids_pca):
        plt.text(x, y, str(i), fontsize=15, fontweight='bold', 
                 ha='center', va='center', color='white')
    
    plt.title('PCA Visualization of Clusters', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return the PCA components for further analysis if needed
    return X_pca, centroids_pca, pca.explained_variance_ratio_

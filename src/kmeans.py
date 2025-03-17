import numpy as np
from scipy.spatial.distance import cdist
import random
from tqdm import tqdm

class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    Parameters:
    -----------
    n_clusters : int, default=10
        Number of clusters (K)
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def _init_centroids(self, X):
        """
        Initialize centroids by randomly selecting data points.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Initial centroids
        """
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def fit(self, X):
        """
        Fit K-Means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Initialize centroids
        self.centroids = self._init_centroids(X)
        prev_centroids = np.zeros_like(self.centroids)
        
        # Main K-Means loop
        for iteration in tqdm(range(self.max_iter), desc="K-Means Iterations"):
            # Assign points to clusters
            distances = cdist(X, self.centroids, metric='euclidean')
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(self.n_clusters):
                if np.sum(self.labels == i) > 0:  # Check if cluster is not empty
                    self.centroids[i] = np.mean(X[self.labels == i], axis=0)
            
            # Check for convergence
            centroid_change = np.sum(np.abs(self.centroids - prev_centroids))
            if centroid_change < self.tol:
                print(f"Converged after {iteration+1} iterations")
                break
                
            prev_centroids = self.centroids.copy()
        
        # Calculate inertia (sum of squared distances to the nearest centroid)
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum(np.square(cdist(cluster_points, 
                                                       self.centroids[i].reshape(1, -1), 
                                                       metric='euclidean')))
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Predicted cluster labels
        """
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Fit K-Means clustering and predict cluster labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels

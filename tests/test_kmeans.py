import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kmeans import KMeans
from src.metrics import calculate_davis_bouldin_index, calculate_dunn_index, calculate_c_index

class TestKMeans(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for testing
        self.X = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # Cluster 1
            [5, 5], [6, 5], [5, 6], [6, 6],  # Cluster 2
            [10, 0], [11, 0], [10, 1], [11, 1]  # Cluster 3
        ])
        
        # Expected cluster assignments (3 clusters)
        self.expected_labels_3 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Expected centroids for 3 clusters
        self.expected_centroids_3 = np.array([
            [0.5, 0.5],  # Centroid for cluster 1
            [5.5, 5.5],  # Centroid for cluster 2
            [10.5, 0.5]  # Centroid for cluster 3
        ])

    def test_kmeans_initialization(self):
        """Test KMeans initialization with different parameters"""
        kmeans = KMeans(n_clusters=3, max_iter=100, tol=1e-4, random_state=42)
        
        self.assertEqual(kmeans.n_clusters, 3)
        self.assertEqual(kmeans.max_iter, 100)
        self.assertEqual(kmeans.tol, 1e-4)
        self.assertEqual(kmeans.random_state, 42)
        self.assertIsNone(kmeans.centroids)
        self.assertIsNone(kmeans.labels)
        self.assertIsNone(kmeans.inertia_)

    def test_kmeans_fit_predict(self):
        """Test KMeans fitting and prediction"""
        # Initialize KMeans with fixed initial centroids
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Override the _init_centroids method to use fixed centroids for testing
        def fixed_init_centroids(X):
            return np.array([
                [0.5, 0.5],   # Near cluster 1
                [5.5, 5.5],   # Near cluster 2
                [10.5, 0.5]   # Near cluster 3
            ])
        
        kmeans._init_centroids = fixed_init_centroids
        
        # Fit the model
        kmeans.fit(self.X)
        
        # Check if centroids are calculated correctly
        np.testing.assert_allclose(kmeans.centroids, self.expected_centroids_3, rtol=1e-5)
        
        # Check if labels are assigned correctly
        np.testing.assert_array_equal(kmeans.labels, self.expected_labels_3)
        
        # Test predict method
        new_points = np.array([[0.5, 0.5], [5.5, 5.5], [10.5, 0.5]])
        predicted_labels = kmeans.predict(new_points)
        np.testing.assert_array_equal(predicted_labels, [0, 1, 2])
        
        # Test fit_predict method
        labels = kmeans.fit_predict(self.X)
        np.testing.assert_array_equal(labels, self.expected_labels_3)

    def test_kmeans_inertia(self):
        """Test KMeans inertia calculation"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Override the _init_centroids method to use fixed centroids for testing
        def fixed_init_centroids(X):
            return np.array([
                [0.5, 0.5],   # Near cluster 1
                [5.5, 5.5],   # Near cluster 2
                [10.5, 0.5]   # Near cluster 3
            ])
        
        kmeans._init_centroids = fixed_init_centroids
        
        # Fit the model
        kmeans.fit(self.X)
        
        # Calculate expected inertia manually
        expected_inertia = 0
        for i in range(len(self.X)):
            centroid = kmeans.centroids[kmeans.labels[i]]
            expected_inertia += np.sum((self.X[i] - centroid) ** 2)
        
        # Check if inertia is calculated correctly
        self.assertAlmostEqual(kmeans.inertia_, expected_inertia, places=5)

    def test_davis_bouldin_index(self):
        """Test Davis-Bouldin Index calculation"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Override the _init_centroids method to use fixed centroids for testing
        def fixed_init_centroids(X):
            return np.array([
                [0.5, 0.5],   # Near cluster 1
                [5.5, 5.5],   # Near cluster 2
                [10.5, 0.5]   # Near cluster 3
            ])
        
        kmeans._init_centroids = fixed_init_centroids
        
        # Fit the model
        kmeans.fit(self.X)
        
        # Calculate Davis-Bouldin Index
        db_index = calculate_davis_bouldin_index(self.X, kmeans.labels, kmeans.centroids)
        
        # The index should be a non-negative float
        self.assertGreaterEqual(db_index, 0.0)
        
        # For well-separated clusters, DB index should be low
        self.assertLess(db_index, 1.0)

    def test_dunn_index(self):
        """Test Dunn Index calculation"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Override the _init_centroids method to use fixed centroids for testing
        def fixed_init_centroids(X):
            return np.array([
                [0.5, 0.5],   # Near cluster 1
                [5.5, 5.5],   # Near cluster 2
                [10.5, 0.5]   # Near cluster 3
            ])
        
        kmeans._init_centroids = fixed_init_centroids
        
        # Fit the model
        kmeans.fit(self.X)
        
        # Calculate Dunn Index
        dunn_index = calculate_dunn_index(self.X, kmeans.labels, kmeans.centroids)
        
        # The index should be a non-negative float
        self.assertGreaterEqual(dunn_index, 0.0)
        
        # For well-separated clusters, Dunn index should be high
        self.assertGreater(dunn_index, 0.5)

    def test_c_index(self):
        """Test C-Index calculation"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Override the _init_centroids method to use fixed centroids for testing
        def fixed_init_centroids(X):
            return np.array([
                [0.5, 0.5],   # Near cluster 1
                [5.5, 5.5],   # Near cluster 2
                [10.5, 0.5]   # Near cluster 3
            ])
        
        kmeans._init_centroids = fixed_init_centroids
        
        # Fit the model
        kmeans.fit(self.X)
        
        # Calculate C-Index
        c_index = calculate_c_index(self.X, kmeans.labels)
        
        # The index should be between 0 and 1
        self.assertGreaterEqual(c_index, 0.0)
        self.assertLessEqual(c_index, 1.0)
        
        # For well-separated clusters, C-index should be low
        self.assertLess(c_index, 0.5)

if __name__ == '__main__':
    unittest.main()

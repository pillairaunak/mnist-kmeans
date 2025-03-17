import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    calculate_davis_bouldin_index,
    calculate_dunn_index,
    calculate_c_index,
    calculate_goodman_kruskal_index,
    calculate_silhouette_score,
    calculate_all_metrics
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with 3 well-separated clusters
        self.X = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # Cluster 1
            [5, 5], [6, 5], [5, 6], [6, 6],  # Cluster 2
            [10, 0], [11, 0], [10, 1], [11, 1]  # Cluster 3
        ])
        
        # Labels for 3 well-separated clusters
        self.labels_good = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Centroids for 3 well-separated clusters
        self.centroids_good = np.array([
            [0.5, 0.5],   # Centroid for cluster 1
            [5.5, 5.5],   # Centroid for cluster 2
            [10.5, 0.5]   # Centroid for cluster 3
        ])
        
        # Labels for poorly clustered data (randomly assigned)
        self.labels_poor = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        # Centroids for poorly clustered data
        self.centroids_poor = np.array([
            [2.75, 0.25],  # Centroid for cluster 1
            [6.0, 2.0],    # Centroid for cluster 2
            [5.0, 2.33]    # Centroid for cluster 3
        ])
        
        # True class labels (for supervised evaluation)
        self.true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    def test_davis_bouldin_index(self):
        """Test Davis-Bouldin Index calculation"""
        # Good clustering should have a lower Davis-Bouldin index
        db_good = calculate_davis_bouldin_index(self.X, self.labels_good, self.centroids_good)
        db_poor = calculate_davis_bouldin_index(self.X, self.labels_poor, self.centroids_poor)
        
        # Both should be non-negative
        self.assertGreaterEqual(db_good, 0.0)
        self.assertGreaterEqual(db_poor, 0.0)
        
        # Good clustering should have a lower index than poor clustering
        self.assertLess(db_good, db_poor)
        
        # Test with single cluster
        single_label = np.zeros(len(self.X))
        single_centroid = np.array([np.mean(self.X, axis=0)])
        db_single = calculate_davis_bouldin_index(self.X, single_label, single_centroid)
        self.assertEqual(db_single, A=0.0)
        
        # Test with empty clusters
        labels_with_empty = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        centroids_with_empty = np.array([
            [0.5, 0.5],    # Cluster 1
            [5.5, 5.5],    # Cluster 2
            [10.5, 0.5],   # Cluster 3
            [20.0, 20.0]   # Empty cluster
        ])
        db_with_empty = calculate_davis_bouldin_index(self.X, labels_with_empty, centroids_with_empty[:-1])
        self.assertGreaterEqual(db_with_empty, 0.0)

    def test_dunn_index(self):
        """Test Dunn Index calculation"""
        # Good clustering should have a higher Dunn index
        dunn_good = calculate_dunn_index(self.X, self.labels_good, self.centroids_good)
        dunn_poor = calculate_dunn_index(self.X, self.labels_poor, self.centroids_poor)
        
        # Both should be non-negative
        self.assertGreaterEqual(dunn_good, 0.0)
        self.assertGreaterEqual(dunn_poor, 0.0)
        
        # Good clustering should have a higher index than poor clustering
        self.assertGreater(dunn_good, dunn_poor)
        
        # Test with single cluster
        single_label = np.zeros(len(self.X))
        single_centroid = np.array([np.mean(self.X, axis=0)])
        dunn_single = calculate_dunn_index(self.X, single_label, single_centroid)
        self.assertEqual(dunn_single, 0.0)

    def test_c_index(self):
        """Test C-Index calculation"""
        # Good clustering should have a lower C-index
        c_good = calculate_c_index(self.X, self.labels_good)
        c_poor = calculate_c_index(self.X, self.labels_poor)
        
        # Both should be between 0 and 1
        self.assertGreaterEqual(c_good, 0.0)
        self.assertLessEqual(c_good, 1.0)
        self.assertGreaterEqual(c_poor, 0.0)
        self.assertLessEqual(c_poor, 1.0)
        
        # Good clustering should have a lower index than poor clustering
        self.assertLess(c_good, c_poor)
        
        # Test with single cluster
        single_label = np.zeros(len(self.X))
        c_single = calculate_c_index(self.X, single_label)
        self.assertEqual(c_single, 0.0)

    def test_goodman_kruskal_index(self):
        """Test Goodman-Kruskal Index calculation"""
        # Good clustering should have a higher Goodman-Kruskal index when matches true labels
        gk_good = calculate_goodman_kruskal_index(self.labels_good, self.true_labels)
        gk_poor = calculate_goodman_kruskal_index(self.labels_poor, self.true_labels)
        
        # Both should be between -1 and 1
        self.assertGreaterEqual(gk_good, -1.0)
        self.assertLessEqual(gk_good, 1.0)
        self.assertGreaterEqual(gk_poor, -1.0)
        self.assertLessEqual(gk_poor, 1.0)
        
        # Good clustering should have a higher index than poor clustering
        self.assertGreater(gk_good, gk_poor)
        
        # Perfect match should have index 1.0
        gk_perfect = calculate_goodman_kruskal_index(self.true_labels, self.true_labels)
        self.assertAlmostEqual(gk_perfect, 1.0)
        
        # Test with single cluster
        single_label = np.zeros(len(self.X))
        gk_single = calculate_goodman_kruskal_index(single_label, self.true_labels)
        self.assertLessEqual(gk_single, 0.0)  # Should be negative or zero
        
        # Test with empty arrays
        with self.assertRaises(ValueError):
            calculate_goodman_kruskal_index(np.array([]), np.array([]))

    def test_silhouette_score(self):
        """Test Silhouette Score calculation"""
        # Good clustering should have a higher silhouette score
        try:
            # This might fail if scikit-learn is not installed
            sil_good = calculate_silhouette_score(self.X, self.labels_good)
            sil_poor = calculate_silhouette_score(self.X, self.labels_poor)
            
            # Both should be between -1 and 1
            self.assertGreaterEqual(sil_good, -1.0)
            self.assertLessEqual(sil_good, 1.0)
            self.assertGreaterEqual(sil_poor, -1.0)
            self.assertLessEqual(sil_poor, 1.0)
            
            # Good clustering should have a higher score than poor clustering
            self.assertGreater(sil_good, sil_poor)
        except ImportError:
            self.skipTest("scikit-learn not installed, skipping silhouette score test")
        
        # Test with single cluster
        single_label = np.zeros(len(self.X))
        with self.assertRaises(ValueError):
            calculate_silhouette_score(self.X, single_label)

    def test_calculate_all_metrics(self):
        """Test the all-in-one metrics calculation function"""
        # Calculate all metrics for good clustering
        all_metrics_good = calculate_all_metrics(
            self.X, self.labels_good, self.centroids_good, self.true_labels
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['davis_bouldin_index', 'dunn_index', 'c_index', 
                          'silhouette_score', 'goodman_kruskal_index']
        
        for metric in expected_metrics:
            self.assertIn(metric, all_metrics_good)
        
        # Check that metric values are in expected ranges
        self.assertGreaterEqual(all_metrics_good['davis_bouldin_index'], 0.0)
        self.assertGreaterEqual(all_metrics_good['dunn_index'], 0.0)
        self.assertGreaterEqual(all_metrics_good['c_index'], 0.0)
        self.assertLessEqual(all_metrics_good['c_index'], 1.0)
        
        if 'silhouette_score' in all_metrics_good:
            self.assertGreaterEqual(all_metrics_good['silhouette_score'], -1.0)
            self.assertLessEqual(all_metrics_good['silhouette_score'], 1.0)
        
        self.assertGreaterEqual(all_metrics_good['goodman_kruskal_index'], -1.0)
        self.assertLessEqual(all_metrics_good['goodman_kruskal_index'], 1.0)
        
        # Test without true labels
        all_metrics_no_true = calculate_all_metrics(
            self.X, self.labels_good, self.centroids_good
        )
        
        self.assertNotIn('goodman_kruskal_index', all_metrics_no_true)

if __name__ == '__main__':
    unittest.main()

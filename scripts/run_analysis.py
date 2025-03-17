#!/usr/bin/env python3
"""
Main script to run K-Means clustering analysis on MNIST dataset.
This script implements the required steps for the assignment:
1. Load and preprocess MNIST data
2. Run K-Means with different K values
3. Evaluate clustering with validation metrics
4. Optionally classify test data based on clusters
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.data_loader import load_mnist, visualize_sample_digits
from src.kmeans import KMeans
from src.metrics import calculate_all_metrics
from src.visualization import (
    visualize_centroids, 
    plot_digit_clusters, 
    plot_cluster_distribution,
    plot_clustering_metrics,
    visualize_pca_clusters
)
from src.utils import (
    run_kmeans_multiple_k, 
    find_optimal_k, 
    classify_test_data, 
    create_summary_table,
    save_results,
    ensure_dir
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run K-Means clustering on MNIST dataset')
    
    parser.add_argument('--train-path', type=str, default='data/train.csv',
                        help='Path to the training data CSV')
    parser.add_argument('--test-path', type=str, default='data/test.csv',
                        help='Path to the test data CSV')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 7, 9, 10, 12, 15],
                        help='K values to test')
    parser.add_argument('--n-init', type=int, default=3,
                        help='Number of initializations for each K')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Number of samples to use (use 0 for all)')
    parser.add_argument('--classify', action='store_true',
                        help='Run classification based on clusters')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create results directory
    ensure_dir(args.results_dir)
    
    # Configure output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_prefix = f"{args.results_dir}/kmeans_{timestamp}"
    
    print(f"K-Means Clustering Analysis")
    print(f"============================")
    print(f"K values: {args.k_values}")
    print(f"Initializations per K: {args.n_init}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'All'}")
    print(f"Results will be saved to: {results_prefix}_*.csv/png")
    print()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_features, train_labels, test_features, test_labels = load_mnist(
        train_path=args.train_path,
        test_path=args.test_path,
        normalize=True
    )
    
    # Use a subset of the data if specified
    if args.sample_size > 0 and args.sample_size < len(train_features):
        print(f"Using a subset of {args.sample_size} samples for analysis")
        sample_indices = np.random.choice(len(train_features), args.sample_size, replace=False)
        X_subset = train_features[sample_indices]
        y_subset = train_labels[sample_indices]
    else:
        print("Using all training samples")
        X_subset = train_features
        y_subset = train_labels
    
    # Visualize some sample digits and save the figure
    print("Visualizing sample digits...")
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        digit_indices = np.where(y_subset == i)[0]
        if len(digit_indices) > 0:
            sample_idx = np.random.choice(digit_indices)
            plt.imshow(X_subset[sample_idx].reshape(28, 28), cmap='gray')
            plt.title(f"Digit: {i}")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{results_prefix}_sample_digits.png")
    plt.close()
    
    # Run K-Means with multiple K values
    print("\nRunning K-Means with multiple K values...")
    results = run_kmeans_multiple_k(
        X_subset, 
        k_values=args.k_values, 
        n_init=args.n_init, 
        true_labels=y_subset
    )
    
    # Save results
    save_results(results, f"kmeans_results_{timestamp}.pkl", args.results_dir)
    
    # Create summary table
    summary_table = create_summary_table(results, args.k_values)
    summary_table.to_csv(f"{results_prefix}_summary.csv", index=False)
    print("\nSummary of results:")
    print(summary_table)
    
    # Find optimal K for each metric
    print("\nFinding optimal K values...")
    metrics_to_optimize = [
        ('davis_bouldin_index', 'min', 'Davis-Bouldin Index'),
        ('dunn_index', 'max', 'Dunn Index'),
        ('c_index', 'min', 'C-Index'),
        ('silhouette_score', 'max', 'Silhouette Score'),
        ('goodman_kruskal_index', 'max', 'Goodman-Kruskal Index')
    ]
    
    optimal_k_values = {}
    for metric, direction, display_name in metrics_to_optimize:
        try:
            optimal_k = find_optimal_k(results, metric, direction)
            optimal_k_values[metric] = optimal_k
            print(f"{display_name}: Optimal K = {optimal_k}")
        except Exception as e:
            print(f"Could not find optimal K for {display_name}: {e}")
    
    # Visualize optimal centroids
    for metric, k in optimal_k_values.items():
        print(f"\nVisualizing centroids for optimal K={k} based on {metric}...")
        kmeans = KMeans(n_clusters=k, random_state=args.seed)
        kmeans.fit(X_subset)
        
        # Save centroids as images
        fig, axes = plt.subplots(1, k, figsize=(3*k, 3))
        if k == 1:
            axes = [axes]  # Make it iterable if k=1
        for i, centroid in enumerate(kmeans.centroids):
            axes[i].imshow(centroid.reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Centroid {i}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f"{results_prefix}_centroids_k{k}_{metric}.png")
        plt.close()
    
    # Plot metrics for different K values
    print("\nPlotting metrics for different K values...")
    metric_names = ['davis_bouldin_index', 'dunn_index', 'c_index', 
                   'silhouette_score', 'goodman_kruskal_index', 'inertia']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metric_names):
        plt.subplot(2, 3, i+1)
        values = []
        for k in args.k_values:
            if k in results and results[k]:
                metric_values = [result[metric] for result in results[k] if metric in result]
                if metric_values:
                    values.append(np.mean(metric_values))
                else:
                    values.append(np.nan)
        
        plt.plot(args.k_values, values, marker='o', linestyle='-')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Metric Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(args.k_values)
    
    plt.tight_layout()
    plt.savefig(f"{results_prefix}_metrics.png")
    plt.close()
    
    # Optional: Classification based on clusters
    if args.classify:
        print("\nClassifying test data based on clusters...")
        classification_results = {}
        
        for k in args.k_values:
            print(f"Classifying with K={k}...")
            accuracy, _, _ = classify_test_data(
                X_subset, y_subset, test_features, test_labels, k=k
            )
            classification_results[k] = accuracy
            print(f"Test accuracy: {accuracy:.4f}")
        
        # Save classification results
        classification_df = pd.DataFrame({
            'K': args.k_values,
            'Accuracy': [classification_results[k] for k in args.k_values]
        })
        classification_df.to_csv(f"{results_prefix}_classification.csv", index=False)
        
        # Plot classification accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(args.k_values, [classification_results[k] for k in args.k_values], 
                 marker='o', linestyle='-')
        plt.title('Classification Accuracy vs. K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(args.k_values)
        plt.savefig(f"{results_prefix}_classification_accuracy.png")
        plt.close()
        
        # Find optimal K for classification
        optimal_k_classification = max(classification_results, key=classification_results.get)
        print(f"Optimal K for classification: {optimal_k_classification} with accuracy: {classification_results[optimal_k_classification]:.4f}")
    
    print("\nAnalysis complete. Results saved to:", args.results_dir)

if __name__ == "__main__":
    main()

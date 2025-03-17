import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import json

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results(results, filename, directory='results'):
    """
    Save results to a file.
    
    Parameters:
    -----------
    results : object
        Results to save
    filename : str
        Filename
    directory : str, default='results'
        Directory to save to
    """
    ensure_dir(directory)
    
    file_path = os.path.join(directory, filename)
    
    # Determine file type from extension
    _, ext = os.path.splitext(filename)
    
    if ext.lower() == '.pkl':
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    elif ext.lower() == '.json':
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    elif ext.lower() in ['.csv', '.txt']:
        if isinstance(results, pd.DataFrame):
            results.to_csv(file_path, index=False)
        elif isinstance(results, dict):
            pd.DataFrame(results).to_csv(file_path, index=False)
        else:
            with open(file_path, 'w') as f:
                f.write(str(results))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    print(f"Results saved to {file_path}")

def load_results(filename, directory='results'):
    """
    Load results from a file.
    
    Parameters:
    -----------
    filename : str
        Filename
    directory : str, default='results'
        Directory to load from
        
    Returns:
    --------
    results : object
        Loaded results
    """
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type from extension
    _, ext = os.path.splitext(filename)
    
    if ext.lower() == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif ext.lower() == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif ext.lower() == '.csv':
        return pd.read_csv(file_path)
    elif ext.lower() == '.txt':
        with open(file_path, 'r') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def generate_timestamp():
    """
    Generate a timestamp string for naming files.
    
    Returns:
    --------
    timestamp : str
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_kmeans_multiple_k(X, k_values, n_init=3, true_labels=None):
    """
    Run K-Means with multiple K values and initializations.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    k_values : list of int
        List of K values to test
    n_init : int, default=3
        Number of initializations for each K
    true_labels : array-like, shape (n_samples,), optional
        Ground truth labels for supervised evaluation
        
    Returns:
    --------
    results : dict
        Dictionary containing results for each K value
    """
    from src.kmeans import KMeans
    from src.metrics import calculate_all_metrics
    
    results = {}
    
    for k in k_values:
        k_results = []
        
        for init in range(n_init):
            print(f"Running K-Means with k={k}, initialization {init+1}/{n_init}")
            
            # Initialize and fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=42+init)
            kmeans.fit(X)
            
            # Evaluate clustering
            metrics = calculate_all_metrics(X, kmeans.labels, kmeans.centroids, true_labels)
            metrics['inertia'] = kmeans.inertia_
            metrics['init'] = init
            k_results.append(metrics)
            
            # Display key metrics
            print(f"  Inertia: {kmeans.inertia_:.2f}")
            for metric_name, value in metrics.items():
                if metric_name not in ['init', 'inertia']:
                    print(f"  {metric_name}: {value:.4f}")
        
        # Store results for this K value
        results[k] = k_results
    
    return results

def find_optimal_k(results, metric_name, optimize_direction='min'):
    """
    Find the optimal K value based on a metric.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each K value
    metric_name : str
        Name of the metric to use for selection
    optimize_direction : str, default='min'
        Direction of optimization, 'min' or 'max'
        
    Returns:
    --------
    optimal_k : int
        Optimal K value
    """
    k_values = list(results.keys())
    mean_metrics = []
    
    for k in k_values:
        metric_values = [result[metric_name] for result in results[k] if metric_name in result]
        if metric_values:
            mean_metrics.append(np.mean(metric_values))
        else:
            mean_metrics.append(np.nan)
    
    # Filter out nan values
    valid_indices = ~np.isnan(mean_metrics)
    valid_k_values = [k_values[i] for i in range(len(k_values)) if valid_indices[i]]
    valid_metrics = [mean_metrics[i] for i in range(len(mean_metrics)) if valid_indices[i]]
    
    if not valid_metrics:
        raise ValueError(f"No valid metrics found for {metric_name}")
    
    if optimize_direction == 'min':
        optimal_idx = np.argmin(valid_metrics)
    else:  # 'max'
        optimal_idx = np.argmax(valid_metrics)
    
    return valid_k_values[optimal_idx]

def classify_test_data(train_features, train_labels, test_features, test_labels, k):
    """
    Classify test data using K-Means clustering.
    
    Parameters:
    -----------
    train_features : array-like, shape (n_train_samples, n_features)
        Training data features
    train_labels : array-like, shape (n_train_samples,)
        Training data labels
    test_features : array-like, shape (n_test_samples, n_features)
        Test data features
    test_labels : array-like, shape (n_test_samples,)
        Test data labels
    k : int
        Number of clusters
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    predicted_labels : array, shape (n_test_samples,)
        Predicted labels for test data
    """
    from src.kmeans import KMeans
    
    # Run K-Means on training data
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_features)
    
    # Assign each cluster the majority class label
    cluster_labels = np.zeros(k, dtype=int)
    for i in range(k):
        cluster_points = train_labels[kmeans.labels == i]
        if len(cluster_points) > 0:
            cluster_labels[i] = np.bincount(cluster_points.astype(int)).argmax()
    
    # Predict cluster for test data
    test_clusters = kmeans.predict(test_features)
    
    # Assign majority class label for each test point
    predicted_labels = cluster_labels[test_clusters]
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == test_labels)
    
    return accuracy, predicted_labels, cluster_labels

def create_summary_table(results, k_values):
    """
    Create a summary table of all metrics for all K values.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each K value
    k_values : list of int
        List of K values
        
    Returns:
    --------
    summary_df : pandas.DataFrame
        Summary table
    """
    summary = []
    
    for k in k_values:
        row = {'K': k}
        
        # Get all metric names from the first initialization of this K
        if k in results and results[k]:
            metric_names = [name for name in results[k][0].keys() if name != 'init']
            
            # Calculate mean and std for each metric
            for metric in metric_names:
                values = [result[metric] for result in results[k] if metric in result]
                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
        
        summary.append(row)
    
    return pd.DataFrame(summary)

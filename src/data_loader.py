import pandas as pd
import numpy as np
import os

def load_mnist(train_path='data/train.csv', test_path='data/test.csv', normalize=True):
    """
    Load MNIST dataset from CSV files.
    
    Parameters:
    -----------
    train_path : str, default='data/train.csv'
        Path to the training data file
    test_path : str, default='data/test.csv'
        Path to the test data file
    normalize : bool, default=True
        Whether to normalize pixel values to [0, 1]
        
    Returns:
    --------
    train_features : numpy.ndarray
        Training data features
    train_labels : numpy.ndarray
        Training data labels
    test_features : numpy.ndarray
        Test data features
    test_labels : numpy.ndarray
        Test data labels
    """
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found at {test_path}")
    
    # Load data
    print(f"Loading training data from {train_path}...")
    train_data = pd.read_csv(train_path)
    
    print(f"Loading test data from {test_path}...")
    test_data = pd.read_csv(test_path)
    
    # Extract labels and features
    train_labels = train_data.iloc[:, 0].values
    train_features = train_data.iloc[:, 1:].values
    
    test_labels = test_data.iloc[:, 0].values
    test_features = test_data.iloc[:, 1:].values
    
    # Normalize pixel values to [0, 1] if requested
    if normalize:
        train_features = train_features / 255.0
        test_features = test_features / 255.0
    
    print(f"Loaded {train_features.shape[0]} training samples and {test_features.shape[0]} test samples")
    print(f"Each sample has {train_features.shape[1]} features")
    
    return train_features, train_labels, test_features, test_labels

def visualize_digit(data, idx, label=None):
    """
    Visualize a single digit from the dataset.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array containing the digit images
    idx : int
        Index of the digit to visualize
    label : int, optional
        The label of the digit
    """
    import matplotlib.pyplot as plt
    
    # Reshape the digit to 28x28
    digit = data[idx].reshape(28, 28)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(digit, cmap='gray')
    
    if label is not None:
        plt.title(f"Label: {label}")
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_sample_digits(features, labels, n_samples=5):
    """
    Visualize a random sample of digits from the dataset.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Data array containing the digit images
    labels : numpy.ndarray
        Array of labels for the digits
    n_samples : int, default=5
        Number of random samples to visualize
    """
    import matplotlib.pyplot as plt
    import random
    
    # Sample random indices
    indices = random.sample(range(len(features)), n_samples)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 3))
    
    for i, idx in enumerate(indices):
        # Reshape the digit to 28x28
        digit = features[idx].reshape(28, 28)
        
        # Plot on the corresponding subplot
        axes[i].imshow(digit, cmap='gray')
        axes[i].set_title(f"Label: {labels[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

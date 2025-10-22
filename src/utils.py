# src/utils.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

def load_dataset():
    """Load and enhance the Iris dataset with additional synthetic data for better training."""
    # Load original Iris dataset
    iris = load_iris(as_frame=True)
    X_original = iris.data
    y_original = iris.target
    
    # Create synthetic data to augment the dataset
    # Generate additional samples around the original data distribution
    np.random.seed(42)
    
    # Get class statistics for generating realistic synthetic data
    synthetic_samples = []
    synthetic_labels = []
    
    for class_id in range(3):
        class_mask = y_original == class_id
        class_data = X_original[class_mask]
        
        # Calculate mean and std for each class
        class_mean = class_data.mean()
        class_std = class_data.std()
        
        # Generate synthetic samples (50% more data per class)
        n_synthetic = len(class_data) // 2
        synthetic_class_data = np.random.normal(
            class_mean.values, 
            class_std.values * 0.3,  # Reduced variance for realistic data
            (n_synthetic, 4)
        )
        
        # Ensure realistic bounds for iris measurements
        synthetic_class_data = np.clip(synthetic_class_data, 
                                    [4.0, 2.0, 1.0, 0.1],  # min bounds
                                    [8.0, 4.5, 7.0, 2.5])  # max bounds
        
        synthetic_samples.append(synthetic_class_data)
        synthetic_labels.extend([class_id] * n_synthetic)
    
    # Combine original and synthetic data
    X_synthetic = np.vstack(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)
    
    # Combine original and synthetic data
    X_combined = np.vstack([X_original.values, X_synthetic])
    y_combined = np.hstack([y_original.values, y_synthetic])
    
    # Create DataFrame
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df = pd.DataFrame(X_combined, columns=feature_names)
    df['target'] = y_combined
    
    # Remove any duplicate rows if they exist
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate rows from dataset")
    
    print(f"Enhanced dataset: {len(df)} samples (original: {len(X_original)}, synthetic: {len(X_synthetic)})")
    print(f"Class distribution: {df['target'].value_counts().sort_index().to_dict()}")
    
    return df

def get_class_names():
    """Get human-readable class names for Iris species."""
    return ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

def get_feature_descriptions():
    """Get descriptions of the features."""
    return {
        'sepal_length': 'Length of the sepal in centimeters',
        'sepal_width': 'Width of the sepal in centimeters', 
        'petal_length': 'Length of the petal in centimeters',
        'petal_width': 'Width of the petal in centimeters'
    }

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

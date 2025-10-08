import numpy as np

def min_max_scaling(X):
    X = np.array(X, dtype=float)
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    
    X_scaled = (X - X_min) / (X_max - X_min)
    
    return X_scaled, X_max, X_min

def inverse_min_max_scaling(X_scaled, X_max, X_min):
    return X_scaled*(X_max - X_min) + X_min

def z_score_scaling(X):
    X = np.array(X, dtype=float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    X_scaled = (X - X_mean) / X_std
    
    return X_scaled, X_mean, X_std

def inverse_z_score_scaling(X_scaled, X_mean, X_std):
    return X_scaled*X_std+ X_mean

data1 = [1, 2, 3]
data2 = [4, 5, 6]
data3 = [7, 8, 9]
X = [data1, data2, data3]

# Min-Max
X_minmax, mins, maxs = min_max_scaling(X)
print("Min-Max Scaled:", X_minmax)
print("Recovered Min-Max:", inverse_min_max_scaling(X_minmax, mins, maxs))

# Z-Score
X_zscore, means, stds = z_score_scaling(X)
print("\nZ-Score Scaled:", X_zscore)
print("Recovered Z-Score:", inverse_z_score_scaling(X_zscore, means, stds))

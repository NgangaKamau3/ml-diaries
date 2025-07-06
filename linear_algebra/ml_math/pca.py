"""
Principal Component Analysis
===========================

PCA implementation using eigendecomposition and SVD.
"""

from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector
from eigenvalues.eigendecomposition import EigenDecomposition
from eigenvalues.svd import SVD


class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.is_fitted = False
    
    def fit(self, X: Matrix) -> 'PCA':
        """Fit PCA to data matrix X (samples × features)."""
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = Vector([sum(X[i, j] for i in range(n_samples)) / n_samples 
                            for j in range(n_features)])
        
        # Create centered data matrix
        X_centered = Matrix.zeros(n_samples, n_features)
        for i in range(n_samples):
            for j in range(n_features):
                X_centered[i, j] = X[i, j] - self.mean_[j]
        
        # Compute covariance matrix
        cov_matrix = self._compute_covariance(X_centered)
        
        # Eigendecomposition of covariance matrix
        eigen_decomp = EigenDecomposition(cov_matrix)
        eigenvalues, eigenvectors = eigen_decomp.compute()
        
        # Sort by eigenvalue magnitude (descending)
        eigen_pairs = [(abs(eigenvalues[i]), eigenvectors.get_col(i)) 
                      for i in range(len(eigenvalues.components))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract components
        n_comp = self.n_components or n_features
        n_comp = min(n_comp, n_features)
        
        self.explained_variance_ = Vector([pair[0] for pair in eigen_pairs[:n_comp]])
        component_vectors = [pair[1] for pair in eigen_pairs[:n_comp]]
        self.components_ = Matrix.from_vectors(component_vectors, by_columns=False)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        """Transform data to principal component space."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        
        n_samples, n_features = X.shape
        
        # Center the data
        X_centered = Matrix.zeros(n_samples, n_features)
        for i in range(n_samples):
            for j in range(n_features):
                X_centered[i, j] = X[i, j] - self.mean_[j]
        
        # Project onto principal components
        return X_centered * self.components_.transpose()
    
    def fit_transform(self, X: Matrix) -> Matrix:
        """Fit PCA and transform data."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: Matrix) -> Matrix:
        """Transform back to original space."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse_transform")
        
        # X_original = X_transformed * components + mean
        X_reconstructed = X_transformed * self.components_
        
        # Add back the mean
        n_samples = X_reconstructed.rows
        for i in range(n_samples):
            for j in range(len(self.mean_.components)):
                X_reconstructed[i, j] += self.mean_[j]
        
        return X_reconstructed
    
    def explained_variance_ratio(self) -> Vector:
        """Ratio of variance explained by each component."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        
        total_variance = sum(self.explained_variance_.components)
        if total_variance == 0:
            return Vector([0.0] * len(self.explained_variance_.components))
        
        ratios = [var / total_variance for var in self.explained_variance_.components]
        return Vector(ratios)
    
    def _compute_covariance(self, X_centered: Matrix) -> Matrix:
        """Compute covariance matrix from centered data."""
        n_samples, n_features = X_centered.shape
        
        # Cov = (1/(n-1)) * X^T * X
        cov = Matrix.zeros(n_features, n_features)
        
        for i in range(n_features):
            for j in range(n_features):
                cov_ij = sum(X_centered[k, i] * X_centered[k, j] 
                           for k in range(n_samples))
                cov[i, j] = cov_ij / (n_samples - 1)
        
        return cov


def pca_svd(X: Matrix, n_components: Optional[int] = None) -> Tuple[Matrix, Vector, Matrix]:
    """PCA using SVD (alternative implementation)."""
    n_samples, n_features = X.shape
    
    # Center data
    mean = Vector([sum(X[i, j] for i in range(n_samples)) / n_samples 
                  for j in range(n_features)])
    
    X_centered = Matrix.zeros(n_samples, n_features)
    for i in range(n_samples):
        for j in range(n_features):
            X_centered[i, j] = X[i, j] - mean[j]
    
    # SVD of centered data
    svd = SVD(X_centered)
    U, S, Vt = svd.compute()
    
    # Principal components are rows of V^T
    n_comp = n_components or min(n_samples, n_features)
    
    # Explained variance = (singular_values^2) / (n_samples - 1)
    explained_var = Vector([s**2 / (n_samples - 1) for s in S.components[:n_comp]])
    
    return Vt, explained_var, mean
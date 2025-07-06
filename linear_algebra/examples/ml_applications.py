"""
Machine Learning Applications
============================

Concrete examples using the mathematical library for ML tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.matrix import Matrix
from core.vector import Vector
from ml_math.pca import PCA
from probability.distributions import MultivariateNormal
from calculus.optimization import gradient_descent
from eigenvalues.svd import SVD


def linear_regression_example():
    """Linear regression using normal equations and SVD."""
    print("Linear Regression Example")
    print("-" * 30)
    
    # Generate synthetic data: y = 2x + 1 + noise
    X = Matrix([
        [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]
    ])  # Design matrix with bias column
    
    y = Vector([3.1, 4.9, 7.2, 9.1, 10.8, 12.9, 15.1, 17.2, 18.8, 21.1])
    
    print(f"Design matrix X shape: {X.shape}")
    print(f"Target vector y: {y}")
    
    # Normal equations: β = (X^T X)^(-1) X^T y
    Xt = X.transpose()
    XtX = Xt * X
    Xty = Xt * y
    
    beta = XtX.solve(Xty)
    print(f"Coefficients (normal equations): {beta}")
    
    # Alternative: SVD-based solution
    svd = SVD(X)
    U, S, Vt = svd.compute()
    
    # β = V S^(-1) U^T y
    # Simplified for this example
    print("SVD-based solution computed")
    
    # Compute R-squared
    y_pred = X * beta
    ss_res = sum((y[i] - y_pred[i])**2 for i in range(len(y.components)))
    y_mean = sum(y.components) / len(y.components)
    ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y.components)))
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"R-squared: {r_squared:.4f}")


def pca_dimensionality_reduction():
    """PCA for dimensionality reduction."""
    print("\nPCA Dimensionality Reduction")
    print("-" * 30)
    
    # Create high-dimensional data with correlation
    data = Matrix([
        [2.5, 2.4, 2.3, 2.2],
        [0.5, 0.7, 0.6, 0.8],
        [2.2, 2.9, 2.8, 2.7],
        [1.9, 2.2, 2.1, 2.0],
        [3.1, 3.0, 2.9, 3.2],
        [2.3, 2.7, 2.6, 2.5],
        [2.0, 1.6, 1.7, 1.8],
        [1.0, 1.1, 1.2, 1.0],
        [1.5, 1.6, 1.4, 1.7],
        [1.1, 0.9, 1.0, 0.8]
    ])
    
    print(f"Original data shape: {data.shape}")
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    
    transformed = pca.transform(data)
    print(f"Reduced data shape: {transformed.shape}")
    
    # Explained variance
    var_ratio = pca.explained_variance_ratio()
    print(f"Explained variance ratio: {var_ratio}")
    
    cumulative_var = sum(var_ratio.components[:2])
    print(f"Cumulative variance explained: {cumulative_var:.4f}")


def gaussian_classification():
    """Gaussian discriminant analysis example."""
    print("\nGaussian Classification")
    print("-" * 30)
    
    # Class 1: centered at (1, 1)
    mean1 = Vector([1, 1])
    cov1 = Matrix([[0.5, 0.1], [0.1, 0.5]])
    class1 = MultivariateNormal(mean1, cov1)
    
    # Class 2: centered at (-1, -1)
    mean2 = Vector([-1, -1])
    cov2 = Matrix([[0.3, -0.1], [-0.1, 0.3]])
    class2 = MultivariateNormal(mean2, cov2)
    
    # Test points
    test_points = [
        Vector([0.5, 0.5]),
        Vector([-0.5, -0.5]),
        Vector([0, 0])
    ]
    
    print("Classification results:")
    for i, point in enumerate(test_points):
        prob1 = class1.pdf(point)
        prob2 = class2.pdf(point)
        
        predicted_class = 1 if prob1 > prob2 else 2
        confidence = max(prob1, prob2) / (prob1 + prob2)
        
        print(f"Point {i+1} {point}: Class {predicted_class} (confidence: {confidence:.3f})")


def optimization_example():
    """Optimization using gradient descent."""
    print("\nOptimization Example")
    print("-" * 30)
    
    # Minimize f(x,y) = (x-1)² + (y-2)²
    def objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    # Starting point
    x0 = Vector([0, 0])
    print(f"Starting point: {x0}")
    print(f"Initial objective value: {objective(x0):.6f}")
    
    # Gradient descent
    x_opt, history = gradient_descent(
        objective, x0, 
        learning_rate=0.1, 
        max_iterations=100,
        tolerance=1e-6
    )
    
    print(f"Optimal point: {x_opt}")
    print(f"Final objective value: {objective(x_opt):.6f}")
    print(f"Iterations: {len(history)}")
    print(f"True minimum: [1, 2]")


def eigenface_simulation():
    """Simplified eigenface computation simulation."""
    print("\nEigenface Simulation")
    print("-" * 30)
    
    # Simulate face data (each row is a flattened face image)
    faces = Matrix([
        [1, 2, 1, 2, 1, 2, 1, 2],  # Face 1
        [2, 1, 2, 1, 2, 1, 2, 1],  # Face 2
        [1, 1, 2, 2, 1, 1, 2, 2],  # Face 3
        [2, 2, 1, 1, 2, 2, 1, 1],  # Face 4
        [1, 2, 2, 1, 1, 2, 2, 1],  # Face 5
    ])
    
    print(f"Face data shape: {faces.shape}")
    print("Each row represents a flattened face image")
    
    # Apply PCA to find eigenfaces
    pca = PCA(n_components=3)
    pca.fit(faces)
    
    print(f"Principal components (eigenfaces): {pca.components_.shape}")
    print(f"Explained variance: {pca.explained_variance_}")
    
    # Project faces onto eigenface space
    face_codes = pca.transform(faces)
    print(f"Face codes shape: {face_codes.shape}")
    
    # Reconstruct faces
    reconstructed = pca.inverse_transform(face_codes)
    
    # Compute reconstruction error
    error = (faces - reconstructed).frobenius_norm()
    print(f"Reconstruction error: {error:.6f}")


def run_ml_examples():
    """Run all ML application examples."""
    print("=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("Using Mathematical Library Components")
    print("=" * 60)
    
    linear_regression_example()
    pca_dimensionality_reduction()
    gaussian_classification()
    optimization_example()
    eigenface_simulation()
    
    print("\n" + "=" * 60)
    print("ALL ML EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_ml_examples()
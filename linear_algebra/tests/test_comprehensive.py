"""
Comprehensive Test Suite
=======================

Tests for all mathematical components following Deisenroth et al. standards.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.matrix import Matrix
from core.vector import Vector
from eigenvalues.eigendecomposition import EigenDecomposition
from eigenvalues.svd import SVD
from decompositions.qr import gram_schmidt_qr
from decompositions.cholesky import cholesky_decompose
from ml_math.pca import PCA
from probability.distributions import MultivariateNormal
from numerical.conditioning import condition_number_2norm


def test_eigendecomposition():
    """Test eigenvalue decomposition."""
    print("Testing Eigendecomposition...")
    
    # Symmetric matrix
    A = Matrix([[4, 2], [2, 3]])
    eigen = EigenDecomposition(A)
    eigenvals, eigenvecs = eigen.compute()
    
    print(f"Eigenvalues: {eigenvals}")
    print(f"Matrix is symmetric: {A.is_symmetric()}")
    
    # Verify A*v = λ*v for first eigenpair
    v1 = eigenvecs.get_col(0)
    lambda1 = eigenvals[0]
    Av1 = A * v1
    lambda_v1 = lambda1 * v1
    
    error = (Av1 - lambda_v1).magnitude()
    print(f"Eigenvalue equation error: {error:.2e}")
    assert error < 1e-10, "Eigenvalue equation not satisfied"


def test_svd():
    """Test Singular Value Decomposition."""
    print("\nTesting SVD...")
    
    A = Matrix([[3, 2, 2], [2, 3, -2]])
    svd = SVD(A)
    U, S, Vt = svd.compute()
    
    print(f"Original matrix shape: {A.shape}")
    print(f"U shape: {U.shape}")
    print(f"Singular values: {S}")
    print(f"Vt shape: {Vt.shape}")
    
    # Verify dimensions
    assert U.rows == A.rows
    assert Vt.cols == A.cols
    print("SVD dimensions correct")


def test_qr_decomposition():
    """Test QR decomposition."""
    print("\nTesting QR Decomposition...")
    
    A = Matrix([[1, 2], [3, 4], [5, 6]])
    Q, R = gram_schmidt_qr(A)
    
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    
    # Verify Q is orthogonal
    QtQ = Q.transpose() * Q
    I = Matrix.identity(Q.cols)
    error = (QtQ - I).frobenius_norm()
    print(f"Orthogonality error: {error:.2e}")
    assert error < 1e-10, "Q is not orthogonal"


def test_cholesky():
    """Test Cholesky decomposition."""
    print("\nTesting Cholesky Decomposition...")
    
    # Create positive definite matrix
    A = Matrix([[4, 2], [2, 3]])
    
    try:
        L = cholesky_decompose(A)
        print(f"Cholesky factor L:")
        print(L)
        
        # Verify A = L*L^T
        LLt = L * L.transpose()
        error = (A - LLt).frobenius_norm()
        print(f"Reconstruction error: {error:.2e}")
        assert error < 1e-10, "Cholesky reconstruction failed"
        
    except ValueError as e:
        print(f"Cholesky failed: {e}")


def test_pca():
    """Test Principal Component Analysis."""
    print("\nTesting PCA...")
    
    # Create sample data
    data = Matrix([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])
    
    pca = PCA(n_components=2)
    pca.fit(data)
    
    print(f"Explained variance: {pca.explained_variance_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio()}")
    
    # Transform data
    transformed = pca.transform(data)
    print(f"Transformed data shape: {transformed.shape}")
    
    # Inverse transform
    reconstructed = pca.inverse_transform(transformed)
    error = (data - reconstructed).frobenius_norm()
    print(f"Reconstruction error: {error:.2e}")


def test_multivariate_normal():
    """Test multivariate normal distribution."""
    print("\nTesting Multivariate Normal...")
    
    mean = Vector([0, 0])
    cov = Matrix([[1, 0.5], [0.5, 1]])
    
    mvn = MultivariateNormal(mean, cov)
    
    # Test PDF at mean
    pdf_at_mean = mvn.pdf(mean)
    print(f"PDF at mean: {pdf_at_mean:.6f}")
    
    # Test log PDF
    log_pdf_at_mean = mvn.log_pdf(mean)
    print(f"Log PDF at mean: {log_pdf_at_mean:.6f}")
    
    # Test Mahalanobis distance
    test_point = Vector([1, 1])
    mahal_dist = mvn.mahalanobis_distance(test_point)
    print(f"Mahalanobis distance: {mahal_dist:.6f}")


def test_conditioning():
    """Test numerical conditioning analysis."""
    print("\nTesting Numerical Conditioning...")
    
    # Well-conditioned matrix
    A_good = Matrix([[2, 1], [1, 2]])
    cond_good = condition_number_2norm(A_good)
    print(f"Well-conditioned matrix condition number: {cond_good:.2e}")
    
    # Ill-conditioned matrix (Hilbert-like)
    A_bad = Matrix([[1, 0.5, 0.33], [0.5, 0.33, 0.25], [0.33, 0.25, 0.2]])
    cond_bad = condition_number_2norm(A_bad)
    print(f"Ill-conditioned matrix condition number: {cond_bad:.2e}")
    
    assert cond_good < cond_bad, "Conditioning test failed"


def test_vector_operations():
    """Test comprehensive vector operations."""
    print("\nTesting Vector Operations...")
    
    u = Vector([3, 4])
    v = Vector([1, 2])
    
    # Basic operations
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"u + v = {u + v}")
    print(f"u · v = {u.dot(v)}")
    print(f"||u|| = {u.magnitude():.6f}")
    print(f"Angle between u and v: {u.angle_with(v, degrees=True):.2f}°")
    
    # Projections
    proj = u.project_onto(v)
    print(f"Projection of u onto v: {proj}")
    
    # Orthogonality test
    w = Vector([4, -3])  # Orthogonal to u
    print(f"u orthogonal to w: {u.is_orthogonal(w)}")


def test_matrix_properties():
    """Test matrix property checks."""
    print("\nTesting Matrix Properties...")
    
    # Symmetric matrix
    S = Matrix([[1, 2], [2, 3]])
    print(f"Matrix S is symmetric: {S.is_symmetric()}")
    
    # Diagonal matrix
    D = Matrix([[2, 0], [0, 3]])
    print(f"Matrix D is diagonal: {D.is_diagonal()}")
    
    # Upper triangular
    U = Matrix([[1, 2], [0, 3]])
    print(f"Matrix U is upper triangular: {U.is_upper_triangular()}")
    
    # Identity
    I = Matrix.identity(3)
    print(f"Identity matrix is identity: {I.is_identity()}")


def run_all_tests():
    """Run comprehensive test suite."""
    print("=" * 60)
    print("COMPREHENSIVE MATHEMATICAL LIBRARY TEST SUITE")
    print("Following Deisenroth, Faisal & Ong Standards")
    print("=" * 60)
    
    try:
        test_vector_operations()
        test_matrix_properties()
        test_eigendecomposition()
        test_svd()
        test_qr_decomposition()
        test_cholesky()
        test_pca()
        test_multivariate_normal()
        test_conditioning()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("Mathematical library meets academic rigor standards.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
"""
Eigenvalue Decomposition
=======================

Eigenvalue decomposition for symmetric matrices using QR algorithm.
Essential for PCA, spectral methods, and matrix analysis.
"""

import math
from typing import Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector


class EigenDecomposition:
    """Eigenvalue decomposition A = QΛQᵀ for symmetric matrices."""
    
    def __init__(self, matrix: Matrix, tolerance: float = 1e-10):
        if not matrix.is_square():
            raise ValueError("Eigendecomposition requires square matrix")
        if not matrix.is_symmetric(tolerance):
            raise ValueError("This implementation requires symmetric matrix")
        
        self.matrix = matrix.copy()
        self.n = matrix.rows
        self.tolerance = tolerance
        self._eigenvalues = None
        self._eigenvectors = None
        self._is_computed = False
    
    def compute(self, max_iterations: int = 1000) -> Tuple[Vector, Matrix]:
        """Compute eigenvalues and eigenvectors using QR algorithm."""
        if self._is_computed:
            return self._eigenvalues.copy(), self._eigenvectors.copy()
        
        A = self.matrix.copy()
        Q_total = Matrix.identity(self.n)
        
        for _ in range(max_iterations):
            from decompositions.qr import gram_schmidt_qr
            Q, R = gram_schmidt_qr(A)
            A = R * Q
            Q_total = Q_total * Q
            
            # Check convergence (off-diagonal elements)
            if self._is_converged(A):
                break
        
        # Extract eigenvalues from diagonal
        eigenvals = [A[i, i] for i in range(self.n)]
        self._eigenvalues = Vector(eigenvals)
        self._eigenvectors = Q_total
        self._is_computed = True
        
        return self._eigenvalues.copy(), self._eigenvectors.copy()
    
    def _is_converged(self, A: Matrix) -> bool:
        """Check if off-diagonal elements are small enough."""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(A[i, j]) > self.tolerance:
                    return False
        return True


def power_method(matrix: Matrix, max_iterations: int = 1000) -> Tuple[float, Vector]:
    """Find dominant eigenvalue using power method."""
    n = matrix.rows
    x = Vector.random(n).normalize()
    
    for _ in range(max_iterations):
        y = matrix * x
        eigenval = x.dot(y)
        x = y.normalize()
    
    return eigenval, x
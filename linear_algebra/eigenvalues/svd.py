"""
Singular Value Decomposition
===========================

SVD implementation: A = UΣVᵀ
Critical for dimensionality reduction, least squares, and matrix analysis.
"""

import math
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector


class SVD:
    """Singular Value Decomposition A = UΣVᵀ."""
    
    def __init__(self, matrix: Matrix):
        self.matrix = matrix.copy()
        self.m, self.n = matrix.shape
        self._U = None
        self._S = None
        self._Vt = None
        self._is_computed = False
    
    def compute(self) -> Tuple[Matrix, Vector, Matrix]:
        """Compute SVD using eigendecomposition of AᵀA and AAᵀ."""
        if self._is_computed:
            return self._U.copy(), self._S.copy(), self._Vt.copy()
        
        A = self.matrix
        At = A.transpose()
        
        # Compute AᵀA for V
        AtA = At * A
        from .eigendecomposition import EigenDecomposition
        eigen_AtA = EigenDecomposition(AtA)
        eigenvals_AtA, V = eigen_AtA.compute()
        
        # Sort eigenvalues/vectors in descending order
        pairs = [(eigenvals_AtA[i], V.get_col(i)) for i in range(self.n)]
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract singular values and V
        singular_vals = [math.sqrt(max(0, val)) for val, _ in pairs]
        V_sorted = Matrix.from_vectors([vec for _, vec in pairs], by_columns=True)
        
        # Compute U = AVΣ⁻¹
        U_cols = []
        for i in range(min(self.m, self.n)):
            if singular_vals[i] > 1e-15:
                u_i = (A * V_sorted.get_col(i)) / singular_vals[i]
                U_cols.append(u_i)
        
        # Pad U if needed
        while len(U_cols) < self.m:
            # Add orthogonal vectors
            u_new = Vector.random(self.m)
            for u_existing in U_cols:
                u_new = u_new - u_new.project_onto(u_existing)
            if not u_new.is_zero():
                U_cols.append(u_new.normalize())
        
        self._U = Matrix.from_vectors(U_cols[:self.m], by_columns=True)
        self._S = Vector(singular_vals)
        self._Vt = V_sorted.transpose()
        self._is_computed = True
        
        return self._U.copy(), self._S.copy(), self._Vt.copy()
    
    def rank(self, tolerance: float = 1e-10) -> int:
        """Compute numerical rank using singular values."""
        if not self._is_computed:
            self.compute()
        return sum(1 for s in self._S.components if s > tolerance)


def matrix_rank_svd(matrix: Matrix) -> int:
    """Compute matrix rank using SVD."""
    svd = SVD(matrix)
    return svd.rank()
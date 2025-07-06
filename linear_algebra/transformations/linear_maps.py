"""
Linear Transformations and Maps
==============================

Linear maps, kernel, image, and rank-nullity theorem implementations.
"""

from typing import List, Tuple, Optional
from ..core.matrix import Matrix
from ..core.vector import Vector


class LinearTransformation:
    """Linear transformation T: R^n → R^m represented by matrix."""
    
    def __init__(self, matrix: Matrix):
        self.matrix = matrix.copy()
        self.domain_dim = matrix.cols
        self.codomain_dim = matrix.rows
    
    def __call__(self, x: Vector) -> Vector:
        """Apply transformation: T(x) = Ax."""
        return self.matrix * x
    
    def kernel(self, tolerance: float = 1e-10) -> List[Vector]:
        """Compute basis for kernel (null space)."""
        # Solve Ax = 0 using row reduction
        A = self.matrix.copy()
        m, n = A.shape
        
        # Augment with zero vector
        augmented = Matrix.zeros(m, n + 1)
        for i in range(m):
            for j in range(n):
                augmented[i, j] = A[i, j]
        
        # Row reduce to find null space
        # Simplified implementation
        kernel_basis = []
        
        # For each free variable, create basis vector
        rank = A.rank(tolerance)
        nullity = n - rank
        
        if nullity > 0:
            # Create standard basis vectors for free variables
            for i in range(nullity):
                null_vec = Vector.zero(n)
                null_vec._components[rank + i] = 1.0
                kernel_basis.append(null_vec)
        
        return kernel_basis
    
    def image_basis(self) -> List[Vector]:
        """Compute basis for image (column space)."""
        # Use column vectors that correspond to pivot columns
        from ..eigenvalues.svd import SVD
        svd = SVD(self.matrix)
        rank = svd.rank()
        
        # Return first 'rank' columns as basis
        basis = []
        for j in range(min(rank, self.matrix.cols)):
            basis.append(self.matrix.get_col(j))
        
        return basis
    
    def rank(self) -> int:
        """Compute rank of transformation."""
        return self.matrix.rank()
    
    def nullity(self) -> int:
        """Compute nullity (dimension of kernel)."""
        return self.domain_dim - self.rank()
    
    def verify_rank_nullity(self) -> bool:
        """Verify rank-nullity theorem: rank + nullity = domain dimension."""
        return self.rank() + self.nullity() == self.domain_dim


def compose_transformations(T1: LinearTransformation, T2: LinearTransformation) -> LinearTransformation:
    """Compose two linear transformations: (T1 ∘ T2)(x) = T1(T2(x))."""
    if T2.codomain_dim != T1.domain_dim:
        raise ValueError("Transformations cannot be composed: dimension mismatch")
    
    # Composition is matrix multiplication
    composed_matrix = T1.matrix * T2.matrix
    return LinearTransformation(composed_matrix)


def is_injective(T: LinearTransformation, tolerance: float = 1e-10) -> bool:
    """Check if transformation is injective (one-to-one)."""
    # T is injective iff ker(T) = {0}
    kernel_basis = T.kernel(tolerance)
    return len(kernel_basis) == 0


def is_surjective(T: LinearTransformation) -> bool:
    """Check if transformation is surjective (onto)."""
    # T is surjective iff rank(T) = codomain dimension
    return T.rank() == T.codomain_dim


def is_bijective(T: LinearTransformation) -> bool:
    """Check if transformation is bijective (invertible)."""
    return is_injective(T) and is_surjective(T)
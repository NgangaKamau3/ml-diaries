"""
Inner Product Spaces
===================

Inner products, orthogonality, and projections.
"""

from typing import List, Callable
from ..core.vector import Vector
from ..core.matrix import Matrix


def standard_inner_product(u: Vector, v: Vector) -> float:
    """Standard inner product ⟨u,v⟩ = u·v."""
    return u.dot(v)


def weighted_inner_product(u: Vector, v: Vector, weights: Vector) -> float:
    """Weighted inner product ⟨u,v⟩_W = Σ wᵢuᵢvᵢ."""
    if not (u.dimension == v.dimension == weights.dimension):
        raise ValueError("All vectors must have same dimension")
    
    return sum(weights[i] * u[i] * v[i] for i in range(u.dimension))


def matrix_inner_product(u: Vector, v: Vector, A: Matrix) -> float:
    """Matrix-induced inner product ⟨u,v⟩_A = u^T A v."""
    if not A.is_square() or A.rows != u.dimension:
        raise ValueError("Matrix dimensions incompatible")
    
    return u.dot(A * v)


class InnerProductSpace:
    """Inner product space with custom inner product."""
    
    def __init__(self, inner_product: Callable[[Vector, Vector], float]):
        self.inner_product = inner_product
    
    def norm(self, v: Vector) -> float:
        """Induced norm ||v|| = √⟨v,v⟩."""
        return (self.inner_product(v, v)) ** 0.5
    
    def distance(self, u: Vector, v: Vector) -> float:
        """Induced distance d(u,v) = ||u-v||."""
        return self.norm(u - v)
    
    def angle(self, u: Vector, v: Vector) -> float:
        """Angle between vectors: cos θ = ⟨u,v⟩/(||u|| ||v||)."""
        import math
        
        norm_u = self.norm(u)
        norm_v = self.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            raise ValueError("Cannot compute angle with zero vector")
        
        cos_theta = self.inner_product(u, v) / (norm_u * norm_v)
        cos_theta = max(-1, min(1, cos_theta))  # Clamp for numerical stability
        
        return math.acos(cos_theta)
    
    def is_orthogonal(self, u: Vector, v: Vector, tolerance: float = 1e-10) -> bool:
        """Check if vectors are orthogonal: ⟨u,v⟩ = 0."""
        return abs(self.inner_product(u, v)) < tolerance
    
    def project(self, u: Vector, v: Vector) -> Vector:
        """Project u onto v: proj_v(u) = (⟨u,v⟩/⟨v,v⟩)v."""
        v_norm_sq = self.inner_product(v, v)
        if v_norm_sq == 0:
            raise ValueError("Cannot project onto zero vector")
        
        coeff = self.inner_product(u, v) / v_norm_sq
        return coeff * v
    
    def gram_schmidt(self, vectors: List[Vector]) -> List[Vector]:
        """Gram-Schmidt orthogonalization in this inner product space."""
        if not vectors:
            return []
        
        orthogonal = []
        
        for v in vectors:
            # Start with current vector
            u = v.copy()
            
            # Subtract projections onto previous orthogonal vectors
            for orth_vec in orthogonal:
                proj = self.project(v, orth_vec)
                u = u - proj
            
            # Check if result is zero (linearly dependent)
            if self.norm(u) < 1e-15:
                continue  # Skip linearly dependent vectors
            
            orthogonal.append(u)
        
        return orthogonal
    
    def orthonormalize(self, vectors: List[Vector]) -> List[Vector]:
        """Create orthonormal basis."""
        orthogonal = self.gram_schmidt(vectors)
        return [v / self.norm(v) for v in orthogonal if self.norm(v) > 1e-15]


def create_orthonormal_basis(vectors: List[Vector]) -> List[Vector]:
    """Create orthonormal basis using standard inner product."""
    space = InnerProductSpace(standard_inner_product)
    return space.orthonormalize(vectors)


def orthogonal_complement(subspace_basis: List[Vector], ambient_dim: int) -> List[Vector]:
    """Find orthogonal complement of subspace."""
    if not subspace_basis:
        # Complement of {0} is entire space
        return [Vector.standard_basis(ambient_dim, i) for i in range(ambient_dim)]
    
    # Create matrix with subspace basis as rows
    n = len(subspace_basis)
    A = Matrix.from_vectors(subspace_basis, by_columns=False)
    
    # Find null space of A (vectors orthogonal to all basis vectors)
    from ..transformations.linear_maps import LinearTransformation
    T = LinearTransformation(A)
    return T.kernel()


def is_orthogonal_set(vectors: List[Vector], tolerance: float = 1e-10) -> bool:
    """Check if set of vectors is orthogonal."""
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(vectors[i].dot(vectors[j])) > tolerance:
                return False
    return True


def is_orthonormal_set(vectors: List[Vector], tolerance: float = 1e-10) -> bool:
    """Check if set of vectors is orthonormal."""
    if not is_orthogonal_set(vectors, tolerance):
        return False
    
    # Check if all vectors are unit vectors
    for v in vectors:
        if abs(v.magnitude() - 1.0) > tolerance:
            return False
    
    return True
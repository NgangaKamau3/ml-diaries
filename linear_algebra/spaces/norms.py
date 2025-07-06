"""
Vector and Matrix Norms
=======================

Comprehensive norm implementations for vectors and matrices.
Essential for numerical analysis and optimization.
"""

import math
from typing import Union
from ..core.vector import Vector
from ..core.matrix import Matrix


def vector_norm(v: Vector, p: Union[int, float, str] = 2) -> float:
    """Compute p-norm of vector."""
    return v.norm(p)


def matrix_frobenius_norm(A: Matrix) -> float:
    """Frobenius norm: ||A||_F = √(Σᵢⱼ |aᵢⱼ|²)."""
    return A.frobenius_norm()


def matrix_spectral_norm(A: Matrix) -> float:
    """Spectral norm (largest singular value)."""
    return A.spectral_norm_estimate()


def matrix_nuclear_norm(A: Matrix) -> float:
    """Nuclear norm (sum of singular values)."""
    from ..eigenvalues.svd import SVD
    svd = SVD(A)
    _, S, _ = svd.compute()
    return sum(S.components)


def matrix_max_norm(A: Matrix) -> float:
    """Maximum absolute element."""
    return A.max_norm()


def condition_number(A: Matrix, norm_type: str = 'frobenius') -> float:
    """Condition number κ(A) = ||A|| × ||A⁻¹||."""
    if norm_type == 'frobenius':
        norm_A = matrix_frobenius_norm(A)
        try:
            A_inv = A.inverse()
            norm_A_inv = matrix_frobenius_norm(A_inv)
            return norm_A * norm_A_inv
        except ValueError:
            return float('inf')
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented")


def vector_angle(u: Vector, v: Vector) -> float:
    """Angle between vectors in radians."""
    return u.angle_with(v)


def orthogonality_measure(vectors: list) -> float:
    """Measure how close vectors are to being orthogonal."""
    n = len(vectors)
    if n < 2:
        return 0.0
    
    max_dot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dot_prod = abs(vectors[i].normalize().dot(vectors[j].normalize()))
            max_dot = max(max_dot, dot_prod)
    
    return max_dot
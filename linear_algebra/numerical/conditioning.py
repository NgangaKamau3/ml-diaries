"""
Numerical Conditioning and Stability Analysis
=============================================

Tools for analyzing numerical stability and conditioning of matrices.
"""

import math
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector


def condition_number_2norm(A: Matrix) -> float:
    """Condition number in 2-norm using SVD."""
    from eigenvalues.svd import SVD
    svd = SVD(A)
    _, S, _ = svd.compute()
    
    singular_values = [s for s in S.components if s > 1e-15]
    if not singular_values:
        return float('inf')
    
    return max(singular_values) / min(singular_values)


def effective_rank(A: Matrix, tolerance: float = 1e-12) -> int:
    """Effective rank based on singular value threshold."""
    from eigenvalues.svd import SVD
    svd = SVD(A)
    _, S, _ = svd.compute()
    
    max_sv = max(S.components) if S.components else 0
    threshold = tolerance * max_sv
    
    return sum(1 for s in S.components if s > threshold)


def matrix_perturbation_bound(A: Matrix, delta_A: Matrix) -> float:
    """Bound on eigenvalue perturbation."""
    # ||λ_perturbed - λ_original|| ≤ ||δA||
    return delta_A.frobenius_norm()


def backward_error_analysis(A: Matrix, x: Vector, b: Vector) -> dict:
    """Analyze backward error for linear system Ax = b."""
    residual = A * x - b
    residual_norm = residual.magnitude()
    
    # Backward error: smallest δA, δb such that (A + δA)x = b + δb
    backward_error = residual_norm / (A.frobenius_norm() * x.magnitude() + b.magnitude())
    
    # Forward error bound
    cond_A = condition_number_2norm(A)
    forward_error_bound = cond_A * backward_error
    
    return {
        'residual_norm': residual_norm,
        'backward_error': backward_error,
        'condition_number': cond_A,
        'forward_error_bound': forward_error_bound
    }


def iterative_refinement(A: Matrix, b: Vector, x0: Vector, max_iterations: int = 5) -> Vector:
    """Iterative refinement for improved accuracy."""
    from linalg.lu import LUDecomposition
    
    lu_decomp = LUDecomposition(A)
    x = x0.copy()
    
    for _ in range(max_iterations):
        # Compute residual in higher precision if possible
        r = b - A * x
        
        # Solve for correction
        delta_x = lu_decomp.solve(r)
        
        # Apply correction
        x = x + delta_x
        
        # Check convergence
        if delta_x.magnitude() < 1e-15:
            break
    
    return x


def stability_analysis(A: Matrix) -> dict:
    """Comprehensive stability analysis."""
    n = A.rows
    
    # Condition number
    cond_num = condition_number_2norm(A)
    
    # Effective rank
    eff_rank = effective_rank(A)
    
    # Numerical rank deficiency
    rank_deficiency = n - eff_rank
    
    # Frobenius norm
    frob_norm = A.frobenius_norm()
    
    # Spectral radius estimate
    spectral_radius = A.spectral_norm_estimate()
    
    # Stability classification
    if cond_num < 1e2:
        stability = "well-conditioned"
    elif cond_num < 1e6:
        stability = "moderately ill-conditioned"
    elif cond_num < 1e12:
        stability = "ill-conditioned"
    else:
        stability = "severely ill-conditioned"
    
    return {
        'condition_number': cond_num,
        'effective_rank': eff_rank,
        'rank_deficiency': rank_deficiency,
        'frobenius_norm': frob_norm,
        'spectral_radius': spectral_radius,
        'stability_class': stability,
        'is_singular': cond_num > 1e15
    }
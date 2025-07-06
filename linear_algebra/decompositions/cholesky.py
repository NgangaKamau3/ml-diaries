"""
Cholesky Decomposition
=====================

Cholesky decomposition for positive definite matrices: A = LLᵀ
More efficient than LU for symmetric positive definite systems.
"""

import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector


def cholesky_decompose(matrix: Matrix) -> Matrix:
    """Compute Cholesky decomposition A = LLᵀ."""
    if not matrix.is_square():
        raise ValueError("Cholesky requires square matrix")
    if not matrix.is_symmetric():
        raise ValueError("Cholesky requires symmetric matrix")
    
    n = matrix.rows
    L = Matrix.zeros(n, n)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal element
                sum_sq = sum(L[i, k]**2 for k in range(j))
                val = matrix[i, i] - sum_sq
                if val <= 0:
                    raise ValueError("Matrix not positive definite")
                L[i, j] = math.sqrt(val)
            else:
                # Off-diagonal element
                sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (matrix[i, j] - sum_prod) / L[j, j]
    
    return L


def solve_cholesky(L: Matrix, b: Vector) -> Vector:
    """Solve Ax = b using Cholesky decomposition A = LLᵀ."""
    # Forward substitution: Ly = b
    y = solve_lower_triangular(L, b)
    # Back substitution: Lᵀx = y
    x = solve_upper_triangular(L.transpose(), y)
    return x


def solve_lower_triangular(L: Matrix, b: Vector) -> Vector:
    """Solve Lx = b where L is lower triangular."""
    n = L.rows
    x = Vector.zero(n)
    
    for i in range(n):
        sum_val = sum(L[i, j] * x[j] for j in range(i))
        x._components[i] = (b[i] - sum_val) / L[i, i]
    
    return x


def solve_upper_triangular(U: Matrix, b: Vector) -> Vector:
    """Solve Ux = b where U is upper triangular."""
    n = U.rows
    x = Vector.zero(n)
    
    for i in range(n - 1, -1, -1):
        sum_val = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x._components[i] = (b[i] - sum_val) / U[i, i]
    
    return x
"""
QR Decomposition
===============

QR decomposition using Gram-Schmidt and Householder methods.
Essential for eigenvalue computation and least squares.
"""

import math
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.matrix import Matrix
from core.vector import Vector


def gram_schmidt_qr(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Gram-Schmidt orthogonalization."""
    m, n = matrix.shape
    Q = Matrix.zeros(m, n)
    R = Matrix.zeros(n, n)
    
    for j in range(n):
        v = matrix.get_col(j)
        
        for i in range(j):
            q_i = Q.get_col(i)
            R[i, j] = q_i.dot(v)
            v = v - R[i, j] * q_i
        
        R[j, j] = v.magnitude()
        if abs(R[j, j]) < 1e-15:
            raise ValueError("Matrix columns are linearly dependent")
        
        Q.set_col(j, v.normalize())
    
    return Q, R


def householder_qr(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Householder reflections."""
    A = matrix.copy()
    m, n = A.shape
    Q = Matrix.identity(m)
    
    for k in range(min(m-1, n)):
        # Get column vector below diagonal
        x = Vector([A[i, k] for i in range(k, m)])
        
        # Compute Householder vector
        alpha = -math.copysign(x.magnitude(), x[0])
        e1 = Vector.standard_basis(len(x.components), 0)
        u = x - alpha * e1
        
        if u.magnitude() < 1e-15:
            continue
            
        v = u.normalize()
        
        # Apply Householder reflection to A
        for j in range(k, n):
            col = Vector([A[i, j] for i in range(k, m)])
            col_new = col - 2 * v.dot(col) * v
            for i in range(k, m):
                A[i, j] = col_new[i - k]
        
        # Update Q
        for j in range(m):
            col = Vector([Q[i, j] for i in range(k, m)])
            col_new = col - 2 * v.dot(col) * v
            for i in range(k, m):
                Q[i, j] = col_new[i - k]
    
    R = A
    return Q, R
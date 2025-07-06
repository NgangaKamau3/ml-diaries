"""
Linear Algebra from Scratch: LU Decomposition
============================================

This module implements LU decomposition and related linear system solving methods
using pure Python, following the mathematical rigor of Deisenroth, Faisal & Ong's 
"Mathematics for Machine Learning".

LU decomposition is a fundamental matrix factorization that decomposes a matrix A
into the product of a lower triangular matrix L and an upper triangular matrix U,
with optional partial pivoting for numerical stability.

Mathematical Foundation:
For a square matrix A ∈ R^(n×n), LU decomposition finds matrices L, U such that:
- PA = LU (with partial pivoting)
- L is lower triangular with unit diagonal (Lᵢᵢ = 1)
- U is upper triangular
- P is a permutation matrix (represented as pivot vector)

The decomposition is based on Gaussian elimination with partial pivoting,
which systematically eliminates elements below the diagonal while maintaining
numerical stability through row swapping.

Applications:
- Solving linear systems Ax = b efficiently
- Computing matrix determinants
- Matrix inversion
- Numerical stability analysis

Author: Ng'ang'a Kamau
License: MIT
"""

import math
from typing import List, Union, Tuple, Optional
from numbers import Number

# Import dependencies
try:
    from ..core.vector import Vector
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from core.vector import Vector

# We'll use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.matrix import Matrix
else:
    # Runtime import handling
    Matrix = None


class LUDecomposition:
    """
    LU Decomposition with partial pivoting implementation.
    
    Mathematical Foundation:
    Given a square matrix A ∈ R^(n×n), we seek matrices L, U, and permutation P such that:
    PA = LU
    
    Where:
    - L ∈ R^(n×n) is lower triangular with unit diagonal (Lᵢᵢ = 1)
    - U ∈ R^(n×n) is upper triangular  
    - P ∈ R^(n×n) is a permutation matrix
    
    The algorithm uses Gaussian elimination with partial pivoting:
    1. For each column k, find the largest element in magnitude from row k onwards
    2. Swap rows to bring this element to the diagonal (pivoting)
    3. Eliminate all elements below the diagonal in column k
    4. Store elimination factors in L
    
    Complexity: O(n³) arithmetic operations
    Storage: O(n²) for L and U matrices
    
    Numerical Properties:
    - Partial pivoting ensures |Lᵢⱼ| ≤ 1, providing numerical stability
    - Growth factor bounded by 2^(n-1) in worst case
    - Backward stable for most practical matrices
    """
    
    def __init__(self, matrix):
        """
        Initialize LU decomposition for the given matrix.
        
        Args:
            matrix: Square matrix to decompose
            
        Raises:
            ValueError: If matrix is not square
            TypeError: If matrix is not a Matrix instance
        """
        # Check if it's a Matrix-like object (has required methods)
        if not hasattr(matrix, 'is_square') or not hasattr(matrix, 'rows') or not hasattr(matrix, 'copy'):
            raise TypeError("Input must be a Matrix instance")
        
        if not matrix.is_square():
            raise ValueError("LU decomposition requires a square matrix")
        
        self._original_matrix = matrix.copy()
        self._n = matrix.rows
        self._L = None
        self._U = None
        self._pivot = None
        self._is_decomposed = False
        self._is_singular = False
        
    @property
    def original_matrix(self) -> Matrix:
        """Return the original matrix."""
        return self._original_matrix.copy()
    
    @property
    def size(self) -> int:
        """Return the size of the square matrix."""
        return self._n
    
    @property
    def is_decomposed(self) -> bool:
        """Check if decomposition has been computed."""
        return self._is_decomposed
    
    @property
    def is_singular(self) -> bool:
        """Check if the matrix was found to be singular during decomposition."""
        return self._is_singular
    
    def decompose(self, tolerance: float = 1e-15) -> Tuple[Matrix, Matrix, List[int]]:
        """
        Compute LU decomposition with partial pivoting: PA = LU
        
        Algorithm (Gaussian Elimination with Partial Pivoting):
        1. Initialize L = I, U = A, pivot = [0,1,2,...,n-1]
        2. For k = 0 to n-2:
           a. Find row with largest |U[i,k]| for i ≥ k (partial pivoting)
           b. Swap rows k and max_row in U and update pivot
           c. For i = k+1 to n-1:
              - Compute multiplier: m = U[i,k] / U[k,k]
              - Store multiplier: L[i,k] = m
              - Eliminate: U[i,j] -= m * U[k,j] for j = k to n-1
        
        Mathematical Properties:
        - If A is nonsingular, decomposition exists and is unique
        - Partial pivoting ensures numerical stability
        - |L[i,j]| ≤ 1 for all i,j (bounded multipliers)
        
        Args:
            tolerance: Threshold for detecting singular matrices
            
        Returns:
            Tuple of (L, U, pivot_list) where:
            - L: Lower triangular matrix with unit diagonal
            - U: Upper triangular matrix
            - pivot_list: Permutation vector representing row swaps
            
        Raises:
            ValueError: If matrix is singular (zero pivot encountered)
        """
        if self._is_decomposed:
            return self._L.copy(), self._U.copy(), self._pivot.copy()
        
        n = self._n
        
        # Initialize L as identity matrix and U as copy of original matrix
        # Get Matrix class dynamically
        Matrix = type(self._original_matrix)
        L = Matrix.identity(n)
        U = self._original_matrix.copy()
        pivot = list(range(n))
        
        # Gaussian elimination with partial pivoting
        for k in range(n - 1):
            # Find pivot: largest element in column k from row k onwards
            max_row = k
            max_val = abs(U[k, k])
            
            for i in range(k + 1, n):
                if abs(U[i, k]) > max_val:
                    max_val = abs(U[i, k])
                    max_row = i
            
            # Swap rows if needed (partial pivoting)
            if max_row != k:
                # Swap rows in U
                for j in range(n):
                    U[k, j], U[max_row, j] = U[max_row, j], U[k, j]
                
                # Update pivot vector
                pivot[k], pivot[max_row] = pivot[max_row], pivot[k]
                
                # Swap corresponding rows in L (for already computed part)
                for j in range(k):
                    L[k, j], L[max_row, j] = L[max_row, j], L[k, j]
            
            # Check for zero pivot (singular matrix)
            if abs(U[k, k]) < tolerance:
                self._is_singular = True
                raise ValueError(f"Matrix is singular: zero pivot encountered at position ({k},{k})")
            
            # Elimination step
            for i in range(k + 1, n):
                # Compute elimination factor (multiplier)
                factor = U[i, k] / U[k, k]
                
                # Store factor in L matrix
                L[i, k] = factor
                
                # Eliminate elements in row i
                for j in range(k, n):
                    U[i, j] -= factor * U[k, j]
        
        # Check final diagonal element
        if abs(U[n-1, n-1]) < tolerance:
            self._is_singular = True
            raise ValueError(f"Matrix is singular: zero pivot at position ({n-1},{n-1})")
        
        # Store results
        self._L = L
        self._U = U
        self._pivot = pivot
        self._is_decomposed = True
        
        return L.copy(), U.copy(), pivot.copy()
    
    def get_L(self) -> Matrix:
        """
        Get the lower triangular matrix L.
        
        Returns:
            Lower triangular matrix with unit diagonal
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            raise RuntimeError("Must call decompose() first")
        return self._L.copy()
    
    def get_U(self) -> Matrix:
        """
        Get the upper triangular matrix U.
        
        Returns:
            Upper triangular matrix
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            raise RuntimeError("Must call decompose() first")
        return self._U.copy()
    
    def get_pivot(self) -> List[int]:
        """
        Get the pivot vector representing row permutations.
        
        Returns:
            List representing the permutation applied to rows
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            raise RuntimeError("Must call decompose() first")
        return self._pivot.copy()
    
    def get_permutation_matrix(self) -> Matrix:
        """
        Get the permutation matrix P such that PA = LU.
        
        Mathematical Foundation:
        The permutation matrix P is constructed from the pivot vector.
        P[i,j] = 1 if j = pivot[i], 0 otherwise.
        
        Returns:
            Permutation matrix P
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            raise RuntimeError("Must call decompose() first")
        
        n = self._n
        Matrix = type(self._original_matrix)
        P = Matrix.zeros(n, n)
        
        for i in range(n):
            P[i, self._pivot[i]] = 1.0
        
        return P
    
    def verify_decomposition(self, tolerance: float = 1e-12) -> bool:
        """
        Verify that PA = LU within numerical tolerance.
        
        This is crucial for validating the correctness of the decomposition,
        especially in the presence of numerical errors.
        
        Args:
            tolerance: Maximum allowed error in matrix elements
            
        Returns:
            True if decomposition is valid, False otherwise
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            raise RuntimeError("Must call decompose() first")
        
        # Compute PA
        P = self.get_permutation_matrix()
        PA = P * self._original_matrix
        
        # Compute LU
        LU = self._L * self._U
        
        # Check if PA ≈ LU
        diff = PA - LU
        max_error = max(abs(diff[i, j]) for i in range(self._n) for j in range(self._n))
        
        return max_error < tolerance
    
    def solve(self, b: Vector) -> Vector:
        """
        Solve the linear system Ax = b using the LU decomposition.
        
        Algorithm:
        1. Ensure LU decomposition is computed
        2. Apply permutation to b: b' = Pb
        3. Forward substitution: solve Ly = b' for y
        4. Back substitution: solve Ux = y for x
        
        Mathematical Foundation:
        Given PA = LU, the system Ax = b becomes:
        PAx = Pb ⟹ LUx = Pb
        
        This is solved in two steps:
        - Ly = Pb (forward substitution)
        - Ux = y (back substitution)
        
        Complexity: O(n²) after decomposition is computed
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
            
        Raises:
            ValueError: If dimensions don't match
            RuntimeError: If decomposition hasn't been computed or matrix is singular
        """
        if not hasattr(b, 'dimension') or not hasattr(b, '__getitem__'):
            raise TypeError("Right-hand side must be a Vector")
        
        if b.dimension != self._n:
            raise ValueError(f"Vector dimension {b.dimension} doesn't match matrix size {self._n}")
        
        if not self._is_decomposed:
            self.decompose()
        
        if self._is_singular:
            raise RuntimeError("Cannot solve system: matrix is singular")
        
        # Apply permutation to b
        b_permuted = Vector([b[self._pivot[i]] for i in range(self._n)])
        
        # Forward substitution: Ly = Pb
        y = self._solve_lower_triangular(self._L, b_permuted, unit_diagonal=True)
        
        # Back substitution: Ux = y
        x = self._solve_upper_triangular(self._U, y)
        
        return x
    
    def solve_multiple(self, B: Matrix) -> Matrix:
        """
        Solve multiple systems AX = B where B has multiple columns.
        
        This is more efficient than solving each system individually
        when the same matrix A is used with different right-hand sides.
        
        Args:
            B: Matrix where each column is a right-hand side vector
            
        Returns:
            Matrix X where each column is a solution vector
            
        Raises:
            ValueError: If dimensions don't match
        """
        if not hasattr(B, 'rows') or not hasattr(B, 'cols') or not hasattr(B, 'get_col'):
            raise TypeError("Right-hand side must be a Matrix")
        
        if B.rows != self._n:
            raise ValueError(f"Matrix rows {B.rows} don't match matrix size {self._n}")
        
        # Solve for each column
        solutions = []
        for j in range(B.cols):
            b_j = B.get_col(j)
            x_j = self.solve(b_j)
            solutions.append(x_j)
        
        Matrix = type(self._original_matrix)
        return Matrix.from_vectors(solutions, by_columns=True)
    
    def determinant(self) -> float:
        """
        Compute the determinant using the LU decomposition.
        
        Mathematical Foundation:
        For PA = LU:
        det(A) = det(P)⁻¹ × det(L) × det(U)
        
        Since:
        - det(L) = 1 (unit diagonal)
        - det(U) = ∏ᵢ U[i,i] (product of diagonal elements)
        - det(P) = (-1)^(number of row swaps)
        
        Therefore: det(A) = (-1)^(swaps) × ∏ᵢ U[i,i]
        
        Returns:
            Determinant of the original matrix
            
        Raises:
            RuntimeError: If decomposition hasn't been computed
        """
        if not self._is_decomposed:
            self.decompose()
        
        if self._is_singular:
            return 0.0
        
        # Count number of row swaps to determine sign
        swaps = 0
        temp_pivot = list(range(self._n))
        
        for i in range(self._n):
            if temp_pivot[i] != self._pivot[i]:
                # Find where pivot[i] is located and swap
                j = temp_pivot.index(self._pivot[i])
                temp_pivot[i], temp_pivot[j] = temp_pivot[j], temp_pivot[i]
                swaps += 1
        
        # Compute product of diagonal elements of U
        det_U = 1.0
        for i in range(self._n):
            det_U *= self._U[i, i]
        
        # Apply sign from permutation
        sign = 1 if swaps % 2 == 0 else -1
        
        return sign * det_U
    
    def _solve_lower_triangular(self, L: Matrix, b: Vector, unit_diagonal: bool = False) -> Vector:
        """
        Solve Lx = b where L is lower triangular using forward substitution.
        
        Forward Substitution Algorithm:
        For i = 0 to n-1:
            x[i] = (b[i] - Σⱼ₌₀ⁱ⁻¹ L[i,j] × x[j]) / L[i,i]
        
        If unit_diagonal=True, assumes L[i,i] = 1 for all i.
        
        Mathematical Foundation:
        Lower triangular systems can be solved efficiently by forward substitution
        because each equation involves only previously computed unknowns:
        
        L₁₁x₁ = b₁                    ⟹ x₁ = b₁/L₁₁
        L₂₁x₁ + L₂₂x₂ = b₂          ⟹ x₂ = (b₂ - L₂₁x₁)/L₂₂
        L₃₁x₁ + L₃₂x₂ + L₃₃x₃ = b₃  ⟹ x₃ = (b₃ - L₃₁x₁ - L₃₂x₂)/L₃₃
        
        Complexity: O(n²)
        
        Args:
            L: Lower triangular matrix
            b: Right-hand side vector
            unit_diagonal: If True, assumes diagonal elements are 1
            
        Returns:
            Solution vector x
        """
        n = L.rows
        x = Vector.zero(n)
        
        for i in range(n):
            # Compute sum of L[i,j] * x[j] for j < i
            sum_val = sum(L[i, j] * x[j] for j in range(i))
            
            if unit_diagonal:
                x._components[i] = b[i] - sum_val
            else:
                if abs(L[i, i]) < 1e-15:
                    raise ValueError(f"Zero diagonal element at position {i}")
                x._components[i] = (b[i] - sum_val) / L[i, i]
        
        return x
    
    def _solve_upper_triangular(self, U: Matrix, b: Vector) -> Vector:
        """
        Solve Ux = b where U is upper triangular using back substitution.
        
        Back Substitution Algorithm:
        For i = n-1 down to 0:
            x[i] = (b[i] - Σⱼ₌ᵢ₊₁ⁿ⁻¹ U[i,j] × x[j]) / U[i,i]
        
        Mathematical Foundation:
        Upper triangular systems are solved by back substitution, starting
        from the last equation and working backwards:
        
        Uₙₙxₙ = bₙ                           ⟹ xₙ = bₙ/Uₙₙ
        Uₙ₋₁,ₙ₋₁xₙ₋₁ + Uₙ₋₁,ₙxₙ = bₙ₋₁     ⟹ xₙ₋₁ = (bₙ₋₁ - Uₙ₋₁,ₙxₙ)/Uₙ₋₁,ₙ₋₁
        
        Complexity: O(n²)
        
        Args:
            U: Upper triangular matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        n = U.rows
        x = Vector.zero(n)
        
        for i in range(n - 1, -1, -1):
            # Compute sum of U[i,j] * x[j] for j > i
            sum_val = sum(U[i, j] * x[j] for j in range(i + 1, n))
            
            if abs(U[i, i]) < 1e-15:
                raise ValueError(f"Zero diagonal element at position {i}")
            
            x._components[i] = (b[i] - sum_val) / U[i, i]
        
        return x


# ==================== STANDALONE FUNCTIONS ====================

def lu_decompose(matrix, tolerance: float = 1e-15):
    """
    Compute LU decomposition with partial pivoting for a given matrix.
    
    This is a convenience function that creates an LUDecomposition instance
    and returns the decomposition matrices.
    
    Args:
        matrix: Square matrix to decompose
        tolerance: Threshold for detecting singular matrices
        
    Returns:
        Tuple of (L, U, pivot_list)
        
    Raises:
        ValueError: If matrix is not square or is singular
    """
    lu_decomp = LUDecomposition(matrix)
    return lu_decomp.decompose(tolerance)


def solve_linear_system(A, b: Vector) -> Vector:
    """
    Solve the linear system Ax = b using LU decomposition.
    
    This is a convenience function for solving a single linear system.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        Solution vector x
        
    Raises:
        ValueError: If matrix is not square, dimensions don't match, or matrix is singular
    """
    lu_decomp = LUDecomposition(A)
    return lu_decomp.solve(b)


def matrix_determinant_lu(matrix) -> float:
    """
    Compute matrix determinant using LU decomposition.
    
    This is often more numerically stable than direct computation
    for larger matrices.
    
    Args:
        matrix: Square matrix
        
    Returns:
        Determinant of the matrix
        
    Raises:
        ValueError: If matrix is not square
    """
    lu_decomp = LUDecomposition(matrix)
    return lu_decomp.determinant()


def matrix_inverse_lu(matrix):
    """
    Compute matrix inverse using LU decomposition.
    
    Algorithm:
    1. Compute LU decomposition of A
    2. Solve AX = I column by column
    3. Each column xᵢ of X satisfies Axᵢ = eᵢ
    
    This is more efficient than using the adjugate matrix method
    for larger matrices.
    
    Args:
        matrix: Square invertible matrix
        
    Returns:
        Inverse matrix A⁻¹
        
    Raises:
        ValueError: If matrix is not square or is singular
    """
    if not matrix.is_square():
        raise ValueError("Only square matrices can be inverted")
    
    n = matrix.rows
    lu_decomp = LUDecomposition(matrix)
    
    # Check if matrix is invertible
    if abs(lu_decomp.determinant()) < 1e-15:
        raise ValueError("Matrix is singular and cannot be inverted")
    
    # Create identity matrix and solve AX = I
    Matrix = type(matrix)
    I = Matrix.identity(n)
    X = lu_decomp.solve_multiple(I)
    
    return X


# ==================== NUMERICAL ANALYSIS FUNCTIONS ====================

def condition_number_lu(matrix) -> float:
    """
    Estimate the condition number using LU decomposition.
    
    The condition number κ(A) = ||A|| × ||A⁻¹|| measures how sensitive
    the solution of Ax = b is to perturbations in A or b.
    
    Args:
        matrix: Square matrix
        
    Returns:
        Condition number estimate
        
    Raises:
        ValueError: If matrix is not square or is singular
    """
    if not matrix.is_square():
        raise ValueError("Condition number is only defined for square matrices")
    
    try:
        # Compute matrix and its inverse
        A_inv = matrix_inverse_lu(matrix)
        
        # Estimate norms (using Frobenius norm for simplicity)
        norm_A = matrix.frobenius_norm()
        norm_A_inv = A_inv.frobenius_norm()
        
        return norm_A * norm_A_inv
    
    except ValueError:
        # Matrix is singular
        return float('inf')


def growth_factor(matrix) -> float:
    """
    Compute the growth factor for LU decomposition.
    
    The growth factor ρ = max|U[i,j]| / max|A[i,j]| measures how much
    the elements can grow during Gaussian elimination. Large growth
    factors indicate potential numerical instability.
    
    Mathematical Foundation:
    For partial pivoting, the growth factor is bounded by 2^(n-1),
    but this bound is rarely achieved in practice. Most matrices
    have growth factors close to 1.
    
    Args:
        matrix: Square matrix
        
    Returns:
        Growth factor ρ
        
    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError("Growth factor is only defined for square matrices")
    
    lu_decomp = LUDecomposition(matrix)
    
    try:
        L, U, _ = lu_decomp.decompose()
        
        # Find maximum element in original matrix
        max_A = max(abs(matrix[i, j]) for i in range(matrix.rows) for j in range(matrix.cols))
        
        # Find maximum element in U
        max_U = max(abs(U[i, j]) for i in range(U.rows) for j in range(U.cols))
        
        if max_A < 1e-15:
            return 1.0  # Avoid division by zero for zero matrix
        
        return max_U / max_A
    
    except ValueError:
        # Matrix is singular
        return float('inf')


def residual_norm(A, x: Vector, b: Vector) -> float:
    """
    Compute the residual norm ||Ax - b|| for solution verification.
    
    The residual measures how well x satisfies the equation Ax = b.
    Small residuals indicate accurate solutions, while large residuals
    may indicate numerical errors or ill-conditioning.
    
    Args:
        A: Coefficient matrix
        x: Solution vector
        b: Right-hand side vector
        
    Returns:
        Residual norm ||Ax - b||₂
        
    Raises:
        ValueError: If dimensions don't match
    """
    if A.rows != b.dimension or A.cols != x.dimension:
        raise ValueError("Matrix and vector dimensions must be compatible")
    
    # Compute residual r = Ax - b
    Ax = A * x
    residual = Ax - b
    
    return residual.magnitude()


# ==================== SPECIALIZED LU VARIANTS ====================

class LUDecompositionNoPivoting:
    """
    LU decomposition without pivoting for special matrices.
    
    This variant is faster but less numerically stable. It should only
    be used for matrices that are known to be well-conditioned and
    don't require pivoting (e.g., diagonally dominant matrices).
    
    Mathematical Foundation:
    For matrices where pivoting is not necessary, we can perform
    Gaussian elimination directly without row swaps:
    A = LU (no permutation matrix needed)
    
    Advantages:
    - Faster computation (no pivot search)
    - Simpler implementation
    - Preserves matrix structure (e.g., band matrices)
    
    Disadvantages:
    - May be numerically unstable
    - Can fail for general matrices
    - No guarantee of small multipliers
    """
    
    def __init__(self, matrix):
        """Initialize LU decomposition without pivoting."""
        if not hasattr(matrix, 'is_square') or not hasattr(matrix, 'rows') or not hasattr(matrix, 'copy'):
            raise TypeError("Input must be a Matrix instance")
        
        if not matrix.is_square():
            raise ValueError("LU decomposition requires a square matrix")
        
        self._original_matrix = matrix.copy()
        self._n = matrix.rows
        self._L = None
        self._U = None
        self._is_decomposed = False
    
    def decompose(self, tolerance: float = 1e-15) -> Tuple[Matrix, Matrix]:
        """
        Compute LU decomposition without pivoting: A = LU
        
        Algorithm:
        1. Initialize L = I, U = A
        2. For k = 0 to n-2:
           a. Check if U[k,k] ≠ 0 (no pivoting available)
           b. For i = k+1 to n-1:
              - Compute multiplier: m = U[i,k] / U[k,k]
              - Store multiplier: L[i,k] = m
              - Eliminate: U[i,j] -= m * U[k,j] for j = k to n-1
        
        Args:
            tolerance: Threshold for detecting zero pivots
            
        Returns:
            Tuple of (L, U)
            
        Raises:
            ValueError: If zero pivot is encountered
        """
        if self._is_decomposed:
            return self._L.copy(), self._U.copy()
        
        n = self._n
        Matrix = type(self._original_matrix)
        L = Matrix.identity(n)
        U = self._original_matrix.copy()
        
        for k in range(n - 1):
            # Check for zero pivot
            if abs(U[k, k]) < tolerance:
                raise ValueError(f"Zero pivot encountered at position ({k},{k}). Consider using pivoting.")
            
            # Elimination without pivoting
            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                
                for j in range(k, n):
                    U[i, j] -= factor * U[k, j]
        
        # Check final diagonal element
        if abs(U[n-1, n-1]) < tolerance:
            raise ValueError(f"Zero pivot at final position ({n-1},{n-1})")
        
        self._L = L
        self._U = U
        self._is_decomposed = True
        
        return L.copy(), U.copy()
    
    def solve(self, b: Vector) -> Vector:
        """Solve Ax = b using LU decomposition without pivoting."""
        if not self._is_decomposed:
            self.decompose()
        
        # Forward substitution: Ly = b
        y = self._solve_lower_triangular(self._L, b, unit_diagonal=True)
        
        # Back substitution: Ux = y
        x = self._solve_upper_triangular(self._U, y)
        
        return x
    
    def _solve_lower_triangular(self, L: Matrix, b: Vector, unit_diagonal: bool = False) -> Vector:
        """Solve lower triangular system."""
        n = L.rows
        x = Vector.zero(n)
        
        for i in range(n):
            sum_val = sum(L[i, j] * x[j] for j in range(i))
            
            if unit_diagonal:
                x._components[i] = b[i] - sum_val
            else:
                x._components[i] = (b[i] - sum_val) / L[i, i]
        
        return x
    
    def _solve_upper_triangular(self, U: Matrix, b: Vector) -> Vector:
        """Solve upper triangular system."""
        n = U.rows
        x = Vector.zero(n)
        
        for i in range(n - 1, -1, -1):
            sum_val = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x._components[i] = (b[i] - sum_val) / U[i, i]
        
        return x


# ==================== TESTING AND VALIDATION ====================

def test_lu_decomposition():
    """
    Comprehensive test suite for LU decomposition implementations.
    
    Tests mathematical properties, numerical accuracy, and edge cases.
    """
    print("Testing LU Decomposition Library...")
    print("=" * 50)
    
    # Test 1: Basic decomposition
    print("\n1. Testing basic LU decomposition:")
    A = Matrix([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    lu_decomp = LUDecomposition(A)
    L, U, pivot = lu_decomp.decompose()
    
    print(f"Original matrix A:")
    print(A)
    print(f"\nL matrix:")
    print(L)
    print(f"\nU matrix:")
    print(U)
    print(f"\nPivot vector: {pivot}")
    
    # Verify decomposition
    is_valid = lu_decomp.verify_decomposition()
    print(f"Decomposition valid: {is_valid}")
    
    # Test 2: Solving linear system
    print("\n2. Testing linear system solving:")
    b = Vector([4, 10, 24])
    x = lu_decomp.solve(b)
    print(f"Right-hand side b: {b}")
    print(f"Solution x: {x}")
    
    # Verify solution
    residual = residual_norm(A, x, b)
    print(f"Residual norm: {residual:.2e}")
    
    # Test 3: Determinant computation
    print("\n3. Testing determinant computation:")
    det_lu = lu_decomp.determinant()
    print(f"Determinant (via LU): {det_lu:.6f}")
    
    # Test 4: Matrix inversion
    print("\n4. Testing matrix inversion:")
    try:
        A_inv = matrix_inverse_lu(A)
        print(f"Inverse matrix:")
        print(A_inv)
        
        # Verify A * A^(-1) ≈ I
        product = A * A_inv
        print(f"A * A^(-1):")
        print(product)
        
    except ValueError as e:
        print(f"Inversion failed: {e}")
    
    # Test 5: Condition number
    print("\n5. Testing condition number:")
    cond_num = condition_number_lu(A)
    print(f"Condition number: {cond_num:.2e}")
    
    # Test 6: Growth factor
    print("\n6. Testing growth factor:")
    growth = growth_factor(A)
    print(f"Growth factor: {growth:.6f}")
    
    # Test 7: Ill-conditioned matrix (Hilbert matrix)
    print("\n7. Testing ill-conditioned matrix:")
    try:
        # Create 4x4 Hilbert matrix
        H = Matrix([[1/(i+j+1) for j in range(4)] for i in range(4)])
        print(f"4x4 Hilbert matrix:")
        print(H)
        
        lu_hilbert = LUDecomposition(H)
        cond_hilbert = condition_number_lu(H)
        print(f"Hilbert matrix condition number: {cond_hilbert:.2e}")
        
        # Test solving with Hilbert matrix
        b_hilbert = Vector([1, 1, 1, 1])
        x_hilbert = lu_hilbert.solve(b_hilbert)
        residual_hilbert = residual_norm(H, x_hilbert, b_hilbert)
        print(f"Hilbert system residual: {residual_hilbert:.2e}")
        
    except Exception as e:
        print(f"Hilbert matrix test failed: {e}")
    
    # Test 8: Singular matrix
    print("\n8. Testing singular matrix:")
    try:
        S = Matrix([[1, 2, 3], [2, 4, 6], [1, 2, 3]])  # Rank-deficient
        lu_singular = LUDecomposition(S)
        lu_singular.decompose()
        print("ERROR: Singular matrix should have raised an exception!")
    except ValueError as e:
        print(f"Correctly detected singular matrix: {e}")
    
    print("\n" + "=" * 50)
    print("LU Decomposition tests completed!")


def demonstrate_lu_applications():
    """
    Demonstrate practical applications of LU decomposition.
    """
    print("\nLU Decomposition Applications Demo")
    print("=" * 40)
    
    # Application 1: Solving multiple systems with same coefficient matrix
    print("\n1. Multiple Right-Hand Sides:")
    A = Matrix([[3, 2, 1], [2, 3, 2], [1, 2, 3]])
    B = Matrix([[6, 1], [8, 2], [9, 3]])  # Two right-hand sides
    
    lu_decomp = LUDecomposition(A)
    X = lu_decomp.solve_multiple(B)
    
    print(f"Coefficient matrix A:")
    print(A)
    print(f"Right-hand sides B:")
    print(B)
    print(f"Solutions X:")
    print(X)
    
    # Application 2: Iterative refinement
    print("\n2. Iterative Refinement for Improved Accuracy:")
    A = Matrix([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])
    b = Vector([32, 23, 33, 31])
    
    lu_decomp = LUDecomposition(A)
    x = lu_decomp.solve(b)
    
    print(f"Initial solution: {x}")
    print(f"Initial residual: {residual_norm(A, x, b):.2e}")
    
    # One step of iterative refinement
    r = b - A * x  # Compute residual
    dx = lu_decomp.solve(r)  # Solve for correction
    x_refined = x + dx  # Apply correction
    
    print(f"Refined solution: {x_refined}")
    print(f"Refined residual: {residual_norm(A, x_refined, b):.2e}")
    
    # Application 3: Computing matrix powers efficiently
    print("\n3. Matrix Powers via Repeated Solving:")
    A = Matrix([[0.8, 0.3], [0.2, 0.7]])  # Markov transition matrix
    
    # Compute A^10 by solving (I - A)x = -A^9 iteratively
    # This is just a demonstration - not the most efficient method
    print(f"Original matrix A:")
    print(A)
    
    det_A = matrix_determinant_lu(A)
    print(f"Determinant of A: {det_A:.6f}")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Run comprehensive tests
    test_lu_decomposition()
    
    # Demonstrate applications
    demonstrate_lu_applications()
    
    print("\n" + "=" * 60)
    print("LU DECOMPOSITION MODULE DEMONSTRATION COMPLETE")
    print("=" * 60)
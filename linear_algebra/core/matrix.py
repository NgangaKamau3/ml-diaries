"""
Linear Algebra from Scratch: Matrix Operations
============================================

This module implements fundamental matrix operations using pure Python,
following the mathematical rigor of Deisenroth, Faisal & Ong's "Mathematics for Machine Learning".

A matrix A ∈ R^(m×n) is a rectangular array of real numbers with m rows and n columns.
Matrices are fundamental to linear algebra as they represent linear transformations,
systems of equations, and provide a compact notation for multivariate operations.

Mathematical Foundation:
- A matrix A = [aᵢⱼ] where aᵢⱼ is the element in row i, column j
- Matrix operations follow specific algebraic rules that preserve linearity
- Matrices can represent linear transformations T: R^n → R^m via T(x) = Ax

Author: Ng'ang'a Kamau
License: MIT
"""

import math
from typing import List, Union, Tuple, Optional, Callable
from numbers import Number
import copy

# Import our vector module for integration
try:
    from .vector import Vector
except ImportError:
    # Fallback for standalone testing
    from vector import Vector


class Matrix:
    """
    A matrix in R^(m×n) implemented as a pure Python class.
    
    Mathematical Foundation:
    A matrix A ∈ R^(m×n) is a rectangular array of real numbers:
    
    A = [a₁₁  a₁₂  ...  a₁ₙ]
        [a₂₁  a₂₂  ...  a₂ₙ]
        [ ⋮    ⋮   ⋱    ⋮ ]
        [aₘ₁  aₘ₂  ...  aₘₙ]
    
    The set of all m×n matrices forms a vector space under matrix addition
    and scalar multiplication, with dimension mn.
    
    Key Properties:
    1. Matrix addition is commutative and associative
    2. Matrix multiplication is associative but not commutative
    3. Distributive laws hold for matrix operations
    4. The zero matrix is the additive identity
    """
    
    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Initialize a matrix from a list of lists (row-major order).
        
        Args:
            data: List of lists representing matrix rows
            
        Raises:
            ValueError: If data is empty or rows have inconsistent lengths
            TypeError: If data contains non-numeric values
        """
        if not data:
            raise ValueError("Matrix must have at least one row")
        
        if not all(isinstance(row, list) for row in data):
            raise TypeError("Matrix data must be a list of lists")
        
        if not data[0]:
            raise ValueError("Matrix must have at least one column")
        
        # Check all rows have same length
        n_cols = len(data[0])
        for i, row in enumerate(data):
            if len(row) != n_cols:
                raise ValueError(f"Row {i} has {len(row)} elements, expected {n_cols}")
        
        # Validate numeric data
        for i, row in enumerate(data):
            for j, elem in enumerate(row):
                if not isinstance(elem, Number):
                    raise TypeError(f"Element at ({i},{j}) must be numeric, got {type(elem)}")
        
        # Store as list of lists of floats
        self._data = [[float(elem) for elem in row] for row in data]
        self._rows = len(data)
        self._cols = n_cols
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape (rows, columns) of the matrix."""
        return (self._rows, self._cols)
    
    @property
    def rows(self) -> int:
        """Return the number of rows."""
        return self._rows
    
    @property
    def cols(self) -> int:
        """Return the number of columns."""
        return self._cols
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self._rows * self._cols
    
    def __getitem__(self, key) -> Union[float, List[float]]:
        """
        Access matrix elements or rows.
        
        Usage:
        - A[i, j]: Get element at row i, column j
        - A[i]: Get row i as a list
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix indexing requires exactly 2 indices")
            i, j = key
            return self._data[i][j]
        else:
            # Return entire row
            return self._data[key].copy()
    
    def __setitem__(self, key, value):
        """
        Set matrix elements or rows.
        
        Usage:
        - A[i, j] = value: Set element at row i, column j
        - A[i] = row_list: Set entire row i
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix indexing requires exactly 2 indices")
            i, j = key
            if not isinstance(value, Number):
                raise TypeError("Matrix element must be numeric")
            self._data[i][j] = float(value)
        else:
            # Set entire row
            if not isinstance(value, (list, tuple)):
                raise TypeError("Row must be a list or tuple")
            if len(value) != self._cols:
                raise ValueError(f"Row must have {self._cols} elements")
            self._data[key] = [float(x) for x in value]
    
    def __str__(self) -> str:
        """String representation of matrix with aligned columns."""
        if self._rows == 0 or self._cols == 0:
            return "Matrix([])"
        
        # Format each element
        str_elements = [[f"{elem:8.4f}" for elem in row] for row in self._data]
        
        # Create aligned output
        lines = []
        for i, row in enumerate(str_elements):
            if i == 0:
                lines.append("⎡" + " ".join(row) + "⎤")
            elif i == self._rows - 1:
                lines.append("⎣" + " ".join(row) + "⎦")
            else:
                lines.append("⎢" + " ".join(row) + "⎥")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Matrix({self._data})"
    
    def __eq__(self, other) -> bool:
        """
        Test matrix equality with numerical tolerance.
        
        Two matrices are equal if they have the same shape and
        corresponding elements are equal within floating-point precision.
        """
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        
        tolerance = 1e-10
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j] - other._data[i][j]) > tolerance:
                    return False
        return True
    
    # ==================== MATRIX CONSTRUCTION METHODS ====================
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        """
        Create a zero matrix of specified dimensions.
        
        The zero matrix 0 ∈ R^(m×n) has all elements equal to 0.
        It serves as the additive identity: A + 0 = A for any matrix A.
        """
        if rows < 1 or cols < 1:
            raise ValueError("Matrix dimensions must be positive")
        return cls([[0.0] * cols for _ in range(rows)])
    
    @classmethod
    def ones(cls, rows: int, cols: int) -> 'Matrix':
        """Create a matrix of ones of specified dimensions."""
        if rows < 1 or cols < 1:
            raise ValueError("Matrix dimensions must be positive")
        return cls([[1.0] * cols for _ in range(rows)])
    
    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        """
        Create an n×n identity matrix.
        
        The identity matrix I ∈ R^(n×n) has ones on the diagonal and zeros elsewhere:
        I = [δᵢⱼ] where δᵢⱼ = 1 if i=j, 0 otherwise (Kronecker delta)
        
        Properties:
        - AI = IA = A for any compatible matrix A
        - I is the multiplicative identity for square matrices
        - det(I) = 1, rank(I) = n
        """
        if n < 1:
            raise ValueError("Identity matrix dimension must be positive")
        
        data = [[0.0] * n for _ in range(n)]
        for i in range(n):
            data[i][i] = 1.0
        return cls(data)
    
    @classmethod
    def diagonal(cls, diagonal_elements: List[Union[int, float]]) -> 'Matrix':
        """
        Create a diagonal matrix from a list of diagonal elements.
        
        A diagonal matrix D has non-zero elements only on the main diagonal:
        D = diag(d₁, d₂, ..., dₙ)
        
        Properties:
        - Diagonal matrices commute under multiplication
        - Easy to compute powers: D^k = diag(d₁^k, d₂^k, ..., dₙ^k)
        - det(D) = d₁ × d₂ × ... × dₙ
        """
        if not diagonal_elements:
            raise ValueError("Must provide at least one diagonal element")
        
        n = len(diagonal_elements)
        data = [[0.0] * n for _ in range(n)]
        for i, elem in enumerate(diagonal_elements):
            if not isinstance(elem, Number):
                raise TypeError(f"Diagonal element {i} must be numeric")
            data[i][i] = float(elem)
        return cls(data)
    
    @classmethod
    def from_vectors(cls, vectors: List[Vector], by_columns: bool = True) -> 'Matrix':
        """
        Create a matrix from a list of vectors.
        
        Args:
            vectors: List of Vector objects
            by_columns: If True, vectors become columns; if False, vectors become rows
            
        Mathematical interpretation:
        - Column matrix: A = [v₁ | v₂ | ... | vₙ] where vᵢ are column vectors
        - Row matrix: A = [v₁ᵀ; v₂ᵀ; ...; vₙᵀ] where vᵢᵀ are row vectors
        """
        if not vectors:
            raise ValueError("Must provide at least one vector")
        
        # Check all vectors have same dimension
        dim = vectors[0].dimension
        for i, v in enumerate(vectors):
            if not isinstance(v, Vector):
                raise TypeError(f"Element {i} is not a Vector")
            if v.dimension != dim:
                raise ValueError("All vectors must have same dimension")
        
        if by_columns:
            # Vectors become columns
            data = [[0.0] * len(vectors) for _ in range(dim)]
            for j, vec in enumerate(vectors):
                for i in range(dim):
                    data[i][j] = vec[i]
        else:
            # Vectors become rows
            data = [vec.components for vec in vectors]
        
        return cls(data)
    
    @classmethod
    def random(cls, rows: int, cols: int, low: float = -1.0, high: float = 1.0) -> 'Matrix':
        """
        Create a random matrix with elements uniformly distributed in [low, high].
        
        Random matrices are useful for:
        - Testing algorithms and numerical stability
        - Monte Carlo methods
        - Initialization of iterative methods
        """
        import random
        if rows < 1 or cols < 1:
            raise ValueError("Matrix dimensions must be positive")
        
        data = [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]
        return cls(data)
    
    # ==================== MATRIX ARITHMETIC OPERATIONS ====================
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix addition: C = A + B where cᵢⱼ = aᵢⱼ + bᵢⱼ
        
        Mathematical properties:
        - Commutative: A + B = B + A
        - Associative: (A + B) + C = A + (B + C)
        - Identity: A + 0 = A
        - Additive inverse: A + (-A) = 0
        
        Requirement: Both matrices must have the same dimensions.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only add Matrix to Matrix")
        if self.shape != other.shape:
            raise ValueError(f"Cannot add matrices of different shapes: {self.shape} vs {other.shape}")
        
        result_data = []
        for i in range(self._rows):
            result_row = []
            for j in range(self._cols):
                result_row.append(self._data[i][j] + other._data[i][j])
            result_data.append(result_row)
        
        return Matrix(result_data)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix subtraction: C = A - B where cᵢⱼ = aᵢⱼ - bᵢⱼ
        
        Subtraction is defined as addition of the additive inverse: A - B = A + (-B)
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract Matrix from Matrix")
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract matrices of different shapes: {self.shape} vs {other.shape}")
        
        result_data = []
        for i in range(self._rows):
            result_row = []
            for j in range(self._cols):
                result_row.append(self._data[i][j] - other._data[i][j])
            result_data.append(result_row)
        
        return Matrix(result_data)
    
    def __mul__(self, other: Union['Matrix', Number, Vector]) -> Union['Matrix', Vector]:
        """
        Matrix multiplication and scalar multiplication.
        
        Cases:
        1. Matrix * Scalar: Scalar multiplication
        2. Matrix * Matrix: Matrix multiplication
        3. Matrix * Vector: Matrix-vector multiplication
        """
        if isinstance(other, Number):
            # Scalar multiplication: (αA)ᵢⱼ = α × aᵢⱼ
            return self._scalar_multiply(other)
        elif isinstance(other, Matrix):
            # Matrix multiplication
            return self._matrix_multiply(other)
        elif hasattr(other, 'dimension') and hasattr(other, '__getitem__'):
            # Matrix-vector multiplication (duck typing for Vector)
            return self._matrix_vector_multiply(other)
        else:
            raise TypeError(f"Cannot multiply Matrix by {type(other)}")
    
    def __rmul__(self, scalar: Number) -> 'Matrix':
        """Right scalar multiplication: scalar * Matrix"""
        if not isinstance(scalar, Number):
            raise TypeError("Can only right-multiply Matrix by scalar")
        return self._scalar_multiply(scalar)
    
    def __truediv__(self, scalar: Number) -> 'Matrix':
        """Matrix division by scalar: A/α = (1/α)A"""
        if not isinstance(scalar, Number):
            raise TypeError("Can only divide Matrix by scalar")
        if abs(scalar) < 1e-15:
            raise ValueError("Cannot divide by zero or near-zero scalar")
        return self._scalar_multiply(1.0 / scalar)
    
    def __neg__(self) -> 'Matrix':
        """Additive inverse: -A where (-A)ᵢⱼ = -aᵢⱼ"""
        result_data = [[-elem for elem in row] for row in self._data]
        return Matrix(result_data)
    
    def __pow__(self, exponent: int) -> 'Matrix':
        """
        Matrix exponentiation: A^n = A × A × ... × A (n times)
        
        Requirements:
        - Matrix must be square
        - Exponent must be non-negative integer
        
        Special cases:
        - A^0 = I (identity matrix)
        - A^1 = A
        """
        if not isinstance(exponent, int):
            raise TypeError("Matrix exponent must be integer")
        if exponent < 0:
            raise ValueError("Matrix exponentiation with negative exponents not implemented")
        if not self.is_square():
            raise ValueError("Matrix exponentiation requires square matrix")
        
        if exponent == 0:
            return Matrix.identity(self._rows)
        elif exponent == 1:
            return self.copy()
        else:
            # Use repeated squaring for efficiency
            result = Matrix.identity(self._rows)
            base = self.copy()
            
            while exponent > 0:
                if exponent % 2 == 1:
                    result = result * base
                base = base * base
                exponent //= 2
            
            return result
    
    def _scalar_multiply(self, scalar: Number) -> 'Matrix':
        """Internal method for scalar multiplication."""
        result_data = [[scalar * elem for elem in row] for row in self._data]
        return Matrix(result_data)
    
    def _matrix_multiply(self, other: 'Matrix') -> 'Matrix':
        """
        Internal method for matrix multiplication.
        
        Matrix multiplication C = AB is defined as:
        cᵢⱼ = Σₖ aᵢₖ × bₖⱼ (sum over k from 1 to n)
        
        Geometric interpretation: Matrix multiplication represents composition
        of linear transformations. If A represents transformation T₁ and B
        represents transformation T₂, then AB represents T₁ ∘ T₂.
        
        Requirements:
        - Number of columns in A must equal number of rows in B
        - Result has dimensions (A.rows × B.cols)
        
        Complexity: O(mnp) where A is m×n and B is n×p
        """
        if self._cols != other._rows:
            raise ValueError(f"Cannot multiply matrices: {self.shape} × {other.shape}. "
                           f"Number of columns in first matrix ({self._cols}) must equal "
                           f"number of rows in second matrix ({other._rows})")
        
        result_data = []
        for i in range(self._rows):
            result_row = []
            for j in range(other._cols):
                # Compute dot product of row i of self with column j of other
                elem = sum(self._data[i][k] * other._data[k][j] for k in range(self._cols))
                result_row.append(elem)
            result_data.append(result_row)
        
        return Matrix(result_data)
    
    def _matrix_vector_multiply(self, vector: Vector) -> Vector:
        """
        Internal method for matrix-vector multiplication.
        
        Matrix-vector multiplication y = Ax is defined as:
        yᵢ = Σⱼ aᵢⱼ × xⱼ (sum over j)
        
        This represents applying the linear transformation represented by A
        to the vector x.
        
        Requirements:
        - Number of columns in A must equal dimension of x
        - Result is a vector with dimension equal to number of rows in A
        """
        if self._cols != vector.dimension:
            raise ValueError(f"Cannot multiply matrix of shape {self.shape} with vector of dimension {vector.dimension}")
        
        result_components = []
        for i in range(self._rows):
            component = sum(self._data[i][j] * vector[j] for j in range(self._cols))
            result_components.append(component)
        
        return Vector(result_components)
    
    # ==================== MATRIX PROPERTIES AND QUERIES ====================
    
    def is_square(self) -> bool:
        """Check if the matrix is square (rows = columns)."""
        return self._rows == self._cols
    
    def is_symmetric(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is symmetric: A = Aᵀ
        
        A symmetric matrix satisfies aᵢⱼ = aⱼᵢ for all i,j.
        Symmetric matrices have special properties:
        - All eigenvalues are real
        - Eigenvectors are orthogonal
        - Can be diagonalized by orthogonal matrix
        """
        if not self.is_square():
            return False
        
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j] - self._data[j][i]) > tolerance:
                    return False
        return True
    
    def is_antisymmetric(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is antisymmetric (skew-symmetric): A = -Aᵀ
        
        An antisymmetric matrix satisfies aᵢⱼ = -aⱼᵢ for all i,j.
        This implies that all diagonal elements must be zero.
        """
        if not self.is_square():
            return False
        
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j] + self._data[j][i]) > tolerance:
                    return False
        return True
    
    def is_diagonal(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is diagonal: aᵢⱼ = 0 for i ≠ j
        
        Diagonal matrices have non-zero elements only on the main diagonal.
        """
        if not self.is_square():
            return False
        
        for i in range(self._rows):
            for j in range(self._cols):
                if i != j and abs(self._data[i][j]) > tolerance:
                    return False
        return True
    
    def is_upper_triangular(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is upper triangular: aᵢⱼ = 0 for i > j
        
        Upper triangular matrices have zeros below the main diagonal.
        """
        for i in range(self._rows):
            for j in range(min(i, self._cols)):
                if abs(self._data[i][j]) > tolerance:
                    return False
        return True
    
    def is_lower_triangular(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is lower triangular: aᵢⱼ = 0 for i < j
        
        Lower triangular matrices have zeros above the main diagonal.
        """
        for i in range(self._rows):
            for j in range(i + 1, self._cols):
                if abs(self._data[i][j]) > tolerance:
                    return False
        return True
    
    def is_orthogonal(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the matrix is orthogonal: AᵀA = AAᵀ = I
        
        Orthogonal matrices preserve lengths and angles. They represent
        rotations and reflections in Euclidean space.
        
        Properties of orthogonal matrices:
        - det(A) = ±1
        - A⁻¹ = Aᵀ
        - Columns (and rows) form an orthonormal basis
        """
        if not self.is_square():
            return False
        
        try:
            # Check if AᵀA = I
            transpose = self.transpose()
            product = transpose * self
            identity = Matrix.identity(self._rows)
            return product == identity
        except:
            return False
    
    def is_identity(self, tolerance: float = 1e-10) -> bool:
        """Check if the matrix is the identity matrix."""
        if not self.is_square():
            return False
        
        for i in range(self._rows):
            for j in range(self._cols):
                expected = 1.0 if i == j else 0.0
                if abs(self._data[i][j] - expected) > tolerance:
                    return False
        return True
    
    def is_zero(self, tolerance: float = 1e-10) -> bool:
        """Check if the matrix is the zero matrix."""
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j]) > tolerance:
                    return False
        return True
    
    # ==================== MATRIX TRANSFORMATIONS ====================
    
    def transpose(self) -> 'Matrix':
        """
        Compute the transpose of the matrix: (Aᵀ)ᵢⱼ = aⱼᵢ
        
        The transpose operation swaps rows and columns. It has important properties:
        - (Aᵀ)ᵀ = A
        - (A + B)ᵀ = Aᵀ + Bᵀ
        - (AB)ᵀ = BᵀAᵀ
        - (αA)ᵀ = αAᵀ
        
        Geometric interpretation: For linear transformations, the transpose
        represents the adjoint transformation.
        """
        result_data = [[self._data[i][j] for i in range(self._rows)] for j in range(self._cols)]
        return Matrix(result_data)
    
    def conjugate_transpose(self) -> 'Matrix':
        """
        Compute the conjugate transpose (Hermitian transpose).
        
        For real matrices, this is the same as transpose.
        For complex matrices (not implemented here), it would be (A*)ᵀ.
        """
        return self.transpose()
    
    def trace(self) -> float:
        """
        Compute the trace of the matrix: tr(A) = Σᵢ aᵢᵢ
        
        The trace is the sum of diagonal elements. Properties:
        - tr(A + B) = tr(A) + tr(B)
        - tr(αA) = α × tr(A)
        - tr(AB) = tr(BA) (cyclic property)
        - tr(A) = sum of eigenvalues
        
        Requirement: Matrix must be square.
        """
        if not self.is_square():
            raise ValueError("Trace is only defined for square matrices")
        
        return sum(self._data[i][i] for i in range(self._rows))
    
    def diagonal_elements(self) -> List[float]:
        """
        Extract the diagonal elements of the matrix.
        
        Returns the elements aᵢᵢ for i = 0, 1, ..., min(m,n)-1
        """
        min_dim = min(self._rows, self._cols)
        return [self._data[i][i] for i in range(min_dim)]
    
    def get_row(self, i: int) -> Vector:
        """Get row i as a Vector object."""
        if not (0 <= i < self._rows):
            raise IndexError(f"Row index {i} out of range for matrix with {self._rows} rows")
        return Vector(self._data[i])
    
    def get_col(self, j: int) -> Vector:
        """Get column j as a Vector object."""
        if not (0 <= j < self._cols):
            raise IndexError(f"Column index {j} out of range for matrix with {self._cols} columns")
        return Vector([self._data[i][j] for i in range(self._rows)])
    
    def set_row(self, i: int, vector: Vector) -> None:
        """Set row i from a Vector object."""
        if not (0 <= i < self._rows):
            raise IndexError(f"Row index {i} out of range")
        if vector.dimension != self._cols:
            raise ValueError(f"Vector dimension {vector.dimension} doesn't match matrix columns {self._cols}")
        
        self._data[i] = vector.components
    
    def set_col(self, j: int, vector: Vector) -> None:
        """Set column j from a Vector object."""
        if not (0 <= j < self._cols):
            raise IndexError(f"Column index {j} out of range")
        if vector.dimension != self._rows:
            raise ValueError(f"Vector dimension {vector.dimension} doesn't match matrix rows {self._rows}")
        
        for i in range(self._rows):
            self._data[i][j] = vector[i]
    
    # ==================== MATRIX NORMS ====================
    
    def frobenius_norm(self) -> float:
        """
        Compute the Frobenius norm: ||A||_F = √(Σᵢⱼ |aᵢⱼ|²)
        
        The Frobenius norm is the matrix analogue of the vector L2 norm.
        It measures the "size" of a matrix and is induced by the inner product
        ⟨A,B⟩ = tr(AᵀB).
        
        Properties:
        - ||A||_F = √tr(AᵀA)
        - Submultiplicative: ||AB||_F ≤ ||A||_F ||B||_F
        - Unitarily invariant: ||UAV||_F = ||A||_F for orthogonal U,V
        """
        sum_squares = sum(elem**2 for row in self._data for elem in row)
        return math.sqrt(sum_squares)
    
    def spectral_norm_estimate(self, max_iterations: int = 100) -> float:
        """
        Estimate the spectral norm (largest singular value) using power iteration.
        
        The spectral norm ||A||₂ is the largest singular value of A, which equals
        the square root of the largest eigenvalue of AᵀA.
        
        This is an approximation using the power method for eigenvalue estimation.
        """
        if self._rows == 0 or self._cols == 0:
            return 0.0
        
        # Start with random vector
        x = Vector.random(self._cols)
        x = x.normalize()
        
        for _ in range(max_iterations):
            # Apply AᵀA to x
            Ax = self * x
            ATAx = self.transpose() * Ax
            
            # Normalize
            norm = ATAx.magnitude()
            if norm < 1e-15:
                break
            x = ATAx.normalize()
        
        # Final estimate
        Ax = self * x
        return Ax.magnitude()
    
    def max_norm(self) -> float:
        """
        Compute the maximum absolute value of all elements.
        
        This is also known as the max norm or infinity norm for matrices.
        """
        return max(abs(elem) for row in self._data for elem in row)
    
    def row_sum_norm(self) -> float:
        """
        Compute the row sum norm (infinity norm): ||A||_∞ = maxᵢ Σⱼ |aᵢⱼ|
        
        This is the maximum absolute row sum.
        """
        return max(sum(abs(elem) for elem in row) for row in self._data)
    
    def col_sum_norm(self) -> float:
        """
        Compute the column sum norm (1-norm): ||A||₁ = maxⱼ Σᵢ |aᵢⱼ|
        
        This is the maximum absolute column sum.
        """
        return max(sum(abs(self._data[i][j]) for i in range(self._rows)) 
                  for j in range(self._cols))
    
    # ==================== MATRIX DECOMPOSITIONS ====================
    

    

    

    
    # ==================== SOLVING LINEAR SYSTEMS ====================
    

    

    

    
    def solve(self, b: Vector) -> Vector:
        """
        Solve the linear system Ax = b using LU decomposition.
        
        Algorithm:
        1. Compute PA = LU
        2. Solve Ly = Pb using forward substitution
        3. Solve Ux = y using back substitution
        
        This method automatically handles pivoting for numerical stability.
        
        Mathematical foundation:
        The LU decomposition transforms the system Ax = b into two triangular
        systems that can be solved efficiently. Partial pivoting ensures
        numerical stability by avoiding small pivots.
        
        Complexity: O(n³) for decomposition + O(n²) for solving = O(n³)
        """
        if not self.is_square():
            raise ValueError("Can only solve square systems")
        if self._rows != b.dimension:
            raise ValueError("Matrix and vector dimensions must match")
        
        # Use LU decomposition from the lu module
        try:
            from ..linalg.lu import solve_linear_system
            return solve_linear_system(self, b)
        except ImportError:
            # Fallback for standalone usage
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'linalg'))
            from linalg.lu import solve_linear_system
            return solve_linear_system(self, b)
    
    def solve_least_squares(self, b: Vector) -> Vector:
        """
        Solve the least squares problem: minimize ||Ax - b||₂
        
        For an overdetermined system (more equations than unknowns),
        finds the x that minimizes the squared residual.
        
        Mathematical foundation:
        The normal equations AᵀAx = Aᵀb give the least squares solution.
        However, we use QR decomposition for better numerical stability.
        
        Complexity: O(mn²) where A is m×n with m ≥ n
        """
        if self._rows < self._cols:
            raise ValueError("System is underdetermined (more unknowns than equations)")
        if self._rows != b.dimension:
            raise ValueError("Matrix and vector dimensions must match")
        
        # Use QR decomposition from decompositions module
        try:
            from ..decompositions.qr import gram_schmidt_qr
            Q, R = gram_schmidt_qr(self)
        except ImportError:
            # Fallback for standalone usage
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'decompositions'))
            from decompositions.qr import gram_schmidt_qr
            Q, R = gram_schmidt_qr(self)
        
        # Compute Qᵀb
        QT_b = Vector([Q.get_col(i).dot(b) for i in range(self._cols)])
        
        # Solve Rx = Qᵀb using back substitution
        n = R.rows
        x = Vector.zero(n)
        
        for i in range(n - 1, -1, -1):
            sum_val = sum(R[i, j] * x[j] for j in range(i + 1, n))
            if abs(R[i, i]) < 1e-15:
                raise ValueError(f"Zero diagonal element at position {i}")
            x._components[i] = (QT_b[i] - sum_val) / R[i, i]
        
        return x
    
    # ==================== MATRIX CALCULUS ====================
    
    def determinant(self) -> float:
        """
        Compute the determinant using LU decomposition.
        
        Mathematical foundation:
        The determinant is a scalar value that encodes important geometric
        and algebraic properties of the matrix:
        
        - det(A) ≠ 0 ⟺ A is invertible
        - |det(A)| = volume scaling factor of the linear transformation
        - det(AB) = det(A) × det(B)
        - det(Aᵀ) = det(A)
        
        For LU decomposition PA = LU:
        det(A) = det(P)⁻¹ × det(L) × det(U) = ±∏ᵢ uᵢᵢ
        
        The sign depends on the number of row swaps in the permutation.
        
        Complexity: O(n³)
        """
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        
        if self._rows == 1:
            return self._data[0][0]
        elif self._rows == 2:
            return self._data[0][0] * self._data[1][1] - self._data[0][1] * self._data[1][0]
        
        # Use LU decomposition from the lu module
        try:
            from ..linalg.lu import matrix_determinant_lu
            return matrix_determinant_lu(self)
        except ImportError:
            # Fallback for standalone usage
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'linalg'))
            from linalg.lu import matrix_determinant_lu
            return matrix_determinant_lu(self)
    
    def inverse(self) -> 'Matrix':
        """
        Compute the matrix inverse using LU decomposition.
        
        Mathematical foundation:
        The inverse A⁻¹ is the unique matrix such that AA⁻¹ = A⁻¹A = I.
        It exists if and only if det(A) ≠ 0.
        
        Algorithm:
        1. Compute LU decomposition of A
        2. Solve AX = I column by column
        3. Each column xᵢ of X satisfies Axᵢ = eᵢ
        
        Properties of matrix inverse:
        - (A⁻¹)⁻¹ = A
        - (AB)⁻¹ = B⁻¹A⁻¹
        - (Aᵀ)⁻¹ = (A⁻¹)ᵀ
        - det(A⁻¹) = 1/det(A)
        
        Complexity: O(n³)
        """
        if not self.is_square():
            raise ValueError("Only square matrices can be inverted")
        
        # Use LU decomposition from the lu module
        try:
            from ..linalg.lu import matrix_inverse_lu
            return matrix_inverse_lu(self)
        except ImportError:
            # Fallback for standalone usage
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'linalg'))
            from linalg.lu import matrix_inverse_lu
            return matrix_inverse_lu(self)
    
    def pseudo_inverse(self) -> 'Matrix':
        """
        Compute the Moore-Penrose pseudo-inverse A⁺.
        
        Mathematical foundation:
        The pseudo-inverse generalizes the concept of matrix inverse to
        non-square and singular matrices. For any matrix A ∈ R^(m×n),
        the pseudo-inverse A⁺ ∈ R^(n×m) satisfies:
        
        1. AA⁺A = A
        2. A⁺AA⁺ = A⁺
        3. (AA⁺)ᵀ = AA⁺
        4. (A⁺A)ᵀ = A⁺A
        
        Computation via normal equations:
        - If m ≥ n and rank(A) = n: A⁺ = (AᵀA)⁻¹Aᵀ
        - If m < n and rank(A) = m: A⁺ = Aᵀ(AAᵀ)⁻¹
        
        Applications:
        - Solving overdetermined systems (least squares)
        - Solving underdetermined systems (minimum norm solution)
        """
        m, n = self.shape
        AT = self.transpose()
        
        try:
            if m >= n:
                # Overdetermined or square case: A⁺ = (AᵀA)⁻¹Aᵀ
                ATA = AT * self
                ATA_inv = ATA.inverse()
                return ATA_inv * AT
            else:
                # Underdetermined case: A⁺ = Aᵀ(AAᵀ)⁻¹
                AAT = self * AT
                AAT_inv = AAT.inverse()
                return AT * AAT_inv
        except ValueError:
            # Fallback: Use SVD-based computation (simplified version)
            # For a complete implementation, this would use singular value decomposition
            raise NotImplementedError("Pseudo-inverse for singular matrices requires SVD implementation")
    
    # ==================== MATRIX ANALYSIS ====================
    
    def rank(self, tolerance: float = 1e-10) -> int:
        """
        Compute the rank of the matrix using row reduction.
        
        Mathematical foundation:
        The rank of a matrix is the dimension of its column space (or row space).
        It represents the number of linearly independent columns (or rows).
        
        Properties:
        - rank(A) ≤ min(m, n) for A ∈ R^(m×n)
        - rank(A) = rank(Aᵀ)
        - rank(AB) ≤ min(rank(A), rank(B))
        - A is invertible ⟺ rank(A) = n (for square n×n matrix)
        
        This implementation uses Gaussian elimination to count pivot positions.
        """
        # Create a copy for row reduction
        temp = self.copy()
        m, n = temp.shape
        
        rank = 0
        col = 0
        
        for row in range(m):
            if col >= n:
                break
                
            # Find pivot
            pivot_row = row
            for i in range(row + 1, m):
                if abs(temp._data[i][col]) > abs(temp._data[pivot_row][col]):
                    pivot_row = i
            
            if abs(temp._data[pivot_row][col]) < tolerance:
                col += 1
                continue
            
            # Swap rows
            if pivot_row != row:
                temp._data[row], temp._data[pivot_row] = temp._data[pivot_row], temp._data[row]
            
            # Eliminate column
            for i in range(row + 1, m):
                if abs(temp._data[i][col]) > tolerance:
                    factor = temp._data[i][col] / temp._data[row][col]
                    for j in range(col, n):
                        temp._data[i][j] -= factor * temp._data[row][j]
            
            rank += 1
            col += 1
        
        return rank
    
    def condition_number(self) -> float:
        """
        Estimate the condition number κ(A) = ||A|| × ||A⁻¹||.
        
        Mathematical foundation:
        The condition number measures how sensitive the solution of Ax = b
        is to perturbations in A or b. A large condition number indicates
        numerical instability.
        
        Interpretation:
        - κ(A) = 1: Perfect conditioning (orthogonal matrices)
        - κ(A) < 10³: Well-conditioned
        - κ(A) > 10¹²: Ill-conditioned (near singular)
        
        This uses spectral norm estimates for efficiency.
        """
        if not self.is_square():
            raise ValueError("Condition number is only defined for square matrices")
        
        try:
            norm_A = self.spectral_norm_estimate()
            A_inv = self.inverse()
            norm_A_inv = A_inv.spectral_norm_estimate()
            return norm_A * norm_A_inv
        except ValueError:
            # Matrix is singular
            return float('inf')
    
    def null_space_dimension(self, tolerance: float = 1e-10) -> int:
        """
        Compute the dimension of the null space (nullity) of the matrix.
        
        Mathematical foundation:
        The null space (kernel) of A is the set {x : Ax = 0}.
        By the rank-nullity theorem: rank(A) + nullity(A) = n
        where n is the number of columns.
        
        The nullity represents the number of degrees of freedom in the
        homogeneous system Ax = 0.
        """
        return self._cols - self.rank(tolerance)
    
    # ==================== EIGENVALUE METHODS (Power Iteration) ====================
    
    # ==================== UTILITY METHODS ====================
    
    def copy(self) -> 'Matrix':
        """Create a deep copy of the matrix."""
        return Matrix([row.copy() for row in self._data])
    
    def resize(self, new_rows: int, new_cols: int, fill_value: float = 0.0) -> 'Matrix':
        """
        Create a new matrix with different dimensions, padding with fill_value or truncating as needed.
        
        This operation preserves existing elements where possible and fills
        new positions with the specified value.
        """
        if new_rows < 1 or new_cols < 1:
            raise ValueError("New dimensions must be positive")
        
        new_data = [[fill_value] * new_cols for _ in range(new_rows)]
        
        # Copy existing elements
        for i in range(min(self._rows, new_rows)):
            for j in range(min(self._cols, new_cols)):
                new_data[i][j] = self._data[i][j]
        
        return Matrix(new_data)
    
    def submatrix(self, row_start: int, row_end: int, col_start: int, col_end: int) -> 'Matrix':
        """
        Extract a submatrix from the given row and column ranges.
        
        Args:
            row_start, row_end: Row range [row_start, row_end)
            col_start, col_end: Column range [col_start, col_end)
        """
        if not (0 <= row_start < row_end <= self._rows):
            raise ValueError("Invalid row range")
        if not (0 <= col_start < col_end <= self._cols):
            raise ValueError("Invalid column range")
        
        subdata = []
        for i in range(row_start, row_end):
            subdata.append(self._data[i][col_start:col_end])
        
        return Matrix(subdata)
    
    def to_list(self) -> List[List[float]]:
        """Convert matrix to list of lists."""
        return [row.copy() for row in self._data]
    
    def flatten(self) -> Vector:
        """Flatten the matrix to a vector in row-major order."""
        flat_data = []
        for row in self._data:
            flat_data.extend(row)
        return Vector(flat_data)
    
    def apply_elementwise(self, func: Callable[[float], float]) -> 'Matrix':
        """
        Apply a function element-wise to the matrix.
        
        This creates a new matrix where each element aᵢⱼ is replaced by func(aᵢⱼ).
        Useful for operations like exp(A), sin(A), etc.
        """
        new_data = [[func(elem) for elem in row] for row in self._data]
        return Matrix(new_data)
    
    def sum(self) -> float:
        """Compute the sum of all elements."""
        return sum(sum(row) for row in self._data)
    
    def mean(self) -> float:
        """Compute the mean of all elements."""
        return self.sum() / self.size
    
    def min(self) -> float:
        """Find the minimum element."""
        return min(min(row) for row in self._data)
    
    def max(self) -> float:
        """Find the maximum element."""
        return max(max(row) for row in self._data)
    
    def argmin(self) -> Tuple[int, int]:
        """Find the indices of the minimum element."""
        min_val = self.min()
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j] - min_val) < 1e-15:
                    return (i, j)
    
    def argmax(self) -> Tuple[int, int]:
        """Find the indices of the maximum element."""
        max_val = self.max()
        for i in range(self._rows):
            for j in range(self._cols):
                if abs(self._data[i][j] - max_val) < 1e-15:
                    return (i, j)

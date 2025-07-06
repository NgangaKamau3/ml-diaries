"""
Linear Algebra from Scratch: Vector Operations
===========================================

This module implements fundamental vector operations using pure Python,
following the mathematical rigor of Deisenroth, Faisal & Ong's "Mathematics for Machine Learning".

A vector in R^n is an ordered list of n real numbers, which can be interpreted
geometrically as a directed line segment from the origin to a point in n-dimensional space,
or algebraically as an element of a vector space with defined operations.

Author: Ng'ang'a Kamau
License: MIT
"""

import math
from typing import List, Union, Tuple, Optional
from numbers import Number


class Vector:
    """
    A vector in R^n implemented as a pure Python class.
    
    Mathematical Foundation:
    A vector v ∈ R^n is an ordered n-tuple of real numbers v = (v₁, v₂, ..., vₙ)
    where each vᵢ ∈ R is called a component or coordinate of the vector.
    
    The vector space R^n is equipped with:
    1. Vector addition: (u + v)ᵢ = uᵢ + vᵢ
    2. Scalar multiplication: (αv)ᵢ = α · vᵢ
    3. Zero vector: 0 = (0, 0, ..., 0)
    4. Additive inverse: -v = (-v₁, -v₂, ..., -vₙ)
    """
    
    def __init__(self, components: List[Union[int, float]]):
        """
        Initialize a vector with given components.
        
        Args:
            components: List of real numbers representing vector coordinates
            
        Raises:
            ValueError: If components list is empty
            TypeError: If components contain non-numeric values
        """
        if not components:
            raise ValueError("Vector must have at least one component")
        
        # Validate that all components are numeric
        for i, comp in enumerate(components):
            if not isinstance(comp, Number):
                raise TypeError(f"Component {i} must be numeric, got {type(comp)}")
        
        # Store as list of floats for consistent arithmetic
        self._components = [float(c) for c in components]
        self._dimension = len(components)
    
    @property
    def components(self) -> List[float]:
        """Return a copy of the vector components."""
        return self._components.copy()
    
    @property
    def dimension(self) -> int:
        """Return the dimension (number of components) of the vector."""
        return self._dimension
    
    def __getitem__(self, index: int) -> float:
        """Access vector component by index (0-based)."""
        return self._components[index]
    
    def __setitem__(self, index: int, value: Union[int, float]):
        """Set vector component by index."""
        if not isinstance(value, Number):
            raise TypeError("Component must be numeric")
        self._components[index] = float(value)
    
    def __len__(self) -> int:
        """Return the dimension of the vector."""
        return self._dimension
    
    def __str__(self) -> str:
        """String representation of vector."""
        comp_str = ', '.join(f'{c:.6g}' for c in self._components)
        return f"Vector([{comp_str}])"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Vector({self._components})"
    
    def __eq__(self, other) -> bool:
        """
        Test vector equality with numerical tolerance.
        
        Two vectors are equal if they have the same dimension and
        their components are equal within floating-point precision.
        """
        if not isinstance(other, Vector):
            return False
        if self.dimension != other.dimension:
            return False
        
        tolerance = 1e-10
        return all(abs(a - b) < tolerance for a, b in zip(self._components, other._components))
    
    # ==================== VECTOR SPACE OPERATIONS ====================
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Vector addition: u + v = (u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ)
        
        Mathematical property: Vector addition is commutative and associative.
        - Commutative: u + v = v + u
        - Associative: (u + v) + w = u + (v + w)
        - Identity: v + 0 = v
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only add Vector to Vector")
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add vectors of different dimensions: {self.dimension} vs {other.dimension}")
        
        result_components = [a + b for a, b in zip(self._components, other._components)]
        return Vector(result_components)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Vector subtraction: u - v = u + (-v)
        
        Geometrically, u - v represents the vector from the tip of v to the tip of u.
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only subtract Vector from Vector")
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract vectors of different dimensions: {self.dimension} vs {other.dimension}")
        
        result_components = [a - b for a, b in zip(self._components, other._components)]
        return Vector(result_components)
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """
        Scalar multiplication: αv = (αv₁, αv₂, ..., αvₙ)
        
        Geometric interpretation: Scalar multiplication scales the vector's magnitude
        and may reverse its direction (if scalar < 0).
        """
        if not isinstance(scalar, Number):
            raise TypeError("Can only multiply vector by scalar")
        
        result_components = [scalar * c for c in self._components]
        return Vector(result_components)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Right multiplication: scalar * vector"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """Vector division by scalar: v/α = (1/α)v"""
        if not isinstance(scalar, Number):
            raise TypeError("Can only divide vector by scalar")
        if abs(scalar) < 1e-15:
            raise ValueError("Cannot divide by zero or near-zero scalar")
        
        return self * (1.0 / scalar)
    
    def __neg__(self) -> 'Vector':
        """Additive inverse: -v = (-v₁, -v₂, ..., -vₙ)"""
        return Vector([-c for c in self._components])
    
    # ==================== NORMS AND DISTANCES ====================
    
    def norm(self, p: Union[int, float, str] = 2) -> float:
        """
        Compute the p-norm of the vector.
        
        The p-norm is defined as: ||v||_p = (∑|vᵢ|^p)^(1/p)
        
        Special cases:
        - p=1: Manhattan norm ||v||₁ = ∑|vᵢ|
        - p=2: Euclidean norm ||v||₂ = √(∑vᵢ²) (default)
        - p=∞: Maximum norm ||v||_∞ = max|vᵢ|
        
        Mathematical properties:
        1. Non-negativity: ||v|| ≥ 0, with equality iff v = 0
        2. Homogeneity: ||αv|| = |α|||v||
        3. Triangle inequality: ||u + v|| ≤ ||u|| + ||v||
        """
        if p == 1:
            # L1 norm (Manhattan/Taxicab norm)
            return sum(abs(c) for c in self._components)
        
        elif p == 2:
            # L2 norm (Euclidean norm) - most common
            return math.sqrt(sum(c * c for c in self._components))
        
        elif p == float('inf') or p == 'inf':
            # L∞ norm (Maximum/Chebyshev norm)
            return max(abs(c) for c in self._components)
        
        elif isinstance(p, Number) and p > 0:
            # General Lp norm
            return (sum(abs(c) ** p for c in self._components)) ** (1.0 / p)
        
        else:
            raise ValueError(f"Invalid norm parameter: {p}. Must be positive number, 'inf', or string 'inf'")
    
    def magnitude(self) -> float:
        """
        Compute the Euclidean magnitude (L2 norm) of the vector.
        
        This is the most common notion of vector "length" in Euclidean space.
        Geometrically, it represents the distance from the origin to the point
        represented by the vector.
        """
        return self.norm(2)
    
    def squared_magnitude(self) -> float:
        """
        Compute the squared Euclidean magnitude: ||v||₂²
        
        This is computationally cheaper than magnitude() and often sufficient
        for comparisons (since x² is monotonic for x ≥ 0).
        """
        return sum(c * c for c in self._components)
    
    def distance(self, other: 'Vector', p: Union[int, float, str] = 2) -> float:
        """
        Compute the p-distance between two vectors.
        
        The distance is defined as d_p(u,v) = ||u - v||_p
        
        This satisfies the metric axioms:
        1. Non-negativity: d(u,v) ≥ 0
        2. Identity: d(u,v) = 0 iff u = v
        3. Symmetry: d(u,v) = d(v,u)
        4. Triangle inequality: d(u,w) ≤ d(u,v) + d(v,w)
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only compute distance to another Vector")
        
        return (self - other).norm(p)
    
    def is_zero(self, tolerance: float = 1e-10) -> bool:
        """
        Check if this is the zero vector within numerical tolerance.
        
        The zero vector is the additive identity of the vector space.
        """
        return self.norm(2) < tolerance
    
    def is_unit(self, tolerance: float = 1e-10) -> bool:
        """
        Check if this is a unit vector (magnitude = 1) within tolerance.
        
        Unit vectors are important for representing directions without magnitude.
        """
        return abs(self.magnitude() - 1.0) < tolerance
    
    # ==================== NORMALIZATION ====================
    
    def normalize(self) -> 'Vector':
        """
        Return a unit vector in the same direction as this vector.
        
        The normalized vector û = v/||v|| has the property that ||û|| = 1
        and û points in the same direction as v.
        
        Raises:
            ValueError: If attempting to normalize the zero vector
        """
        mag = self.magnitude()
        if mag < 1e-15:
            raise ValueError("Cannot normalize zero vector")
        
        return self / mag
    
    def normalize_inplace(self) -> None:
        """
        Normalize this vector in place.
        
        Modifies the current vector to have unit magnitude.
        """
        normalized = self.normalize()
        self._components = normalized._components
    
    # ==================== DOT PRODUCT AND ANGLES ====================
    
    def dot(self, other: 'Vector') -> float:
        """
        Compute the dot product (inner product) of two vectors.
        
        Definition: u · v = ∑(uᵢvᵢ) = u₁v₁ + u₂v₂ + ... + uₙvₙ
        
        Geometric interpretation: u · v = ||u|| ||v|| cos(θ)
        where θ is the angle between the vectors.
        
        Properties:
        1. Commutative: u · v = v · u
        2. Distributive: u · (v + w) = u · v + u · w
        3. Associative with scalars: (αu) · v = α(u · v)
        4. Positive definite: v · v ≥ 0, with equality iff v = 0
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only compute dot product with another Vector")
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compute dot product of vectors with different dimensions: {self.dimension} vs {other.dimension}")
        
        return sum(a * b for a, b in zip(self._components, other._components))
    
    def angle_with(self, other: 'Vector', degrees: bool = False) -> float:
        """
        Compute the angle between two vectors using the dot product formula.
        
        Formula: cos(θ) = (u · v) / (||u|| ||v||)
        Therefore: θ = arccos((u · v) / (||u|| ||v||))
        
        Args:
            other: The other vector
            degrees: If True, return angle in degrees; otherwise radians
            
        Returns:
            Angle between vectors in [0, π] radians or [0, 180] degrees
            
        Raises:
            ValueError: If either vector is zero (angle undefined)
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only compute angle with another Vector")
        
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        
        if mag_self < 1e-15 or mag_other < 1e-15:
            raise ValueError("Cannot compute angle with zero vector")
        
        # Compute cosine of angle
        cos_theta = self.dot(other) / (mag_self * mag_other)
        
        # Clamp to [-1, 1] to handle numerical errors
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        # Compute angle in radians
        angle_rad = math.acos(cos_theta)
        
        return math.degrees(angle_rad) if degrees else angle_rad
    
    def is_orthogonal(self, other: 'Vector', tolerance: float = 1e-10) -> bool:
        """
        Check if two vectors are orthogonal (perpendicular).
        
        Two vectors are orthogonal if their dot product is zero: u · v = 0
        This means they meet at a 90° angle.
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only check orthogonality with another Vector")
        
        return abs(self.dot(other)) < tolerance
    
    def is_parallel(self, other: 'Vector', tolerance: float = 1e-10) -> bool:
        """
        Check if two vectors are parallel (or anti-parallel).
        
        Two vectors are parallel if one is a scalar multiple of the other.
        Equivalently, the sine of the angle between them is zero.
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only check parallelism with another Vector")
        
        # Handle zero vectors
        if self.is_zero(tolerance) or other.is_zero(tolerance):
            return True  # Zero vector is parallel to any vector
        
        try:
            angle = self.angle_with(other)
            # Parallel if angle is 0 or π (within tolerance)
            return abs(math.sin(angle)) < tolerance
        except ValueError:
            return True  # If angle computation fails, assume parallel
    
    # ==================== VECTOR PROJECTIONS ====================
    
    def project_onto(self, other: 'Vector') -> 'Vector':
        """
        Compute the vector projection of self onto other.
        
        The projection of u onto v is: proj_v(u) = ((u · v) / ||v||²) * v
        
        Geometric interpretation: This is the "shadow" of u cast onto the line
        defined by v. It's the component of u that lies in the direction of v.
        
        Properties:
        - proj_v(u) is parallel to v
        - u - proj_v(u) is orthogonal to v
        - ||proj_v(u)|| ≤ ||u|| (projection cannot be longer than original)
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only project onto another Vector")
        if other.is_zero():
            raise ValueError("Cannot project onto zero vector")
        
        # Compute projection coefficient
        coeff = self.dot(other) / other.squared_magnitude()
        
        # Return scaled version of other
        return coeff * other
    
    def project_onto_unit(self, unit_vector: 'Vector') -> 'Vector':
        """
        Compute projection onto a unit vector (optimized version).
        
        For unit vector û: proj_û(u) = (u · û) * û
        This is more efficient since we don't need to divide by ||û||² = 1.
        """
        if not isinstance(unit_vector, Vector):
            raise TypeError("Can only project onto another Vector")
        if not unit_vector.is_unit():
            raise ValueError("Vector must be unit vector")
        
        return self.dot(unit_vector) * unit_vector
    
    def scalar_projection(self, other: 'Vector') -> float:
        """
        Compute the scalar projection (component) of self onto other.
        
        This is the signed length of the vector projection:
        comp_v(u) = (u · v) / ||v|| = ||u|| cos(θ)
        
        Returns:
            Positive if angle < 90°, negative if angle > 90°, zero if orthogonal
        """
        if not isinstance(other, Vector):
            raise TypeError("Can only compute scalar projection onto another Vector")
        if other.is_zero():
            raise ValueError("Cannot compute scalar projection onto zero vector")
        
        return self.dot(other) / other.magnitude()
    
    def reject_from(self, other: 'Vector') -> 'Vector':
        """
        Compute the vector rejection (orthogonal component).
        
        The rejection is: rej_v(u) = u - proj_v(u)
        
        This gives the component of u that is orthogonal to v.
        Together, proj_v(u) + rej_v(u) = u (vector decomposition).
        """
        return self - self.project_onto(other)
    
    # ==================== VECTOR UTILITIES ====================
    
    @staticmethod
    def zero(dimension: int) -> 'Vector':
        """Create a zero vector of specified dimension."""
        if dimension < 1:
            raise ValueError("Dimension must be positive")
        return Vector([0.0] * dimension)
    
    @staticmethod
    def ones(dimension: int) -> 'Vector':
        """Create a vector of ones of specified dimension."""
        if dimension < 1:
            raise ValueError("Dimension must be positive")
        return Vector([1.0] * dimension)
    
    @staticmethod
    def standard_basis(dimension: int, index: int) -> 'Vector':
        """
        Create a standard basis vector (unit vector along one axis).
        
        The standard basis vectors for R^n are:
        e₁ = (1,0,0,...,0), e₂ = (0,1,0,...,0), ..., eₙ = (0,0,0,...,1)
        
        These form an orthonormal basis for R^n.
        """
        if dimension < 1:
            raise ValueError("Dimension must be positive")
        if not (0 <= index < dimension):
            raise ValueError(f"Index {index} out of range for dimension {dimension}")
        
        components = [0.0] * dimension
        components[index] = 1.0
        return Vector(components)
    
    @staticmethod
    def random(dimension: int, low: float = -1.0, high: float = 1.0) -> 'Vector':
        """
        Create a random vector with components uniformly distributed in [low, high].
        
        Note: This uses Python's built-in random module for simplicity.
        For cryptographic applications, use secrets module instead.
        """
        import random
        if dimension < 1:
            raise ValueError("Dimension must be positive")
        
        components = [random.uniform(low, high) for _ in range(dimension)]
        return Vector(components)
    
    def copy(self) -> 'Vector':
        """Create a deep copy of this vector."""
        return Vector(self._components.copy())
    
    def to_list(self) -> List[float]:
        """Convert vector to list of components."""
        return self._components.copy()
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Convert vector to tuple of components."""
        return tuple(self._components)


# ==================== ADDITIONAL VECTOR FUNCTIONS ====================

def linear_combination(vectors: List[Vector], coefficients: List[Union[int, float]]) -> Vector:
    """
    Compute a linear combination of vectors: c₁v₁ + c₂v₂ + ... + cₙvₙ
    
    A linear combination is fundamental to understanding vector spaces,
    spanning sets, and linear independence.
    
    Args:
        vectors: List of vectors to combine
        coefficients: List of scalar coefficients
        
    Returns:
        The resulting vector from the linear combination
    """
    if not vectors:
        raise ValueError("Must provide at least one vector")
    if len(vectors) != len(coefficients):
        raise ValueError("Number of vectors must equal number of coefficients")
    
    # Check all vectors have same dimension
    dim = vectors[0].dimension
    for i, v in enumerate(vectors):
        if not isinstance(v, Vector):
            raise TypeError(f"Element {i} is not a Vector")
        if v.dimension != dim:
            raise ValueError(f"All vectors must have same dimension")
    
    # Compute linear combination
    result = coefficients[0] * vectors[0]
    for coeff, vec in zip(coefficients[1:], vectors[1:]):
        result = result + coeff * vec
    
    return result


def gram_schmidt_process(vectors: List[Vector]) -> List[Vector]:
    """
    Apply the Gram-Schmidt process to orthogonalize a set of vectors.
    
    The Gram-Schmidt process converts a linearly independent set of vectors
    into an orthogonal (or orthonormal) set that spans the same subspace.
    
    Algorithm:
    u₁ = v₁
    u₂ = v₂ - proj_{u₁}(v₂)
    u₃ = v₃ - proj_{u₁}(v₃) - proj_{u₂}(v₃)
    ...
    
    Args:
        vectors: List of linearly independent vectors
        
    Returns:
        List of orthogonal vectors spanning the same subspace
        
    Raises:
        ValueError: If vectors are linearly dependent
    """
    if not vectors:
        return []
    
    # Check all vectors have same dimension
    dim = vectors[0].dimension
    for i, v in enumerate(vectors):
        if not isinstance(v, Vector):
            raise TypeError(f"Element {i} is not a Vector")
        if v.dimension != dim:
            raise ValueError("All vectors must have same dimension")
    
    orthogonal_vectors = []
    
    for i, v in enumerate(vectors):
        # Start with current vector
        u = v.copy()
        
        # Subtract projections onto all previous orthogonal vectors
        for orth_vec in orthogonal_vectors:
            u = u - v.project_onto(orth_vec)
        
        # Check if result is zero (vectors are linearly dependent)
        if u.is_zero():
            raise ValueError(f"Vector {i} is linearly dependent on previous vectors")
        
        orthogonal_vectors.append(u)
    
    return orthogonal_vectors


def gram_schmidt_orthonormal(vectors: List[Vector]) -> List[Vector]:
    """
    Apply Gram-Schmidt process and normalize to get orthonormal vectors.
    
    Returns an orthonormal basis (all vectors are orthogonal and unit length).
    """
    orthogonal = gram_schmidt_process(vectors)
    return [v.normalize() for v in orthogonal]


def are_linearly_independent(vectors: List[Vector], tolerance: float = 1e-10) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Vectors v₁, v₂, ..., vₙ are linearly independent if the only solution
    to c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 is c₁ = c₂ = ... = cₙ = 0.
    
    This implementation uses the Gram-Schmidt process as a test.
    """
    if not vectors:
        return True  # Empty set is linearly independent by convention
    
    try:
        gram_schmidt_process(vectors)
        return True
    except ValueError:
        return False


def vector_triple_product(a: Vector, b: Vector, c: Vector) -> Vector:
    """
    Compute the vector triple product: a × (b × c) = b(a·c) - c(a·b)
    
    Note: This is only defined for 3D vectors. The cross product is not
    defined in this basic implementation, but the triple product identity
    can be computed using dot products.
    
    Only works for 3D vectors.
    """
    if not all(isinstance(v, Vector) for v in [a, b, c]):
        raise TypeError("All arguments must be Vector objects")
    if not all(v.dimension == 3 for v in [a, b, c]):
        raise ValueError("Vector triple product only defined for 3D vectors")
    
    # a × (b × c) = b(a·c) - c(a·b)
    return b * a.dot(c) - c * a.dot(b)


# ==================== EXAMPLE USAGE AND DEMONSTRATIONS ====================

def demonstrate_vector_operations():
    """
    Demonstrate key vector operations with educational examples.
    This function showcases the mathematical concepts implemented above.
    """
    print("=" * 60)
    print("LINEAR ALGEBRA FROM SCRATCH: VECTOR OPERATIONS DEMO")
    print("=" * 60)
    
    # Create some example vectors
    print("\n1. VECTOR CREATION AND BASIC PROPERTIES")
    u = Vector([3, 4])
    v = Vector([1, 2])
    w = Vector([3, 4, 5])
    
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"w = {w} (3D vector)")
    print(f"u dimension: {u.dimension}")
    print(f"u magnitude: {u.magnitude():.6f}")
    print(f"u is unit vector: {u.is_unit()}")
    
    # Vector space operations
    print("\n2. VECTOR SPACE OPERATIONS")
    print(f"u + v = {u + v}")
    print(f"u - v = {u - v}")
    print(f"2 * u = {2 * u}")
    print(f"u / 2 = {u / 2}")
    print(f"-u = {-u}")
    
    # Norms and distances
    print("\n3. NORMS AND DISTANCES")
    print(f"||u||₁ (L1 norm): {u.norm(1):.6f}")
    print(f"||u||₂ (L2 norm): {u.norm(2):.6f}")
    print(f"||u||∞ (L∞ norm): {u.norm('inf'):.6f}")
    print(f"Distance between u and v: {u.distance(v):.6f}")
    
    # Dot products and angles
    print("\n4. DOT PRODUCTS AND ANGLES")
    print(f"u · v = {u.dot(v):.6f}")
    print(f"Angle between u and v: {u.angle_with(v, degrees=True):.2f}°")
    print(f"u and v orthogonal: {u.is_orthogonal(v)}")
    print(f"u and v parallel: {u.is_parallel(v)}")
    
    # Projections
    print("\n5. VECTOR PROJECTIONS")
    proj_v_u = u.project_onto(v)
    print(f"Projection of u onto v: {proj_v_u}")
    print(f"Scalar projection of u onto v: {u.scalar_projection(v):.6f}")
    rejection = u.reject_from(v)
    print(f"Rejection of u from v: {rejection}")
    
    # Verify projection properties
    print(f"Verification: proj + rej = u? {proj_v_u + rejection == u}")
    print(f"Rejection orthogonal to v? {rejection.is_orthogonal(v)}")
    
    # Normalization
    print("\n6. NORMALIZATION")
    u_hat = u.normalize()
    print(f"Unit vector in direction of u: {u_hat}")
    print(f"Magnitude of normalized u: {u_hat.magnitude():.10f}")
    
    # Special vectors
    print("\n7. SPECIAL VECTORS")
    zero_3d = Vector.zero(3)
    ones_3d = Vector.ones(3)
    e1 = Vector.standard_basis(3, 0)
    e2 = Vector.standard_basis(3, 1)
    print(f"Zero vector (3D): {zero_3d}")
    print(f"Ones vector (3D): {ones_3d}")
    print(f"Standard basis e₁: {e1}")
    print(f"Standard basis e₂: {e2}")
    print(f"e₁ · e₂ = {e1.dot(e2)} (orthogonal)")
    
    # Linear combinations
    print("\n8. LINEAR COMBINATIONS")
    lc = linear_combination([u, v], [2, -1])
    print(f"2u - v = {lc}")
    
    # Gram-Schmidt example
    print("\n9. GRAM-SCHMIDT ORTHOGONALIZATION")
    try:
        # Create some vectors that are not orthogonal
        v1 = Vector([1, 1, 0])
        v2 = Vector([1, 0, 1])
        v3 = Vector([0, 1, 1])
        
        print(f"Original vectors:")
        print(f"v₁ = {v1}")
        print(f"v₂ = {v2}")
        print(f"v₃ = {v3}")
        
        orthogonal = gram_schmidt_process([v1, v2, v3])
        print(f"\nAfter Gram-Schmidt:")
        for i, orth_vec in enumerate(orthogonal):
            print(f"u₁{i+1} = {orth_vec}")
        
        # Verify orthogonality
        print(f"\nOrthogonality check:")
        print(f"u₁ · u₂ = {orthogonal[0].dot(orthogonal[1]):.10f}")
        print(f"u₁ · u₃ = {orthogonal[0].dot(orthogonal[2]):.10f}")
        print(f"u₂ · u₃ = {orthogonal[1].dot(orthogonal[2]):.10f}")
        
        # Create orthonormal basis
        orthonormal = gram_schmidt_orthonormal([v1, v2, v3])
        print(f"\nOrthonormal basis:")
        for i, unit_vec in enumerate(orthonormal):
            print(f"ê₁{i+1} = {unit_vec} (magnitude: {unit_vec.magnitude():.10f})")
        
    except ValueError as e:
        print(f"Error in Gram-Schmidt: {e}")
    
    print("\n" + "=" * 60)
    print("END OF VECTOR OPERATIONS DEMONSTRATION")
    print("=" * 60)


# ==================== ADVANCED VECTOR CONCEPTS ====================

class VectorSpace:
    """
    A mathematical vector space V over field F (typically real numbers R).
    
    This class represents abstract vector space concepts and provides
    utilities for working with subspaces, bases, and linear transformations.
    
    Mathematical Definition:
    A vector space V over field F is a set equipped with two operations:
    1. Vector addition: V × V → V
    2. Scalar multiplication: F × V → V
    
    Satisfying eight axioms:
    A1. Associativity of addition: (u + v) + w = u + (v + w)
    A2. Commutativity of addition: u + v = v + u
    A3. Identity element: ∃ 0 ∈ V such that v + 0 = v
    A4. Inverse elements: ∀v ∈ V, ∃(-v) such that v + (-v) = 0
    M1. Compatibility: a(bv) = (ab)v
    M2. Identity element: 1v = v
    D1. Distributivity over vector addition: a(u + v) = au + av
    D2. Distributivity over scalar addition: (a + b)v = av + bv
    """
    
    def __init__(self, dimension: int, field: str = "real"):
        """
        Initialize a vector space.
        
        Args:
            dimension: Dimension of the vector space
            field: Field over which the vector space is defined ("real" or "complex")
        """
        if dimension < 0:
            raise ValueError("Dimension must be non-negative")
        if field not in ["real", "complex"]:
            raise ValueError("Field must be 'real' or 'complex'")
        
        self.dimension = dimension
        self.field = field
    
    def standard_basis(self) -> List[Vector]:
        """
        Return the standard basis for this vector space.
        
        The standard basis for R^n consists of n vectors:
        e₁ = (1,0,0,...,0), e₂ = (0,1,0,...,0), ..., eₙ = (0,0,0,...,1)
        
        These vectors are:
        1. Linearly independent
        2. Orthonormal (orthogonal and unit length)
        3. Span the entire space
        """
        return [Vector.standard_basis(self.dimension, i) for i in range(self.dimension)]
    
    def contains_vector(self, vector: Vector) -> bool:
        """Check if a vector belongs to this vector space."""
        return isinstance(vector, Vector) and vector.dimension == self.dimension
    
    def random_vector(self, low: float = -1.0, high: float = 1.0) -> Vector:
        """Generate a random vector from this space."""
        return Vector.random(self.dimension, low, high)


def span(vectors: List[Vector]) -> List[Vector]:
    """
    Compute a basis for the span of the given vectors.
    
    The span of vectors {v₁, v₂, ..., vₖ} is the set of all linear combinations:
    span{v₁, v₂, ..., vₖ} = {c₁v₁ + c₂v₂ + ... + cₖvₖ : cᵢ ∈ R}
    
    This function returns a basis for the subspace spanned by the input vectors,
    effectively removing redundant (linearly dependent) vectors.
    
    Returns:
        A list of linearly independent vectors that span the same subspace
    """
    if not vectors:
        return []
    
    # Use Gram-Schmidt to find linearly independent vectors
    basis = []
    for v in vectors:
        try:
            # Try to add this vector to the current basis
            temp_basis = basis + [v]
            gram_schmidt_process(temp_basis)  # This will raise ValueError if dependent
            basis.append(v)
        except ValueError:
            # Vector is linearly dependent, skip it
            continue
    
    return basis


def subspace_intersection(basis1: List[Vector], basis2: List[Vector]) -> List[Vector]:
    """
    Find a basis for the intersection of two subspaces.
    
    This is a more advanced operation that requires solving a system of
    linear equations. For now, we provide a simplified implementation
    that works for small examples.
    
    Mathematical background:
    If U and V are subspaces of R^n, then U ∩ V is also a subspace.
    The intersection consists of all vectors that belong to both U and V.
    """
    # This is a simplified implementation
    # A full implementation would use matrix methods from the matrix module
    
    if not basis1 or not basis2:
        return []
    
    # Check if any vectors from basis1 can be expressed as linear combinations of basis2
    intersection_vectors = []
    
    for v1 in basis1:
        # Check if v1 is in the span of basis2
        # This is a simplified check - a full implementation would solve the system
        try:
            # If we can orthogonalize {basis2 + [v1]} without v1 becoming zero,
            # then v1 is not in span(basis2)
            test_vectors = basis2 + [v1]
            result = gram_schmidt_process(test_vectors)
            
            # If the last vector (corresponding to v1) is very small,
            # then v1 was approximately in the span of basis2
            if result[-1].magnitude() < 1e-10:
                intersection_vectors.append(v1)
                
        except ValueError:
            # v1 is exactly in the span of basis2
            intersection_vectors.append(v1)
    
    return intersection_vectors


# ==================== GEOMETRIC INTERPRETATIONS ====================

def compute_area_parallelogram(u: Vector, v: Vector) -> float:
    """
    Compute the area of the parallelogram spanned by two 2D vectors.
    
    For 2D vectors u = (u₁, u₂) and v = (v₁, v₂), the area is:
    Area = |u₁v₂ - u₂v₁| = |det([u v])|
    
    This is the absolute value of the 2D cross product.
    """
    if not isinstance(u, Vector) or not isinstance(v, Vector):
        raise TypeError("Both arguments must be Vector objects")
    if u.dimension != 2 or v.dimension != 2:
        raise ValueError("Both vectors must be 2-dimensional")
    
    # Area = |u₁v₂ - u₂v₁|
    return abs(u[0] * v[1] - u[1] * v[0])


def compute_volume_parallelepiped(u: Vector, v: Vector, w: Vector) -> float:
    """
    Compute the volume of the parallelepiped spanned by three 3D vectors.
    
    For 3D vectors, the volume is the absolute value of the scalar triple product:
    Volume = |u · (v × w)| = |det([u v w])|
    
    Since we don't implement cross product here, we use the determinant formula.
    """
    if not all(isinstance(vec, Vector) for vec in [u, v, w]):
        raise TypeError("All arguments must be Vector objects")
    if not all(vec.dimension == 3 for vec in [u, v, w]):
        raise ValueError("All vectors must be 3-dimensional")
    
    # Volume = |det([[u₁, v₁, w₁], [u₂, v₂, w₂], [u₃, v₃, w₃]])|
    # Expanding the 3×3 determinant:
    det = (u[0] * (v[1] * w[2] - v[2] * w[1]) - 
           u[1] * (v[0] * w[2] - v[2] * w[0]) + 
           u[2] * (v[0] * w[1] - v[1] * w[0]))
    
    return abs(det)


def closest_point_to_line(point: Vector, line_point: Vector, line_direction: Vector) -> Vector:
    """
    Find the closest point on a line to a given point.
    
    Given:
    - P: a point not on the line
    - A: a point on the line
    - d: direction vector of the line
    
    The line can be parameterized as: L(t) = A + td
    The closest point occurs when (P - L(t)) ⊥ d
    
    Solution: t* = (P - A) · d / ||d||²
    Closest point: A + t*d
    """
    if not all(isinstance(v, Vector) for v in [point, line_point, line_direction]):
        raise TypeError("All arguments must be Vector objects")
    if not (point.dimension == line_point.dimension == line_direction.dimension):
        raise ValueError("All vectors must have the same dimension")
    if line_direction.is_zero():
        raise ValueError("Line direction cannot be zero vector")
    
    # Vector from line point to given point
    AP = point - line_point
    
    # Parameter t for closest point
    t = AP.dot(line_direction) / line_direction.squared_magnitude()
    
    # Closest point on line
    return line_point + t * line_direction


def distance_point_to_line(point: Vector, line_point: Vector, line_direction: Vector) -> float:
    """
    Compute the distance from a point to a line.
    
    This is the length of the perpendicular from the point to the line.
    """
    closest = closest_point_to_line(point, line_point, line_direction)
    return point.distance(closest)


# ==================== NUMERICAL STABILITY AND ERROR ANALYSIS ====================

def condition_number_dot_product(u: Vector, v: Vector) -> float:
    """
    Estimate the condition number for dot product computation.
    
    The condition number indicates how sensitive the dot product is to
    small perturbations in the input vectors. A large condition number
    suggests that the computation may be numerically unstable.
    
    For dot product u · v, the condition number is approximately:
    κ ≈ (||u|| ||v||) / |u · v|
    
    When vectors are nearly orthogonal, the condition number becomes large.
    """
    if not isinstance(u, Vector) or not isinstance(v, Vector):
        raise TypeError("Both arguments must be Vector objects")
    
    dot_prod = u.dot(v)
    if abs(dot_prod) < 1e-15:
        return float('inf')  # Nearly orthogonal vectors
    
    return (u.magnitude() * v.magnitude()) / abs(dot_prod)


def relative_error(computed: float, exact: float) -> float:
    """
    Compute the relative error between computed and exact values.
    
    Relative error = |computed - exact| / |exact|
    
    This is useful for analyzing numerical accuracy of vector operations.
    """
    if abs(exact) < 1e-15:
        return abs(computed - exact)  # Absolute error when exact is near zero
    return abs(computed - exact) / abs(exact)


# ==================== TESTING UTILITIES ====================

def test_vector_properties(v: Vector) -> dict:
    """
    Test various mathematical properties of a vector for debugging/verification.
    
    Returns a dictionary with test results and computed values.
    """
    results = {
        'dimension': v.dimension,
        'magnitude': v.magnitude(),
        'squared_magnitude': v.squared_magnitude(),
        'is_zero': v.is_zero(),
        'is_unit': v.is_unit(),
        'components_sum': sum(v.components),
        'components_max': max(abs(c) for c in v.components),
        'l1_norm': v.norm(1),
        'l2_norm': v.norm(2),
        'linf_norm': v.norm('inf'),
    }
    
    # Test mathematical relationships
    results['magnitude_consistency'] = abs(results['magnitude']**2 - results['squared_magnitude']) < 1e-10
    results['norm_consistency'] = abs(results['l2_norm'] - results['magnitude']) < 1e-10
    
    return results


def verify_orthogonality(vectors: List[Vector], tolerance: float = 1e-10) -> dict:
    """
    Verify that a set of vectors is orthogonal.
    
    Returns detailed information about pairwise dot products and orthogonality.
    """
    n = len(vectors)
    if n < 2:
        return {'orthogonal': True, 'dot_products': [], 'max_dot_product': 0.0}
    
    dot_products = []
    max_dot = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dot_prod = vectors[i].dot(vectors[j])
            dot_products.append((i, j, dot_prod))
            max_dot = max(max_dot, abs(dot_prod))
    
    is_orthogonal = max_dot < tolerance
    
    return {
        'orthogonal': is_orthogonal,
        'dot_products': dot_products,
        'max_dot_product': max_dot,
        'tolerance': tolerance
    }


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demonstrate_vector_operations()
    
    # Additional advanced examples
    print("\n" + "=" * 60)
    print("ADVANCED VECTOR SPACE CONCEPTS")
    print("=" * 60)
    
    # Vector space example
    R3 = VectorSpace(3, "real")
    print(f"\nVector space R³:")
    print(f"Dimension: {R3.dimension}")
    print(f"Field: {R3.field}")
    
    standard_basis = R3.standard_basis()
    print(f"Standard basis: {[str(v) for v in standard_basis]}")
    
    # Test linear independence
    test_vectors = [Vector([1, 2, 3]), Vector([4, 5, 6]), Vector([7, 8, 9])]
    print(f"\nTest vectors: {[str(v) for v in test_vectors]}")
    print(f"Linearly independent: {are_linearly_independent(test_vectors)}")
    
    # Span calculation
    spanning_set = [Vector([1, 0, 0]), Vector([1, 1, 0]), Vector([1, 1, 1]), Vector([2, 1, 0])]
    basis = span(spanning_set)
    print(f"\nSpanning set: {[str(v) for v in spanning_set]}")
    print(f"Basis for span: {[str(v) for v in basis]}")
    
    # Geometric calculations
    print(f"\n" + "=" * 40)
    print("GEOMETRIC APPLICATIONS")
    print("=" * 40)
    
    # Area of parallelogram
    u_2d = Vector([3, 4])
    v_2d = Vector([1, 2])
    area = compute_area_parallelogram(u_2d, v_2d)
    print(f"Vectors: u = {u_2d}, v = {v_2d}")
    print(f"Area of parallelogram: {area}")
    
    # Volume of parallelepiped
    u_3d = Vector([1, 0, 0])
    v_3d = Vector([0, 1, 0])
    w_3d = Vector([0, 0, 1])
    volume = compute_volume_parallelepiped(u_3d, v_3d, w_3d)
    print(f"\nVectors: u = {u_3d}, v = {v_3d}, w = {w_3d}")
    print(f"Volume of parallelepiped: {volume}")
    
    # Point-to-line distance
    point = Vector([1, 1, 1])
    line_pt = Vector([0, 0, 0])
    line_dir = Vector([1, 0, 0])
    distance = distance_point_to_line(point, line_pt, line_dir)
    print(f"\nPoint: {point}")
    print(f"Line through {line_pt} in direction {line_dir}")
    print(f"Distance from point to line: {distance}")
    
    print(f"\n" + "=" * 60)
    print("NUMERICAL ANALYSIS")
    print("=" * 60)
    
    # Condition number analysis
    near_orthogonal_1 = Vector([1, 0.001])
    near_orthogonal_2 = Vector([0.001, 1])
    cond_num = condition_number_dot_product(near_orthogonal_1, near_orthogonal_2)
    print(f"Nearly orthogonal vectors: {near_orthogonal_1}, {near_orthogonal_2}")
    print(f"Condition number for dot product: {cond_num:.2e}")
    
    # Vector property testing
    test_vec = Vector([3, 4, 12])
    properties = test_vector_properties(test_vec)
    print(f"\nVector properties for {test_vec}:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    print(f"\n" + "=" * 60)
    print("VECTOR OPERATIONS MODULE COMPLETE")
    print("=" * 60) 
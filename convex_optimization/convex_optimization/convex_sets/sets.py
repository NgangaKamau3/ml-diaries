"""
Convex Sets Theory
=================

Mathematical foundation for convex sets following Boyd & Vandenberghe.
"""

import math
from typing import List, Callable, Optional, Tuple
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector
from core.matrix import Matrix


class ConvexSet:
    """Abstract base class for convex sets."""
    
    def contains(self, x: Vector) -> bool:
        """Check if point x is in the set."""
        raise NotImplementedError
    
    def project(self, x: Vector) -> Vector:
        """Project point x onto the set."""
        raise NotImplementedError
    
    def is_convex_combination(self, points: List[Vector], weights: List[float]) -> bool:
        """Verify convex combination: Σλᵢ = 1, λᵢ ≥ 0."""
        if len(points) != len(weights):
            return False
        if abs(sum(weights) - 1.0) > 1e-10:
            return False
        return all(w >= -1e-10 for w in weights)


class Hyperplane(ConvexSet):
    """Hyperplane: {x | aᵀx = b}."""
    
    def __init__(self, a: Vector, b: float):
        self.a = a.normalize()  # Normal vector
        self.b = b / a.magnitude()  # Normalized offset
    
    def contains(self, x: Vector, tolerance: float = 1e-10) -> bool:
        return abs(self.a.dot(x) - self.b) < tolerance
    
    def project(self, x: Vector) -> Vector:
        """Project x onto hyperplane."""
        return x - (self.a.dot(x) - self.b) * self.a


class Halfspace(ConvexSet):
    """Halfspace: {x | aᵀx ≤ b}."""
    
    def __init__(self, a: Vector, b: float):
        self.a = a
        self.b = b
    
    def contains(self, x: Vector) -> bool:
        return self.a.dot(x) <= self.b + 1e-10
    
    def project(self, x: Vector) -> Vector:
        """Project x onto halfspace."""
        if self.contains(x):
            return x
        # Project onto boundary hyperplane
        hyperplane = Hyperplane(self.a, self.b)
        return hyperplane.project(x)


class Ball(ConvexSet):
    """Euclidean ball: {x | ||x - center||₂ ≤ radius}."""
    
    def __init__(self, center: Vector, radius: float):
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        self.center = center
        self.radius = radius
    
    def contains(self, x: Vector) -> bool:
        return (x - self.center).magnitude() <= self.radius + 1e-10
    
    def project(self, x: Vector) -> Vector:
        """Project x onto ball."""
        diff = x - self.center
        dist = diff.magnitude()
        
        if dist <= self.radius:
            return x
        
        # Project onto sphere boundary
        return self.center + (self.radius / dist) * diff


class Simplex(ConvexSet):
    """Standard simplex: {x | Σxᵢ = 1, xᵢ ≥ 0}."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    def contains(self, x: Vector, tolerance: float = 1e-10) -> bool:
        if x.dimension != self.dimension:
            return False
        
        # Check non-negativity
        if any(xi < -tolerance for xi in x.components):
            return False
        
        # Check sum equals 1
        return abs(sum(x.components) - 1.0) < tolerance
    
    def project(self, x: Vector) -> Vector:
        """Project onto simplex using Michelot's algorithm."""
        if x.dimension != self.dimension:
            raise ValueError("Dimension mismatch")
        
        # Sort components in descending order
        sorted_indices = sorted(range(self.dimension), 
                              key=lambda i: x[i], reverse=True)
        
        # Find the largest k such that the projection is feasible
        cumsum = 0
        for k in range(self.dimension):
            idx = sorted_indices[k]
            cumsum += x[idx]
            
            # Check if this k works
            theta = (cumsum - 1) / (k + 1)
            if k == self.dimension - 1 or x[sorted_indices[k+1]] <= theta:
                break
        
        # Compute projection
        result = Vector.zero(self.dimension)
        for i in range(self.dimension):
            result._components[i] = max(0, x[i] - theta)
        
        return result


class Polyhedron(ConvexSet):
    """Polyhedron: {x | Ax ≤ b}."""
    
    def __init__(self, A: Matrix, b: Vector):
        if A.rows != b.dimension:
            raise ValueError("Incompatible dimensions")
        self.A = A
        self.b = b
    
    def contains(self, x: Vector) -> bool:
        if x.dimension != self.A.cols:
            return False
        
        Ax = self.A * x
        return all(Ax[i] <= self.b[i] + 1e-10 for i in range(len(self.b.components)))
    
    def project(self, x: Vector) -> Vector:
        """Project onto polyhedron (simplified quadratic programming)."""
        # This is a simplified implementation
        # Full implementation would solve: min ||y-x||² s.t. Ay ≤ b
        
        if self.contains(x):
            return x
        
        # Iterative projection onto violated constraints
        y = x.copy()
        max_iterations = 100
        
        for _ in range(max_iterations):
            Ay = self.A * y
            
            # Find most violated constraint
            max_violation = 0
            worst_constraint = -1
            
            for i in range(len(self.b.components)):
                violation = Ay[i] - self.b[i]
                if violation > max_violation:
                    max_violation = violation
                    worst_constraint = i
            
            if max_violation <= 1e-10:
                break
            
            # Project onto violated constraint
            a_i = self.A.get_row(worst_constraint)
            b_i = self.b[worst_constraint]
            halfspace = Halfspace(a_i, b_i)
            y = halfspace.project(y)
        
        return y


def convex_hull(points: List[Vector]) -> List[Vector]:
    """Compute convex hull vertices (simplified 2D implementation)."""
    if not points:
        return []
    
    if len(points) == 1:
        return points
    
    # For 2D case, use Graham scan
    if points[0].dimension == 2:
        return _graham_scan_2d(points)
    
    # For higher dimensions, return all points (simplified)
    return points


def _graham_scan_2d(points: List[Vector]) -> List[Vector]:
    """Graham scan for 2D convex hull."""
    if len(points) < 3:
        return points
    
    # Find bottom-most point
    start = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort by polar angle
    def polar_angle(p):
        dx, dy = p[0] - start[0], p[1] - start[1]
        return math.atan2(dy, dx)
    
    sorted_points = sorted([p for p in points if p != start], key=polar_angle)
    
    # Build convex hull
    hull = [start, sorted_points[0]]
    
    for p in sorted_points[1:]:
        while len(hull) > 1:
            # Check if we make a right turn
            o1, o2 = hull[-2], hull[-1]
            cross = (o2[0] - o1[0]) * (p[1] - o1[1]) - (o2[1] - o1[1]) * (p[0] - o1[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    
    return hull


def is_convex_set(points: List[Vector], test_points: int = 100) -> bool:
    """Test if a set of points forms a convex set."""
    if len(points) < 2:
        return True
    
    # Generate random convex combinations and check if they're in the set
    import random
    
    for _ in range(test_points):
        # Pick two random points
        p1, p2 = random.sample(points, 2)
        
        # Random convex combination
        t = random.random()
        combo = t * p1 + (1 - t) * p2
        
        # Check if combination is "close" to any point in set
        min_dist = min((combo - p).magnitude() for p in points)
        if min_dist > 1e-6:  # Tolerance for numerical errors
            return False
    
    return True
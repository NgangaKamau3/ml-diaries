"""
Convex Functions Theory
======================

Mathematical foundation for convex functions and their properties.
"""

import math
from typing import Callable, List, Tuple, Optional
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector
from core.matrix import Matrix


class ConvexFunction:
    """Base class for convex functions."""
    
    def __call__(self, x: Vector) -> float:
        """Evaluate function at x."""
        raise NotImplementedError
    
    def gradient(self, x: Vector) -> Vector:
        """Compute gradient at x."""
        raise NotImplementedError
    
    def hessian(self, x: Vector) -> Matrix:
        """Compute Hessian at x."""
        raise NotImplementedError
    
    def is_convex(self, domain_points: List[Vector]) -> bool:
        """Test convexity using Jensen's inequality."""
        for _ in range(50):  # Random tests
            import random
            if len(domain_points) < 2:
                continue
                
            p1, p2 = random.sample(domain_points, 2)
            t = random.random()
            
            # Jensen's inequality: f(tx + (1-t)y) ≤ tf(x) + (1-t)f(y)
            combo_point = t * p1 + (1 - t) * p2
            
            try:
                lhs = self(combo_point)
                rhs = t * self(p1) + (1 - t) * self(p2)
                
                if lhs > rhs + 1e-10:  # Tolerance for numerical errors
                    return False
            except:
                continue
        
        return True
    
    def is_strongly_convex(self, mu: float, domain_points: List[Vector]) -> bool:
        """Test μ-strong convexity."""
        for _ in range(30):
            import random
            if len(domain_points) < 2:
                continue
                
            p1, p2 = random.sample(domain_points, 2)
            t = random.random()
            
            combo_point = t * p1 + (1 - t) * p2
            
            try:
                lhs = self(combo_point)
                rhs = t * self(p1) + (1 - t) * self(p2) - 0.5 * mu * t * (1 - t) * (p1 - p2).squared_magnitude()
                
                if lhs > rhs + 1e-10:
                    return False
            except:
                continue
        
        return True


class QuadraticFunction(ConvexFunction):
    """Quadratic function: f(x) = ½xᵀQx + cᵀx + d."""
    
    def __init__(self, Q: Matrix, c: Vector, d: float = 0.0):
        if not Q.is_square() or Q.rows != c.dimension:
            raise ValueError("Incompatible dimensions")
        
        self.Q = Q
        self.c = c
        self.d = d
        
        # Check positive semidefiniteness for convexity
        self._is_convex = self._check_psd()
    
    def __call__(self, x: Vector) -> float:
        return 0.5 * x.dot(self.Q * x) + self.c.dot(x) + self.d
    
    def gradient(self, x: Vector) -> Vector:
        return self.Q * x + self.c
    
    def hessian(self, x: Vector) -> Matrix:
        return self.Q
    
    def _check_psd(self) -> bool:
        """Check if Q is positive semidefinite."""
        try:
            # Simple test: all eigenvalues ≥ 0
            from eigenvalues.eigendecomposition import EigenDecomposition
            eigen = EigenDecomposition(self.Q)
            eigenvals, _ = eigen.compute()
            return all(val >= -1e-10 for val in eigenvals.components)
        except:
            return False


class NormFunction(ConvexFunction):
    """Norm function: f(x) = ||x||_p."""
    
    def __init__(self, p: float = 2.0):
        if p < 1:
            raise ValueError("p must be ≥ 1 for convexity")
        self.p = p
    
    def __call__(self, x: Vector) -> float:
        return x.norm(self.p)
    
    def gradient(self, x: Vector) -> Vector:
        """Subgradient for non-smooth norms."""
        if self.p == 2:
            norm_x = x.magnitude()
            if norm_x < 1e-15:
                return Vector.zero(x.dimension)
            return x / norm_x
        elif self.p == 1:
            # Subgradient of L1 norm
            result = Vector.zero(x.dimension)
            for i in range(x.dimension):
                if abs(x[i]) > 1e-15:
                    result._components[i] = 1.0 if x[i] > 0 else -1.0
                # else: subgradient is in [-1, 1]
            return result
        else:
            # General Lp norm gradient
            norm_x = x.norm(self.p)
            if norm_x < 1e-15:
                return Vector.zero(x.dimension)
            
            result = Vector.zero(x.dimension)
            for i in range(x.dimension):
                if abs(x[i]) > 1e-15:
                    result._components[i] = (abs(x[i]) ** (self.p - 1)) * (1 if x[i] > 0 else -1)
            
            return result * (norm_x ** (1 - self.p))


class LogSumExp(ConvexFunction):
    """Log-sum-exp function: f(x) = log(Σᵢ exp(xᵢ))."""
    
    def __call__(self, x: Vector) -> float:
        # Numerically stable computation
        max_x = max(x.components)
        return max_x + math.log(sum(math.exp(xi - max_x) for xi in x.components))
    
    def gradient(self, x: Vector) -> Vector:
        max_x = max(x.components)
        exp_shifted = [math.exp(xi - max_x) for xi in x.components]
        sum_exp = sum(exp_shifted)
        
        return Vector([exp_val / sum_exp for exp_val in exp_shifted])
    
    def hessian(self, x: Vector) -> Matrix:
        grad = self.gradient(x)
        n = x.dimension
        
        # Hessian = diag(grad) - grad * gradᵀ
        H = Matrix.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, j] = grad[i] * (1 - grad[i])
                else:
                    H[i, j] = -grad[i] * grad[j]
        
        return H


class SupportFunction(ConvexFunction):
    """Support function of a set: f(x) = sup{yᵀx | y ∈ C}."""
    
    def __init__(self, constraint_set: List[Vector]):
        self.constraint_set = constraint_set
    
    def __call__(self, x: Vector) -> float:
        if not self.constraint_set:
            return 0.0
        
        return max(y.dot(x) for y in self.constraint_set)
    
    def gradient(self, x: Vector) -> Vector:
        """Subgradient is the maximizing y."""
        if not self.constraint_set:
            return Vector.zero(x.dimension)
        
        max_val = float('-inf')
        best_y = self.constraint_set[0]
        
        for y in self.constraint_set:
            val = y.dot(x)
            if val > max_val:
                max_val = val
                best_y = y
        
        return best_y


class IndicatorFunction(ConvexFunction):
    """Indicator function: f(x) = 0 if x ∈ C, +∞ otherwise."""
    
    def __init__(self, constraint_set):
        self.constraint_set = constraint_set
    
    def __call__(self, x: Vector) -> float:
        if hasattr(self.constraint_set, 'contains'):
            return 0.0 if self.constraint_set.contains(x) else float('inf')
        else:
            # Assume it's a list of points
            tolerance = 1e-10
            for point in self.constraint_set:
                if (x - point).magnitude() < tolerance:
                    return 0.0
            return float('inf')


def conjugate_function(f: ConvexFunction, y: Vector, domain_points: List[Vector]) -> float:
    """Compute conjugate function f*(y) = sup{yᵀx - f(x)}."""
    max_val = float('-inf')
    
    for x in domain_points:
        try:
            val = y.dot(x) - f(x)
            max_val = max(max_val, val)
        except:
            continue
    
    return max_val if max_val != float('-inf') else 0.0


def perspective_function(f: ConvexFunction, x: Vector, t: float) -> float:
    """Perspective function: g(x,t) = t·f(x/t) for t > 0."""
    if t <= 0:
        return float('inf')
    
    try:
        return t * f(x / t)
    except:
        return float('inf')


def moreau_envelope(f: ConvexFunction, x: Vector, mu: float, 
                   domain_points: List[Vector]) -> float:
    """Moreau envelope: (f □ μ)(x) = inf{f(y) + (1/2μ)||x-y||²}."""
    if mu <= 0:
        raise ValueError("μ must be positive")
    
    min_val = float('inf')
    
    for y in domain_points:
        try:
            val = f(y) + (1.0 / (2 * mu)) * (x - y).squared_magnitude()
            min_val = min(min_val, val)
        except:
            continue
    
    return min_val if min_val != float('inf') else 0.0


def proximal_operator(f: ConvexFunction, x: Vector, mu: float,
                     domain_points: List[Vector]) -> Vector:
    """Proximal operator: prox_μf(x) = argmin{f(y) + (1/2μ)||x-y||²}."""
    if mu <= 0:
        raise ValueError("μ must be positive")
    
    min_val = float('inf')
    best_y = x
    
    for y in domain_points:
        try:
            val = f(y) + (1.0 / (2 * mu)) * (x - y).squared_magnitude()
            if val < min_val:
                min_val = val
                best_y = y
        except:
            continue
    
    return best_y


def epigraph_projection(f: ConvexFunction, point: Tuple[Vector, float],
                       domain_points: List[Vector]) -> Tuple[Vector, float]:
    """Project point onto epigraph of f."""
    x, t = point
    
    # If point is above epigraph, project down
    try:
        f_x = f(x)
        if t >= f_x:
            return (x, t)  # Already in epigraph
        else:
            return (x, f_x)  # Project onto graph
    except:
        return (x, t)
"""
Optimization Algorithms
======================

Gradient descent, Newton's method, and constrained optimization.
"""

import math
from typing import Callable, Optional, Tuple
from ..core.vector import Vector
from ..core.matrix import Matrix
from .derivatives import gradient_finite_diff, hessian_finite_diff


def gradient_descent(
    f: Callable,
    x0: Vector,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[Vector, List[float]]:
    """Gradient descent optimization."""
    x = x0.copy()
    history = [f(x)]
    
    for _ in range(max_iterations):
        grad = gradient_finite_diff(f, x)
        
        if grad.magnitude() < tolerance:
            break
        
        x = x - learning_rate * grad
        history.append(f(x))
    
    return x, history


def newton_method(
    f: Callable,
    x0: Vector,
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> Tuple[Vector, List[float]]:
    """Newton's method for optimization."""
    x = x0.copy()
    history = [f(x)]
    
    for _ in range(max_iterations):
        grad = gradient_finite_diff(f, x)
        
        if grad.magnitude() < tolerance:
            break
        
        hess = hessian_finite_diff(f, x)
        
        try:
            # Newton step: x_new = x - H⁻¹∇f
            delta = hess.solve(grad)
            x = x - delta
            history.append(f(x))
        except ValueError:
            # Hessian is singular, fall back to gradient descent
            x = x - 0.01 * grad
            history.append(f(x))
    
    return x, history


def line_search_backtrack(
    f: Callable,
    x: Vector,
    direction: Vector,
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4
) -> float:
    """Backtracking line search."""
    grad = gradient_finite_diff(f, x)
    descent_condition = c * grad.dot(direction)
    
    while f(x + alpha * direction) > f(x) + alpha * descent_condition:
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha


def bfgs_method(
    f: Callable,
    x0: Vector,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[Vector, List[float]]:
    """BFGS quasi-Newton method."""
    x = x0.copy()
    n = x.dimension
    B = Matrix.identity(n)  # Approximate Hessian
    history = [f(x)]
    
    grad = gradient_finite_diff(f, x)
    
    for _ in range(max_iterations):
        if grad.magnitude() < tolerance:
            break
        
        # Solve Bp = -grad
        try:
            p = B.solve(-grad)
        except ValueError:
            p = -grad  # Fallback to steepest descent
        
        # Line search
        alpha = line_search_backtrack(f, x, p)
        
        # Update
        x_new = x + alpha * p
        grad_new = gradient_finite_diff(f, x_new)
        
        # BFGS update
        s = x_new - x
        y = grad_new - grad
        
        if s.dot(y) > 1e-10:  # Curvature condition
            # B_new = B + (yy^T)/(y^T s) - (Bss^T B)/(s^T B s)
            rho = 1.0 / y.dot(s)
            I = Matrix.identity(n)
            
            # This is a simplified BFGS update
            # Full implementation would use rank-2 updates
            B = B + rho * Matrix.from_vectors([y], by_columns=True) * Matrix.from_vectors([y], by_columns=False)
        
        x = x_new
        grad = grad_new
        history.append(f(x))
    
    return x, history
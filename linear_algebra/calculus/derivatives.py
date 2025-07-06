"""
Automatic Differentiation and Gradients
=======================================

Basic automatic differentiation for optimization and machine learning.
"""

import math
from typing import Callable, List
from ..core.vector import Vector
from ..core.matrix import Matrix


class DualNumber:
    """Dual number for automatic differentiation."""
    
    def __init__(self, real: float, dual: float = 0.0):
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )
        return DualNumber(self.real * other, self.dual * other)
    
    def __pow__(self, n):
        return DualNumber(
            self.real ** n,
            n * (self.real ** (n-1)) * self.dual
        )
    
    def sin(self):
        return DualNumber(math.sin(self.real), math.cos(self.real) * self.dual)
    
    def cos(self):
        return DualNumber(math.cos(self.real), -math.sin(self.real) * self.dual)
    
    def exp(self):
        exp_val = math.exp(self.real)
        return DualNumber(exp_val, exp_val * self.dual)


def gradient_finite_diff(f: Callable, x: Vector, h: float = 1e-8) -> Vector:
    """Compute gradient using finite differences."""
    n = x.dimension
    grad = Vector.zero(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad._components[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def hessian_finite_diff(f: Callable, x: Vector, h: float = 1e-6) -> Matrix:
    """Compute Hessian using finite differences."""
    n = x.dimension
    H = Matrix.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal element
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                H[i, j] = (f(x_plus) - 2*f(x) + f(x_minus)) / (h**2)
            else:
                # Off-diagonal element
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
    
    return H


def jacobian_finite_diff(f: Callable, x: Vector, h: float = 1e-8) -> Matrix:
    """Compute Jacobian matrix for vector-valued function."""
    n = x.dimension
    f_x = f(x)
    m = len(f_x) if hasattr(f_x, '__len__') else 1
    
    J = Matrix.zeros(m, n)
    
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
        for i in range(m):
            J[i, j] = (f_plus[i] - f_minus[i]) / (2 * h)
    
    return J
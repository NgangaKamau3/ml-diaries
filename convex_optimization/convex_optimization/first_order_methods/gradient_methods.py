"""
First-Order Optimization Methods
===============================

Gradient descent variants with rigorous mathematical analysis.
"""

import math
from typing import Callable, List, Tuple, Optional, Dict
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector
from core.matrix import Matrix


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(self):
        self.x_optimal: Optional[Vector] = None
        self.f_optimal: Optional[float] = None
        self.iterations: int = 0
        self.converged: bool = False
        self.history: List[Tuple[Vector, float]] = []
        self.gradient_norms: List[float] = []


class GradientDescent:
    """Vanilla gradient descent with exact and inexact line search."""
    
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize(self, f: Callable, grad_f: Callable, x0: Vector) -> OptimizationResult:
        """Standard gradient descent: xₖ₊₁ = xₖ - αₖ∇f(xₖ)."""
        result = OptimizationResult()
        x = x0.copy()
        
        for k in range(self.max_iterations):
            f_x = f(x)
            grad_x = grad_f(x)
            grad_norm = grad_x.magnitude()
            
            # Store history
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            # Check convergence
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Gradient step
            x = x - self.step_size * grad_x
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result
    
    def optimize_with_backtracking(self, f: Callable, grad_f: Callable, x0: Vector,
                                  c1: float = 1e-4, rho: float = 0.5) -> OptimizationResult:
        """Gradient descent with Armijo backtracking line search."""
        result = OptimizationResult()
        x = x0.copy()
        
        for k in range(self.max_iterations):
            f_x = f(x)
            grad_x = grad_f(x)
            grad_norm = grad_x.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Backtracking line search
            alpha = self._backtracking_line_search(f, grad_f, x, f_x, grad_x, c1, rho)
            
            # Update
            x = x - alpha * grad_x
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result
    
    def _backtracking_line_search(self, f: Callable, grad_f: Callable, x: Vector,
                                 f_x: float, grad_x: Vector, c1: float, rho: float) -> float:
        """Armijo backtracking: f(x - α∇f) ≤ f(x) - c₁α||∇f||²."""
        alpha = 1.0
        grad_norm_sq = grad_x.squared_magnitude()
        
        while True:
            x_new = x - alpha * grad_x
            f_new = f(x_new)
            
            # Armijo condition
            if f_new <= f_x - c1 * alpha * grad_norm_sq:
                break
            
            alpha *= rho
            
            if alpha < 1e-16:
                break
        
        return alpha


class AcceleratedGradientDescent:
    """Nesterov's accelerated gradient descent."""
    
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize(self, f: Callable, grad_f: Callable, x0: Vector) -> OptimizationResult:
        """Nesterov acceleration: O(1/k²) convergence rate."""
        result = OptimizationResult()
        
        x = x0.copy()
        y = x0.copy()
        t = 1.0
        
        for k in range(self.max_iterations):
            f_x = f(x)
            grad_y = grad_f(y)
            grad_norm = grad_y.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Update x
            x_new = y - self.step_size * grad_y
            
            # Update momentum parameter
            t_new = (1 + math.sqrt(1 + 4 * t**2)) / 2
            
            # Update y with momentum
            beta = (t - 1) / t_new
            y = x_new + beta * (x_new - x)
            
            x = x_new
            t = t_new
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result


class AdaptiveGradientDescent:
    """AdaGrad: Adaptive learning rates."""
    
    def __init__(self, initial_step_size: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, epsilon: float = 1e-8):
        self.initial_step_size = initial_step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
    
    def optimize(self, f: Callable, grad_f: Callable, x0: Vector) -> OptimizationResult:
        """AdaGrad: αₖ = α₀ / √(Σᵢ₌₁ᵏ ||∇f(xᵢ)||²)."""
        result = OptimizationResult()
        x = x0.copy()
        
        # Accumulated squared gradients
        G = Vector.zero(x.dimension)
        
        for k in range(self.max_iterations):
            f_x = f(x)
            grad_x = grad_f(x)
            grad_norm = grad_x.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Update accumulated gradients
            for i in range(x.dimension):
                G._components[i] += grad_x[i] ** 2
            
            # Adaptive step sizes
            adaptive_grad = Vector.zero(x.dimension)
            for i in range(x.dimension):
                adaptive_grad._components[i] = grad_x[i] / (math.sqrt(G[i]) + self.epsilon)
            
            # Update
            x = x - self.initial_step_size * adaptive_grad
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result


class RMSprop:
    """RMSprop: Exponential moving average of squared gradients."""
    
    def __init__(self, step_size: float = 0.01, decay_rate: float = 0.9,
                 max_iterations: int = 1000, tolerance: float = 1e-6, epsilon: float = 1e-8):
        self.step_size = step_size
        self.decay_rate = decay_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
    
    def optimize(self, f: Callable, grad_f: Callable, x0: Vector) -> OptimizationResult:
        """RMSprop: vₖ = γvₖ₋₁ + (1-γ)||∇f||², αₖ = α/√vₖ."""
        result = OptimizationResult()
        x = x0.copy()
        
        # Exponential moving average of squared gradients
        v = Vector.zero(x.dimension)
        
        for k in range(self.max_iterations):
            f_x = f(x)
            grad_x = grad_f(x)
            grad_norm = grad_x.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Update moving average
            for i in range(x.dimension):
                v._components[i] = self.decay_rate * v[i] + (1 - self.decay_rate) * grad_x[i] ** 2
            
            # Adaptive update
            adaptive_grad = Vector.zero(x.dimension)
            for i in range(x.dimension):
                adaptive_grad._components[i] = grad_x[i] / (math.sqrt(v[i]) + self.epsilon)
            
            x = x - self.step_size * adaptive_grad
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result


class Adam:
    """Adam: Adaptive moment estimation."""
    
    def __init__(self, step_size: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 max_iterations: int = 1000, tolerance: float = 1e-6, epsilon: float = 1e-8):
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
    
    def optimize(self, f: Callable, grad_f: Callable, x0: Vector) -> OptimizationResult:
        """Adam optimizer with bias correction."""
        result = OptimizationResult()
        x = x0.copy()
        
        # First and second moment estimates
        m = Vector.zero(x.dimension)
        v = Vector.zero(x.dimension)
        
        for k in range(1, self.max_iterations + 1):
            f_x = f(x)
            grad_x = grad_f(x)
            grad_norm = grad_x.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(grad_norm)
            
            if grad_norm < self.tolerance:
                result.converged = True
                break
            
            # Update biased first moment estimate
            for i in range(x.dimension):
                m._components[i] = self.beta1 * m[i] + (1 - self.beta1) * grad_x[i]
            
            # Update biased second moment estimate
            for i in range(x.dimension):
                v._components[i] = self.beta2 * v[i] + (1 - self.beta2) * grad_x[i] ** 2
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** k)
            v_hat = v / (1 - self.beta2 ** k)
            
            # Update
            adaptive_grad = Vector.zero(x.dimension)
            for i in range(x.dimension):
                adaptive_grad._components[i] = m_hat[i] / (math.sqrt(v_hat[i]) + self.epsilon)
            
            x = x - self.step_size * adaptive_grad
        
        result.x_optimal = x
        result.f_optimal = f(x)
        result.iterations = len(result.history)
        
        return result


class SubgradientMethod:
    """Subgradient method for non-smooth convex optimization."""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize(self, f: Callable, subgrad_f: Callable, x0: Vector,
                step_rule: str = "diminishing") -> OptimizationResult:
        """Subgradient method: xₖ₊₁ = xₖ - αₖgₖ where gₖ ∈ ∂f(xₖ)."""
        result = OptimizationResult()
        x = x0.copy()
        
        best_f = float('inf')
        best_x = x0.copy()
        
        for k in range(1, self.max_iterations + 1):
            f_x = f(x)
            subgrad_x = subgrad_f(x)
            subgrad_norm = subgrad_x.magnitude()
            
            result.history.append((x.copy(), f_x))
            result.gradient_norms.append(subgrad_norm)
            
            # Track best point
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            # Step size rule
            if step_rule == "diminishing":
                alpha_k = 1.0 / math.sqrt(k)
            elif step_rule == "constant":
                alpha_k = 0.01
            else:
                alpha_k = 1.0 / k
            
            # Subgradient step
            if subgrad_norm > 1e-15:
                x = x - alpha_k * subgrad_x
        
        result.x_optimal = best_x
        result.f_optimal = best_f
        result.iterations = len(result.history)
        
        return result


def compare_methods(f: Callable, grad_f: Callable, x0: Vector,
                   methods: List[str] = None) -> Dict[str, OptimizationResult]:
    """Compare different optimization methods."""
    if methods is None:
        methods = ["gd", "accelerated", "adagrad", "rmsprop", "adam"]
    
    results = {}
    
    if "gd" in methods:
        gd = GradientDescent()
        results["Gradient Descent"] = gd.optimize(f, grad_f, x0)
    
    if "accelerated" in methods:
        agd = AcceleratedGradientDescent()
        results["Accelerated GD"] = agd.optimize(f, grad_f, x0)
    
    if "adagrad" in methods:
        adagrad = AdaptiveGradientDescent()
        results["AdaGrad"] = adagrad.optimize(f, grad_f, x0)
    
    if "rmsprop" in methods:
        rmsprop = RMSprop()
        results["RMSprop"] = rmsprop.optimize(f, grad_f, x0)
    
    if "adam" in methods:
        adam = Adam()
        results["Adam"] = adam.optimize(f, grad_f, x0)
    
    return results
"""
Convergence Analysis and Rates
=============================

Mathematical analysis of convergence rates for optimization algorithms.
"""

import math
from typing import List, Tuple, Callable, Dict, Optional
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector


class ConvergenceAnalyzer:
    """Analyze convergence rates of optimization algorithms."""
    
    @staticmethod
    def analyze_linear_convergence(history: List[Tuple[Vector, float]], 
                                 f_optimal: float) -> Dict[str, float]:
        """
        Analyze linear convergence: ||xₖ - x*|| ≤ ρᵏ||x₀ - x*||
        
        Linear convergence rate ρ ∈ (0,1) means geometric decrease.
        """
        if len(history) < 3:
            return {"rate": 0.0, "constant": 0.0}
        
        # Compute function value gaps
        gaps = [abs(f_k - f_optimal) for _, f_k in history if abs(f_k - f_optimal) > 1e-15]
        
        if len(gaps) < 2:
            return {"rate": 0.0, "constant": 0.0}
        
        # Estimate ρ from consecutive ratios
        ratios = []
        for i in range(1, len(gaps)):
            if gaps[i-1] > 1e-15:
                ratios.append(gaps[i] / gaps[i-1])
        
        if not ratios:
            return {"rate": 0.0, "constant": 0.0}
        
        # Average ratio gives convergence rate
        avg_ratio = sum(ratios) / len(ratios)
        
        return {
            "rate": min(avg_ratio, 1.0),
            "constant": gaps[0] if gaps else 0.0,
            "type": "linear" if avg_ratio < 1.0 else "sublinear"
        }
    
    @staticmethod
    def analyze_quadratic_convergence(history: List[Tuple[Vector, float]],
                                    x_optimal: Vector) -> Dict[str, float]:
        """
        Analyze quadratic convergence: ||xₖ₊₁ - x*|| ≤ C||xₖ - x*||²
        
        Quadratic convergence is characteristic of Newton's method near optimum.
        """
        if len(history) < 3:
            return {"rate": 0.0, "constant": 0.0}
        
        # Compute distance to optimum
        distances = [(x_k - x_optimal).magnitude() for x_k, _ in history]
        
        # Check quadratic relationship
        quadratic_ratios = []
        for i in range(1, len(distances) - 1):
            if distances[i] > 1e-15 and distances[i]**2 > 1e-15:
                ratio = distances[i+1] / (distances[i]**2)
                quadratic_ratios.append(ratio)
        
        if not quadratic_ratios:
            return {"rate": 0.0, "constant": 0.0, "type": "not_quadratic"}
        
        avg_constant = sum(quadratic_ratios) / len(quadratic_ratios)
        
        # Check if ratios are roughly constant (indicating quadratic convergence)
        variance = sum((r - avg_constant)**2 for r in quadratic_ratios) / len(quadratic_ratios)
        
        return {
            "constant": avg_constant,
            "variance": variance,
            "type": "quadratic" if variance < 0.1 * avg_constant else "not_quadratic"
        }
    
    @staticmethod
    def analyze_sublinear_convergence(history: List[Tuple[Vector, float]],
                                    f_optimal: float) -> Dict[str, float]:
        """
        Analyze sublinear convergence: f(xₖ) - f* ≤ C/k^α
        
        Common for subgradient methods: α = 1/2
        """
        if len(history) < 5:
            return {"rate": 0.0, "constant": 0.0}
        
        # Function value gaps
        gaps = []
        iterations = []
        
        for k, (_, f_k) in enumerate(history[1:], 1):  # Start from k=1
            gap = abs(f_k - f_optimal)
            if gap > 1e-15:
                gaps.append(gap)
                iterations.append(k)
        
        if len(gaps) < 3:
            return {"rate": 0.0, "constant": 0.0}
        
        # Fit C/k^α model using log-linear regression
        # log(gap) = log(C) - α*log(k)
        
        log_gaps = [math.log(gap) for gap in gaps]
        log_iterations = [math.log(k) for k in iterations]
        
        # Simple linear regression
        n = len(log_gaps)
        sum_x = sum(log_iterations)
        sum_y = sum(log_gaps)
        sum_xy = sum(x * y for x, y in zip(log_iterations, log_gaps))
        sum_x2 = sum(x**2 for x in log_iterations)
        
        # Slope = -α, intercept = log(C)
        alpha = -(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        log_C = (sum_y + alpha * sum_x) / n
        C = math.exp(log_C)
        
        return {
            "rate": alpha,
            "constant": C,
            "type": "sublinear",
            "model": f"C/k^{alpha:.3f} where C={C:.3e}"
        }


class TheoreticalRates:
    """Theoretical convergence rates for different problem classes."""
    
    @staticmethod
    def gradient_descent_smooth(L: float, step_size: float) -> Dict[str, float]:
        """
        GD on L-smooth functions: f(xₖ) - f* ≤ (1 - α/L)ᵏ(f(x₀) - f*)
        
        Args:
            L: Lipschitz constant of gradient
            step_size: Step size α
        """
        if step_size > 2/L:
            return {"rate": float('inf'), "stable": False, "note": "Step size too large"}
        
        convergence_rate = 1 - step_size/L if step_size <= 1/L else 1 - 2*step_size + step_size**2 * L
        
        return {
            "rate": convergence_rate,
            "type": "linear",
            "optimal_step_size": 1/L,
            "stable": step_size <= 2/L
        }
    
    @staticmethod
    def gradient_descent_strongly_convex(mu: float, L: float, step_size: float) -> Dict[str, float]:
        """
        GD on μ-strongly convex, L-smooth functions.
        
        Optimal rate: (1 - μ/L)ᵏ with step size 2/(μ + L)
        """
        kappa = L / mu  # Condition number
        
        if step_size == 2/(mu + L):
            # Optimal step size
            rate = (kappa - 1) / (kappa + 1)
        else:
            # General step size
            rate = max(abs(1 - step_size * mu), abs(1 - step_size * L))
        
        return {
            "rate": rate,
            "condition_number": kappa,
            "optimal_step_size": 2/(mu + L),
            "optimal_rate": (kappa - 1) / (kappa + 1),
            "type": "linear"
        }
    
    @staticmethod
    def accelerated_gradient_descent(L: float) -> Dict[str, float]:
        """
        Nesterov acceleration on L-smooth functions: O(1/k²) rate.
        """
        return {
            "rate": "O(1/k²)",
            "type": "accelerated",
            "step_size": 1/L,
            "improvement_over_gd": "Quadratic vs linear in iteration count"
        }
    
    @staticmethod
    def subgradient_method(G: float, R: float) -> Dict[str, float]:
        """
        Subgradient method: f(x̄ₖ) - f* ≤ (GR)/√k
        
        Args:
            G: Bound on subgradient norms
            R: Bound on distance to optimum
        """
        return {
            "rate": "O(1/√k)",
            "constant": G * R,
            "type": "sublinear",
            "step_size": "O(1/√k) for best rate"
        }
    
    @staticmethod
    def newton_method(L: float, tolerance: float) -> Dict[str, float]:
        """
        Newton's method: quadratic convergence in neighborhood of optimum.
        """
        return {
            "rate": "quadratic",
            "local_convergence": True,
            "basin_of_attraction": f"||x₀ - x*|| < 2/L",
            "iterations_to_tolerance": math.ceil(math.log2(math.log(1/tolerance))),
            "type": "second_order"
        }


def estimate_lipschitz_constant(grad_f: Callable, domain_points: List[Vector]) -> float:
    """
    Estimate Lipschitz constant L: ||∇f(x) - ∇f(y)|| ≤ L||x - y||
    """
    max_ratio = 0.0
    
    for i, x in enumerate(domain_points):
        for y in domain_points[i+1:]:
            try:
                grad_x = grad_f(x)
                grad_y = grad_f(y)
                
                grad_diff_norm = (grad_x - grad_y).magnitude()
                point_diff_norm = (x - y).magnitude()
                
                if point_diff_norm > 1e-15:
                    ratio = grad_diff_norm / point_diff_norm
                    max_ratio = max(max_ratio, ratio)
            except:
                continue
    
    return max_ratio


def estimate_strong_convexity(f: Callable, grad_f: Callable, 
                            domain_points: List[Vector]) -> float:
    """
    Estimate strong convexity parameter μ:
    f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
    """
    min_mu = float('inf')
    
    for i, x in enumerate(domain_points):
        for y in domain_points[i+1:]:
            try:
                f_x = f(x)
                f_y = f(y)
                grad_x = grad_f(x)
                
                diff = y - x
                diff_norm_sq = diff.squared_magnitude()
                
                if diff_norm_sq > 1e-15:
                    # Rearrange strong convexity inequality to solve for μ
                    mu_estimate = 2 * (f_y - f_x - grad_x.dot(diff)) / diff_norm_sq
                    
                    if mu_estimate > 0:
                        min_mu = min(min_mu, mu_estimate)
            except:
                continue
    
    return min_mu if min_mu != float('inf') else 0.0


def convergence_certificate(history: List[Tuple[Vector, float]], 
                          theoretical_rate: Dict[str, float]) -> Dict[str, bool]:
    """
    Verify if observed convergence matches theoretical predictions.
    """
    if len(history) < 5:
        return {"sufficient_data": False}
    
    # Extract function values
    f_values = [f_k for _, f_k in history]
    
    # Check monotonic decrease (for gradient descent)
    monotonic = all(f_values[i+1] <= f_values[i] + 1e-12 for i in range(len(f_values)-1))
    
    # Check if final improvement is small (convergence)
    final_improvement = abs(f_values[-1] - f_values[-5]) / max(abs(f_values[-5]), 1e-15)
    converged = final_improvement < 1e-6
    
    return {
        "sufficient_data": True,
        "monotonic_decrease": monotonic,
        "converged": converged,
        "final_improvement": final_improvement,
        "matches_theory": monotonic and (theoretical_rate.get("type") == "linear" or converged)
    }
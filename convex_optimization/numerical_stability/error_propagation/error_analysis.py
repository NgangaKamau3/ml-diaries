"""
Numerical Error Propagation Analysis
===================================

Analysis of how errors propagate through numerical computations.
"""

import math
from typing import List, Dict, Callable, Tuple, Any
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector
from core.matrix import Matrix


class ErrorPropagationAnalyzer:
    """Analyze error propagation in numerical algorithms."""
    
    def __init__(self, machine_epsilon: float = 2.22e-16):
        self.machine_epsilon = machine_epsilon
    
    def first_order_error_analysis(self, f: Callable, grad_f: Callable, 
                                 x: Vector, delta_x: Vector) -> Dict[str, float]:
        """
        First-order error analysis using Taylor expansion.
        
        f(x + δx) ≈ f(x) + ∇f(x)ᵀδx
        
        Error ≈ ||∇f(x)|| × ||δx||
        """
        try:
            f_x = f(x)
            grad_x = grad_f(x)
            
            # Actual perturbed value
            f_x_pert = f(x + delta_x)
            
            # First-order approximation
            first_order_approx = f_x + grad_x.dot(delta_x)
            
            # Errors
            actual_error = abs(f_x_pert - f_x)
            predicted_error = grad_x.magnitude() * delta_x.magnitude()
            approximation_error = abs(f_x_pert - first_order_approx)
            
            return {
                "actual_error": actual_error,
                "predicted_error": predicted_error,
                "approximation_error": approximation_error,
                "error_amplification": predicted_error / delta_x.magnitude() if delta_x.magnitude() > 1e-15 else 0,
                "prediction_quality": 1 - approximation_error / max(actual_error, 1e-15)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def condition_number_error_bound(self, A: Matrix, x: Vector, b: Vector,
                                   delta_A: Matrix = None, delta_b: Vector = None) -> Dict[str, float]:
        """
        Error bound using condition number for linear systems.
        
        ||δx||/||x|| ≤ κ(A) × (||δA||/||A|| + ||δb||/||b||)
        """
        try:
            # Compute condition number
            from numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer
            analyzer = ConditioningAnalyzer()
            cond_A = analyzer.condition_number_analysis(A)["condition_number"]
            
            # Relative perturbations
            rel_pert_A = delta_A.frobenius_norm() / A.frobenius_norm() if delta_A is not None else 0
            rel_pert_b = delta_b.magnitude() / b.magnitude() if delta_b is not None else 0
            
            # Error bound
            error_bound = cond_A * (rel_pert_A + rel_pert_b)
            
            # If we have actual perturbations, compute actual error
            actual_error = None
            if delta_A is not None or delta_b is not None:
                A_pert = A + delta_A if delta_A is not None else A
                b_pert = b + delta_b if delta_b is not None else b
                
                try:
                    x_pert = A_pert.solve(b_pert)
                    actual_error = (x_pert - x).magnitude() / x.magnitude()
                except:
                    actual_error = float('inf')
            
            return {
                "condition_number": cond_A,
                "relative_perturbation_A": rel_pert_A,
                "relative_perturbation_b": rel_pert_b,
                "theoretical_error_bound": error_bound,
                "actual_relative_error": actual_error,
                "bound_tightness": actual_error / error_bound if actual_error is not None and error_bound > 0 else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def accumulated_rounding_error(self, operations: List[Tuple[str, Any]]) -> Dict[str, float]:
        """
        Analyze accumulated rounding errors in sequence of operations.
        
        Each floating-point operation introduces error ≈ machine_epsilon × |result|
        """
        total_error_bound = 0.0
        current_value = 0.0
        operation_errors = []
        
        for i, (op_type, operands) in enumerate(operations):
            try:
                if op_type == "add":
                    a, b = operands
                    result = a + b
                    error_bound = self.machine_epsilon * abs(result)
                    
                elif op_type == "multiply":
                    a, b = operands
                    result = a * b
                    error_bound = self.machine_epsilon * abs(result)
                    
                elif op_type == "divide":
                    a, b = operands
                    if abs(b) < 1e-15:
                        error_bound = float('inf')
                        result = float('inf')
                    else:
                        result = a / b
                        error_bound = self.machine_epsilon * abs(result)
                
                elif op_type == "sqrt":
                    a = operands[0]
                    if a < 0:
                        error_bound = float('inf')
                        result = float('nan')
                    else:
                        result = math.sqrt(a)
                        error_bound = 0.5 * self.machine_epsilon * abs(result)
                
                else:
                    error_bound = 0.0
                    result = operands[0] if operands else 0.0
                
                total_error_bound += error_bound
                current_value = result
                operation_errors.append({
                    "operation": f"{op_type}({operands})",
                    "result": result,
                    "error_bound": error_bound
                })
                
            except Exception as e:
                operation_errors.append({
                    "operation": f"{op_type}({operands})",
                    "error": str(e)
                })
        
        return {
            "final_value": current_value,
            "total_error_bound": total_error_bound,
            "relative_error_bound": total_error_bound / max(abs(current_value), 1e-15),
            "operation_details": operation_errors,
            "error_growth": "linear" if len(operations) > 0 else "none"
        }
    
    def iterative_error_analysis(self, iteration_function: Callable, 
                                x0: Vector, num_iterations: int,
                                exact_solution: Vector = None) -> Dict[str, List[float]]:
        """
        Analyze error propagation in iterative methods.
        
        Tracks how errors accumulate over iterations.
        """
        x = x0.copy()
        errors = []
        residuals = []
        
        for k in range(num_iterations):
            try:
                x_new = iteration_function(x, k)
                
                # Error relative to exact solution (if known)
                if exact_solution is not None:
                    error = (x_new - exact_solution).magnitude()
                    errors.append(error)
                
                # Residual (change between iterations)
                residual = (x_new - x).magnitude()
                residuals.append(residual)
                
                x = x_new
                
            except Exception as e:
                break
        
        # Analyze convergence rate
        convergence_rate = None
        if len(errors) > 2:
            # Estimate convergence rate from last few iterations
            ratios = []
            for i in range(len(errors) - 3, len(errors) - 1):
                if errors[i] > 1e-15:
                    ratios.append(errors[i+1] / errors[i])
            
            if ratios:
                convergence_rate = sum(ratios) / len(ratios)
        
        return {
            "errors": errors,
            "residuals": residuals,
            "final_error": errors[-1] if errors else None,
            "convergence_rate": convergence_rate,
            "converged": residuals[-1] < 1e-10 if residuals else False
        }
    
    def sensitivity_analysis(self, f: Callable, x: Vector, 
                           perturbation_sizes: List[float] = None) -> Dict[str, List[float]]:
        """
        Analyze sensitivity of function to input perturbations.
        
        Measures how output changes with different perturbation magnitudes.
        """
        if perturbation_sizes is None:
            perturbation_sizes = [1e-8, 1e-6, 1e-4, 1e-2]
        
        f_x = f(x)
        sensitivities = []
        
        for eps in perturbation_sizes:
            max_sensitivity = 0.0
            
            # Test perturbations in each coordinate direction
            for i in range(x.dimension):
                # Positive perturbation
                x_pert = x.copy()
                x_pert._components[i] += eps
                
                try:
                    f_pert = f(x_pert)
                    sensitivity = abs(f_pert - f_x) / eps
                    max_sensitivity = max(max_sensitivity, sensitivity)
                except:
                    max_sensitivity = float('inf')
                    break
                
                # Negative perturbation
                x_pert._components[i] = x[i] - eps
                
                try:
                    f_pert = f(x_pert)
                    sensitivity = abs(f_pert - f_x) / eps
                    max_sensitivity = max(max_sensitivity, sensitivity)
                except:
                    max_sensitivity = float('inf')
                    break
            
            sensitivities.append(max_sensitivity)
        
        return {
            "perturbation_sizes": perturbation_sizes,
            "max_sensitivities": sensitivities,
            "sensitivity_trend": "increasing" if len(sensitivities) > 1 and sensitivities[-1] > sensitivities[0] else "stable"
        }


def demonstrate_error_propagation():
    """Demonstrate error propagation in numerical computations."""
    
    print("Error Propagation Analysis Demo")
    print("=" * 40)
    
    analyzer = ErrorPropagationAnalyzer()
    
    # 1. Accumulated rounding errors
    print("\n1. Accumulated Rounding Errors:")
    operations = [
        ("add", (1.0, 1e-16)),
        ("multiply", (2.0, 1e8)),
        ("divide", (1e8, 3.0)),
        ("sqrt", (4.0,))
    ]
    
    error_analysis = analyzer.accumulated_rounding_error(operations)
    print(f"Final value: {error_analysis['final_value']:.6e}")
    print(f"Total error bound: {error_analysis['total_error_bound']:.6e}")
    print(f"Relative error bound: {error_analysis['relative_error_bound']:.6e}")
    
    # 2. Sensitivity analysis
    print("\n2. Sensitivity Analysis:")
    def test_function(x):
        return x[0]**2 + x[1]**2  # Simple quadratic
    
    x_test = Vector([1.0, 1.0])
    sensitivity = analyzer.sensitivity_analysis(test_function, x_test)
    
    for eps, sens in zip(sensitivity["perturbation_sizes"], sensitivity["max_sensitivities"]):
        print(f"Perturbation {eps:.0e}: Max sensitivity {sens:.2f}")
    
    # 3. Condition number effects
    print("\n3. Condition Number Effects:")
    
    # Well-conditioned system
    A_good = Matrix([[2, 1], [1, 2]])
    b = Vector([3, 3])
    x = A_good.solve(b)
    
    # Small perturbation
    delta_b = Vector([1e-10, 1e-10])
    
    error_bound = analyzer.condition_number_error_bound(A_good, x, b, delta_b=delta_b)
    print(f"Well-conditioned system:")
    print(f"  Condition number: {error_bound['condition_number']:.2f}")
    print(f"  Error bound: {error_bound['theoretical_error_bound']:.2e}")
    
    # Ill-conditioned system
    A_bad = Matrix([[1, 1], [1, 1.000001]])
    try:
        x_bad = A_bad.solve(b)
        error_bound_bad = analyzer.condition_number_error_bound(A_bad, x_bad, b, delta_b=delta_b)
        print(f"\nIll-conditioned system:")
        print(f"  Condition number: {error_bound_bad['condition_number']:.2e}")
        print(f"  Error bound: {error_bound_bad['theoretical_error_bound']:.2e}")
    except:
        print(f"\nIll-conditioned system: Too ill-conditioned to solve")


if __name__ == "__main__":
    demonstrate_error_propagation()
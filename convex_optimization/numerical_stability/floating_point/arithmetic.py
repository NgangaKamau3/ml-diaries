"""
Floating-Point Arithmetic Analysis
=================================

Analysis of floating-point representation and arithmetic errors.
"""

import math
import sys
from typing import Tuple, List, Dict


class FloatingPointAnalyzer:
    """Analyze floating-point arithmetic properties."""
    
    def __init__(self):
        self.machine_epsilon = self._compute_machine_epsilon()
        self.largest_float = sys.float_info.max
        self.smallest_positive_float = sys.float_info.min
        self.precision_digits = sys.float_info.dig
    
    def _compute_machine_epsilon(self) -> float:
        """
        Compute machine epsilon: smallest ε such that 1 + ε > 1.
        
        Machine epsilon characterizes floating-point precision.
        For IEEE 754 double precision: ε ≈ 2.22 × 10⁻¹⁶
        """
        epsilon = 1.0
        while 1.0 + epsilon > 1.0:
            epsilon /= 2.0
        return 2.0 * epsilon
    
    def relative_error(self, computed: float, exact: float) -> float:
        """
        Compute relative error: |computed - exact| / |exact|
        
        Relative error is more meaningful than absolute error for
        floating-point analysis.
        """
        if abs(exact) < 1e-15:
            return abs(computed - exact)  # Absolute error when exact ≈ 0
        return abs(computed - exact) / abs(exact)
    
    def ulp_distance(self, x: float, y: float) -> float:
        """
        Compute distance in Units in the Last Place (ULP).
        
        ULP measures how many representable floating-point numbers
        lie between x and y.
        """
        if x == y:
            return 0.0
        
        # Handle special cases
        if math.isnan(x) or math.isnan(y):
            return float('inf')
        if math.isinf(x) or math.isinf(y):
            return float('inf')
        
        # Get bit representations (simplified)
        return abs(x - y) / (self.machine_epsilon * max(abs(x), abs(y)))
    
    def analyze_catastrophic_cancellation(self, a: float, b: float) -> Dict[str, float]:
        """
        Analyze catastrophic cancellation in subtraction a - b.
        
        When a ≈ b, subtraction loses significant digits.
        """
        if abs(a) < 1e-15 and abs(b) < 1e-15:
            return {"severity": 0.0, "lost_digits": 0}
        
        # Measure how close a and b are
        relative_difference = abs(a - b) / max(abs(a), abs(b))
        
        # Estimate lost digits
        if relative_difference > 0:
            lost_digits = -math.log10(relative_difference)
        else:
            lost_digits = float('inf')
        
        return {
            "relative_difference": relative_difference,
            "lost_digits": min(lost_digits, 16),  # Cap at double precision
            "severity": "high" if lost_digits > 8 else "moderate" if lost_digits > 4 else "low"
        }
    
    def stable_sum(self, numbers: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute sum using Kahan summation for improved accuracy.
        
        Kahan summation reduces accumulated rounding errors.
        """
        if not numbers:
            return 0.0, {"error_bound": 0.0}
        
        # Standard summation
        standard_sum = sum(numbers)
        
        # Kahan summation
        kahan_sum = 0.0
        compensation = 0.0
        
        for x in numbers:
            y = x - compensation
            t = kahan_sum + y
            compensation = (t - kahan_sum) - y
            kahan_sum = t
        
        # Error analysis
        error_bound = len(numbers) * self.machine_epsilon * sum(abs(x) for x in numbers)
        
        return kahan_sum, {
            "standard_sum": standard_sum,
            "kahan_sum": kahan_sum,
            "improvement": abs(standard_sum - kahan_sum),
            "theoretical_error_bound": error_bound
        }
    
    def stable_dot_product(self, x: List[float], y: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute dot product with error analysis.
        
        Dot product accumulates rounding errors from multiplication and addition.
        """
        if len(x) != len(y):
            raise ValueError("Vectors must have same length")
        
        # Standard computation
        standard_dot = sum(xi * yi for xi, yi in zip(x, y))
        
        # Kahan summation for products
        products = [xi * yi for xi, yi in zip(x, y)]
        kahan_dot, sum_info = self.stable_sum(products)
        
        # Error bound: roughly n * ε * ||x|| * ||y||
        norm_x = math.sqrt(sum(xi**2 for xi in x))
        norm_y = math.sqrt(sum(yi**2 for yi in y))
        error_bound = len(x) * self.machine_epsilon * norm_x * norm_y
        
        return kahan_dot, {
            "standard_dot": standard_dot,
            "kahan_dot": kahan_dot,
            "error_bound": error_bound,
            "relative_error_bound": error_bound / max(abs(kahan_dot), 1e-15)
        }


class NumericalStabilityTester:
    """Test numerical stability of algorithms."""
    
    def __init__(self):
        self.fp_analyzer = FloatingPointAnalyzer()
    
    def test_algorithm_stability(self, algorithm: callable, inputs: List,
                               perturbation_scale: float = 1e-12) -> Dict[str, float]:
        """
        Test algorithm stability by perturbing inputs.
        
        A stable algorithm produces similar outputs for similar inputs.
        """
        if not inputs:
            return {"stability": 0.0}
        
        # Compute baseline result
        try:
            baseline_result = algorithm(*inputs)
        except:
            return {"stability": 0.0, "error": "Algorithm failed on baseline inputs"}
        
        # Test with perturbed inputs
        stability_measures = []
        
        for _ in range(10):  # Multiple random perturbations
            perturbed_inputs = []
            
            for inp in inputs:
                if isinstance(inp, (int, float)):
                    # Add relative perturbation
                    perturbation = perturbation_scale * abs(inp) if abs(inp) > 1e-15 else perturbation_scale
                    perturbed_inputs.append(inp + perturbation * (2 * hash(str(inp)) % 2 - 1))
                else:
                    # For non-numeric inputs, use original
                    perturbed_inputs.append(inp)
            
            try:
                perturbed_result = algorithm(*perturbed_inputs)
                
                # Measure stability
                if isinstance(baseline_result, (int, float)):
                    relative_change = self.fp_analyzer.relative_error(perturbed_result, baseline_result)
                    stability_measures.append(relative_change / perturbation_scale)
                
            except:
                stability_measures.append(float('inf'))  # Algorithm became unstable
        
        if not stability_measures:
            return {"stability": 0.0}
        
        avg_stability = sum(s for s in stability_measures if s != float('inf')) / len(stability_measures)
        
        return {
            "stability_ratio": avg_stability,
            "classification": "stable" if avg_stability < 10 else "moderately_stable" if avg_stability < 100 else "unstable",
            "failed_perturbations": sum(1 for s in stability_measures if s == float('inf'))
        }
    
    def condition_number_effect(self, matrix_multiply: callable, 
                              well_conditioned_matrix, ill_conditioned_matrix,
                              test_vector) -> Dict[str, float]:
        """
        Demonstrate effect of condition number on numerical stability.
        """
        results = {}
        
        # Test well-conditioned case
        try:
            result_good = matrix_multiply(well_conditioned_matrix, test_vector)
            stability_good = self.test_algorithm_stability(
                matrix_multiply, [well_conditioned_matrix, test_vector]
            )
            results["well_conditioned"] = stability_good
        except:
            results["well_conditioned"] = {"error": "Failed"}
        
        # Test ill-conditioned case
        try:
            result_bad = matrix_multiply(ill_conditioned_matrix, test_vector)
            stability_bad = self.test_algorithm_stability(
                matrix_multiply, [ill_conditioned_matrix, test_vector]
            )
            results["ill_conditioned"] = stability_bad
        except:
            results["ill_conditioned"] = {"error": "Failed"}
        
        return results


def demonstrate_floating_point_issues():
    """Demonstrate common floating-point arithmetic issues."""
    
    print("Floating-Point Arithmetic Issues Demo")
    print("=" * 40)
    
    analyzer = FloatingPointAnalyzer()
    
    # 1. Machine epsilon
    print(f"\n1. Machine epsilon: {analyzer.machine_epsilon:.2e}")
    print(f"   1 + ε = {1.0 + analyzer.machine_epsilon}")
    print(f"   1 + ε/2 = {1.0 + analyzer.machine_epsilon/2}")
    
    # 2. Catastrophic cancellation
    print(f"\n2. Catastrophic Cancellation:")
    a, b = 1.0000001, 1.0000000
    cancellation = analyzer.analyze_catastrophic_cancellation(a, b)
    print(f"   {a} - {b} = {a - b}")
    print(f"   Lost digits: {cancellation['lost_digits']:.1f}")
    
    # 3. Associativity failure
    print(f"\n3. Associativity Failure:")
    x, y, z = 1e20, -1e20, 1.0
    left_assoc = (x + y) + z
    right_assoc = x + (y + z)
    print(f"   ({x:.0e} + {y:.0e}) + {z} = {left_assoc}")
    print(f"   {x:.0e} + ({y:.0e} + {z}) = {right_assoc}")
    print(f"   Equal? {left_assoc == right_assoc}")
    
    # 4. Kahan summation benefit
    print(f"\n4. Kahan Summation:")
    numbers = [1e10, 1.0, -1e10, 1.0] * 1000
    kahan_result, info = analyzer.stable_sum(numbers)
    print(f"   Standard sum: {info['standard_sum']}")
    print(f"   Kahan sum: {info['kahan_sum']}")
    print(f"   Improvement: {info['improvement']:.2e}")


if __name__ == "__main__":
    demonstrate_floating_point_issues()
"""
Matrix Conditioning and Stability
=================================

Analysis of matrix conditioning and its effect on numerical algorithms.
"""

import math
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.matrix import Matrix
from core.vector import Vector


class ConditioningAnalyzer:
    """Analyze matrix conditioning and stability."""
    
    def condition_number_analysis(self, A: Matrix, norm_type: str = "2") -> Dict[str, float]:
        """
        Comprehensive condition number analysis.
        
        κ(A) = ||A|| × ||A⁻¹|| measures sensitivity to perturbations.
        """
        if not A.is_square():
            raise ValueError("Condition number requires square matrix")
        
        try:
            # Compute condition number
            if norm_type == "2":
                # Spectral condition number (most common)
                cond_num = self._spectral_condition_number(A)
            elif norm_type == "frobenius":
                norm_A = A.frobenius_norm()
                A_inv = A.inverse()
                norm_A_inv = A_inv.frobenius_norm()
                cond_num = norm_A * norm_A_inv
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            # Classification
            if cond_num < 1e2:
                classification = "well-conditioned"
                stability = "excellent"
            elif cond_num < 1e6:
                classification = "moderately ill-conditioned"
                stability = "acceptable"
            elif cond_num < 1e12:
                classification = "ill-conditioned"
                stability = "poor"
            else:
                classification = "severely ill-conditioned"
                stability = "very poor"
            
            # Estimate digits of accuracy lost
            digits_lost = math.log10(cond_num) if cond_num > 1 else 0
            
            return {
                "condition_number": cond_num,
                "classification": classification,
                "stability": stability,
                "digits_lost": digits_lost,
                "relative_error_amplification": cond_num
            }
            
        except ValueError as e:
            return {
                "condition_number": float('inf'),
                "classification": "singular",
                "stability": "unstable",
                "error": str(e)
            }
    
    def _spectral_condition_number(self, A: Matrix) -> float:
        """Compute spectral condition number using SVD."""
        try:
            from eigenvalues.svd import SVD
            svd = SVD(A)
            _, S, _ = svd.compute()
            
            singular_values = [s for s in S.components if s > 1e-15]
            if not singular_values:
                return float('inf')
            
            return max(singular_values) / min(singular_values)
        except:
            # Fallback: estimate using power method
            return A.spectral_norm_estimate() / self._smallest_singular_value_estimate(A)
    
    def _smallest_singular_value_estimate(self, A: Matrix) -> float:
        """Estimate smallest singular value using inverse power method."""
        try:
            A_inv = A.inverse()
            return 1.0 / A_inv.spectral_norm_estimate()
        except:
            return 1e-15  # Essentially singular
    
    def perturbation_analysis(self, A: Matrix, b: Vector, 
                            delta_A: Optional[Matrix] = None,
                            delta_b: Optional[Vector] = None) -> Dict[str, float]:
        """
        Analyze how perturbations in A and b affect solution of Ax = b.
        
        Theory: ||δx||/||x|| ≤ κ(A) × (||δA||/||A|| + ||δb||/||b||)
        """
        if not A.is_square() or A.rows != b.dimension:
            raise ValueError("Incompatible matrix and vector dimensions")
        
        try:
            # Solve original system
            x = A.solve(b)
            
            # Condition number
            cond_A = self.condition_number_analysis(A)["condition_number"]
            
            results = {
                "condition_number": cond_A,
                "original_solution_norm": x.magnitude()
            }
            
            # Analyze perturbations if provided
            if delta_A is not None or delta_b is not None:
                # Create perturbed system
                A_pert = A + delta_A if delta_A is not None else A
                b_pert = b + delta_b if delta_b is not None else b
                
                try:
                    x_pert = A_pert.solve(b_pert)
                    
                    # Actual error
                    delta_x = x_pert - x
                    relative_error = delta_x.magnitude() / x.magnitude() if x.magnitude() > 1e-15 else delta_x.magnitude()
                    
                    # Theoretical bound
                    relative_pert_A = delta_A.frobenius_norm() / A.frobenius_norm() if delta_A is not None else 0
                    relative_pert_b = delta_b.magnitude() / b.magnitude() if delta_b is not None else 0
                    
                    theoretical_bound = cond_A * (relative_pert_A + relative_pert_b)
                    
                    results.update({
                        "actual_relative_error": relative_error,
                        "theoretical_error_bound": theoretical_bound,
                        "bound_tightness": relative_error / theoretical_bound if theoretical_bound > 0 else 0,
                        "perturbation_amplification": relative_error / (relative_pert_A + relative_pert_b) if (relative_pert_A + relative_pert_b) > 0 else 0
                    })
                    
                except ValueError:
                    results["perturbed_system"] = "singular or unsolvable"
            
            return results
            
        except ValueError as e:
            return {"error": f"Original system unsolvable: {e}"}
    
    def iterative_refinement_analysis(self, A: Matrix, b: Vector, 
                                    max_refinements: int = 5) -> Dict[str, List[float]]:
        """
        Analyze iterative refinement for improving solution accuracy.
        
        Algorithm:
        1. Solve Ax₀ = b
        2. For k = 0, 1, 2, ...:
           - Compute residual: rₖ = b - Axₖ
           - Solve Aδxₖ = rₖ
           - Update: xₖ₊₁ = xₖ + δxₖ
        """
        try:
            # Initial solution
            x = A.solve(b)
            
            residual_norms = []
            solution_changes = []
            
            for k in range(max_refinements):
                # Compute residual
                residual = b - A * x
                residual_norm = residual.magnitude()
                residual_norms.append(residual_norm)
                
                if residual_norm < 1e-15:
                    break
                
                # Solve for correction
                try:
                    delta_x = A.solve(residual)
                    solution_change = delta_x.magnitude()
                    solution_changes.append(solution_change)
                    
                    # Update solution
                    x = x + delta_x
                    
                except ValueError:
                    break
            
            return {
                "residual_norms": residual_norms,
                "solution_changes": solution_changes,
                "final_residual": residual_norms[-1] if residual_norms else float('inf'),
                "convergence": len(residual_norms) > 1 and residual_norms[-1] < residual_norms[0] * 1e-6
            }
            
        except ValueError as e:
            return {"error": f"Iterative refinement failed: {e}"}
    
    def backward_stability_analysis(self, A: Matrix, x: Vector, b: Vector) -> Dict[str, float]:
        """
        Analyze backward stability: find smallest perturbation such that
        (A + δA)x = b + δb exactly.
        
        Backward error = ||r|| / (||A|| × ||x|| + ||b||) where r = Ax - b
        """
        # Compute residual
        residual = A * x - b
        residual_norm = residual.magnitude()
        
        # Normalization factors
        A_norm = A.frobenius_norm()
        x_norm = x.magnitude()
        b_norm = b.magnitude()
        
        # Backward error
        denominator = A_norm * x_norm + b_norm
        backward_error = residual_norm / denominator if denominator > 1e-15 else residual_norm
        
        # Forward error bound
        cond_A = self.condition_number_analysis(A)["condition_number"]
        forward_error_bound = cond_A * backward_error
        
        return {
            "residual_norm": residual_norm,
            "backward_error": backward_error,
            "forward_error_bound": forward_error_bound,
            "stability_classification": "excellent" if backward_error < 1e-14 else "good" if backward_error < 1e-10 else "poor"
        }


def create_test_matrices() -> Dict[str, Matrix]:
    """Create matrices with different conditioning properties."""
    
    matrices = {}
    
    # Well-conditioned matrix
    matrices["well_conditioned"] = Matrix([[2, 1], [1, 2]])
    
    # Moderately ill-conditioned (Hilbert matrix)
    n = 4
    hilbert_data = [[1/(i+j+1) for j in range(n)] for i in range(n)]
    matrices["hilbert_4x4"] = Matrix(hilbert_data)
    
    # Severely ill-conditioned (near-singular)
    matrices["near_singular"] = Matrix([[1, 1], [1, 1.000001]])
    
    # Diagonal matrix with varying condition numbers
    matrices["diagonal_good"] = Matrix.diagonal([1, 2, 3])
    matrices["diagonal_bad"] = Matrix.diagonal([1, 0.001, 0.000001])
    
    return matrices


def demonstrate_conditioning_effects():
    """Demonstrate effects of matrix conditioning."""
    
    print("Matrix Conditioning Effects Demo")
    print("=" * 40)
    
    analyzer = ConditioningAnalyzer()
    test_matrices = create_test_matrices()
    
    for name, A in test_matrices.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"Matrix A:")
        print(A)
        
        # Condition number analysis
        cond_analysis = analyzer.condition_number_analysis(A)
        print(f"Condition number: {cond_analysis['condition_number']:.2e}")
        print(f"Classification: {cond_analysis['classification']}")
        
        # Test with a simple right-hand side
        if A.is_square():
            try:
                b = Vector([1.0] * A.rows)
                x = A.solve(b)
                
                # Backward stability
                stability = analyzer.backward_stability_analysis(A, x, b)
                print(f"Backward error: {stability['backward_error']:.2e}")
                print(f"Stability: {stability['stability_classification']}")
                
            except ValueError:
                print("Matrix is singular - cannot solve system")
        
        print("-" * 30)


if __name__ == "__main__":
    demonstrate_conditioning_effects()
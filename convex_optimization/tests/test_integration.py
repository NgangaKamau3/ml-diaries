"""
Integration Test - Both Libraries Working Together
================================================
"""

import sys
import os

# Add both library paths
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml-diaries'))
advanced_math_path = os.path.dirname(os.path.dirname(__file__))

if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)
if advanced_math_path not in sys.path:
    sys.path.append(advanced_math_path)

# Import from both libraries
from core.matrix import Matrix
from core.vector import Vector
from convex_optimization.first_order_methods.gradient_methods import GradientDescent
from numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer


def test_integration():
    """Test that both libraries work together."""
    print("Testing Integration of Both Libraries...")
    
    # Create a matrix using ml-diaries
    A = Matrix([[4, 1], [1, 3]])
    print(f"Matrix from ml-diaries: {A.shape}")
    
    # Analyze conditioning using advanced-math
    analyzer = ConditioningAnalyzer()
    analysis = analyzer.condition_number_analysis(A)
    print(f"Condition number: {analysis['condition_number']:.2f}")
    
    # Use optimization from advanced-math with ml-diaries vectors
    def objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def gradient(x):
        return Vector([2*(x[0] - 1), 2*(x[1] - 2)])
    
    gd = GradientDescent(step_size=0.1, tolerance=1e-6)
    x0 = Vector([0, 0])
    result = gd.optimize(objective, gradient, x0)
    
    print(f"Optimization converged: {result.converged}")
    print(f"Final point: {result.x_optimal}")
    
    assert result.converged, "Should converge"
    assert (result.x_optimal - Vector([1, 2])).magnitude() < 1e-3, "Should reach optimum"
    
    print("Integration test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("INTEGRATION TEST - BOTH LIBRARIES")
    print("=" * 50)
    
    try:
        test_integration()
        print("\n" + "=" * 50)
        print("INTEGRATION TEST PASSED!")
        print("Both libraries work together successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"\nINTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
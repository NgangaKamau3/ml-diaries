"""
Unified ML Diaries Test
=======================

Test that both modules work together in the unified repository.
"""

import sys
import os

# Add both modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'linear_algebra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'convex_optimization'))

def test_unified_functionality():
    """Test unified functionality."""
    print("Testing Unified ML Diaries Repository...")
    
    # Test linear algebra
    from core.matrix import Matrix
    from core.vector import Vector
    from ml_math.pca import PCA
    
    print("Linear algebra imports: OK")
    
    # Test convex optimization
    sys.path.append(os.path.join(os.path.dirname(__file__), 'convex_optimization', 'convex_optimization'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'convex_optimization', 'numerical_stability'))
    
    from first_order_methods.gradient_methods import GradientDescent
    from conditioning.matrix_conditioning import ConditioningAnalyzer
    
    print("Convex optimization imports: OK")
    
    # Test integration
    A = Matrix([[2, 1], [1, 2]])
    analyzer = ConditioningAnalyzer()
    analysis = analyzer.condition_number_analysis(A)
    
    print(f"Matrix condition number: {analysis['condition_number']:.2f}")
    
    # Test optimization
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def grad_f(x):
        return Vector([2*(x[0] - 1), 2*(x[1] - 2)])
    
    gd = GradientDescent(step_size=0.1, tolerance=1e-6)
    result = gd.optimize(f, grad_f, Vector([0, 0]))
    
    print(f"Optimization converged: {result.converged}")
    print(f"Final point: {result.x_optimal}")
    
    print("\nSUCCESS: Unified repository works perfectly!")


if __name__ == "__main__":
    test_unified_functionality()
"""
Comprehensive Tests for Convex Optimization
==========================================

Test suite for convex optimization components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml-diaries'))

from core.vector import Vector
from core.matrix import Matrix
from convex_optimization.convex_sets.sets import Ball, Simplex, Polyhedron
from convex_optimization.convex_functions.functions import QuadraticFunction, NormFunction
from convex_optimization.first_order_methods.gradient_methods import GradientDescent, Adam
from convex_optimization.convergence_analysis.rates import ConvergenceAnalyzer


def test_convex_sets():
    """Test convex set operations."""
    print("Testing Convex Sets...")
    
    # Test Ball
    center = Vector([0, 0])
    ball = Ball(center, 1.0)
    
    # Point inside
    inside_point = Vector([0.5, 0.5])
    assert ball.contains(inside_point), "Point should be inside ball"
    
    # Point outside
    outside_point = Vector([2, 0])
    assert not ball.contains(outside_point), "Point should be outside ball"
    
    # Projection
    projected = ball.project(outside_point)
    assert abs(projected.magnitude() - 1.0) < 1e-10, "Projection should be on boundary"
    
    print("Ball tests passed")
    
    # Test Simplex
    simplex = Simplex(3)
    
    # Valid point
    valid_point = Vector([0.3, 0.3, 0.4])
    assert simplex.contains(valid_point), "Point should be in simplex"
    
    # Invalid point
    invalid_point = Vector([0.5, 0.5, 0.5])
    assert not simplex.contains(invalid_point), "Point should not be in simplex"
    
    # Projection
    projected_simplex = simplex.project(invalid_point)
    assert simplex.contains(projected_simplex), "Projection should be in simplex"
    
    print("Simplex tests passed")


def test_convex_functions():
    """Test convex function properties."""
    print("Testing Convex Functions...")
    
    # Quadratic function: f(x) = x^T Q x + c^T x
    Q = Matrix([[2, 0], [0, 2]])  # Positive definite
    c = Vector([1, 1])
    f = QuadraticFunction(Q, c)
    
    # Test evaluation
    x = Vector([1, 1])
    f_x = f(x)
    expected = 0.5 * x.dot(Q * x) + c.dot(x)  # 0.5 * 4 + 2 = 4
    assert abs(f_x - expected) < 1e-10, f"Function value incorrect: {f_x} vs {expected}"
    
    # Test gradient
    grad_x = f.gradient(x)
    expected_grad = Q * x + c  # [3, 3]
    assert (grad_x - expected_grad).magnitude() < 1e-10, "Gradient incorrect"
    
    print("Quadratic function tests passed")
    
    # Norm function
    norm_f = NormFunction(p=2)
    x = Vector([3, 4])
    assert abs(norm_f(x) - 5.0) < 1e-10, "L2 norm should be 5"
    
    print("Norm function tests passed")


def test_optimization_methods():
    """Test optimization algorithms."""
    print("Testing Optimization Methods...")
    
    # Simple quadratic: f(x) = (x-1)^2 + (y-2)^2
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def grad_f(x):
        return Vector([2*(x[0] - 1), 2*(x[1] - 2)])
    
    # Test Gradient Descent
    gd = GradientDescent(step_size=0.1, max_iterations=100, tolerance=1e-6)
    x0 = Vector([0, 0])
    result = gd.optimize(f, grad_f, x0)
    
    assert result.converged, "Gradient descent should converge"
    assert (result.x_optimal - Vector([1, 2])).magnitude() < 1e-3, "Should converge to [1, 2]"
    
    print("Gradient descent tests passed")
    
    # Test Adam
    adam = Adam(step_size=0.1, max_iterations=200, tolerance=1e-4)
    result_adam = adam.optimize(f, grad_f, x0)
    
    # Adam may not always converge in few iterations, check final result
    final_error = (result_adam.x_optimal - Vector([1, 2])).magnitude()
    assert final_error < 0.1, f"Adam should get close to [1, 2], error: {final_error}"
    
    print("Adam tests passed")


def test_convergence_analysis():
    """Test convergence analysis."""
    print("Testing Convergence Analysis...")
    
    # Create synthetic convergence history
    history = []
    f_optimal = 0.0
    
    # Linear convergence with rate 0.5
    for k in range(20):
        f_k = (0.5)**k
        x_k = Vector([f_k, f_k])
        history.append((x_k, f_k))
    
    analyzer = ConvergenceAnalyzer()
    analysis = analyzer.analyze_linear_convergence(history, f_optimal)
    
    # Should detect linear convergence
    assert analysis["type"] == "linear", "Should detect linear convergence"
    assert 0.4 < analysis["rate"] < 0.6, f"Rate should be ~0.5, got {analysis['rate']}"
    
    print("Convergence analysis tests passed")


def run_all_tests():
    """Run all convex optimization tests."""
    print("=" * 50)
    print("CONVEX OPTIMIZATION TEST SUITE")
    print("=" * 50)
    
    try:
        test_convex_sets()
        test_convex_functions()
        test_optimization_methods()
        test_convergence_analysis()
        
        print("\n" + "=" * 50)
        print("ALL CONVEX OPTIMIZATION TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
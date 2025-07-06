"""
Test Numerical Stability Components
==================================
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Add the ml-diaries path for imports
ml_diaries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml-diaries'))
if ml_diaries_path not in sys.path:
    sys.path.append(ml_diaries_path)

from core.vector import Vector
from core.matrix import Matrix
from numerical_stability.floating_point.arithmetic import FloatingPointAnalyzer
from numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer


def test_floating_point():
    """Test floating-point analysis."""
    print("Testing Floating-Point Analysis...")
    
    analyzer = FloatingPointAnalyzer()
    
    # Test machine epsilon
    assert analyzer.machine_epsilon > 0, "Machine epsilon should be positive"
    assert analyzer.machine_epsilon < 1e-10, "Machine epsilon should be small"
    
    # Test stable sum
    numbers = [1e10, 1.0, -1e10, 1.0]
    kahan_sum, info = analyzer.stable_sum(numbers)
    
    assert abs(kahan_sum - 2.0) < 1e-10, "Kahan sum should be accurate"
    assert info['improvement'] >= 0, "Kahan should improve or equal standard sum"
    
    print("Floating-point tests passed")


def test_conditioning():
    """Test matrix conditioning analysis."""
    print("Testing Matrix Conditioning...")
    
    analyzer = ConditioningAnalyzer()
    
    # Well-conditioned matrix
    A_good = Matrix([[2, 1], [1, 2]])
    analysis = analyzer.condition_number_analysis(A_good)
    
    assert analysis['condition_number'] < 10, "Should be well-conditioned"
    assert analysis['classification'] == "well-conditioned", "Should classify as well-conditioned"
    
    # Test backward stability
    b = Vector([3, 3])
    x = A_good.solve(b)
    stability = analyzer.backward_stability_analysis(A_good, x, b)
    
    assert stability['backward_error'] < 1e-10, "Should have small backward error"
    
    print("Conditioning tests passed")


def run_all_tests():
    """Run all numerical stability tests."""
    print("=" * 50)
    print("NUMERICAL STABILITY TEST SUITE")
    print("=" * 50)
    
    try:
        test_floating_point()
        test_conditioning()
        
        print("\n" + "=" * 50)
        print("ALL NUMERICAL STABILITY TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
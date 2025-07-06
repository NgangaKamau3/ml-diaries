"""
ML Diaries: Complete Mathematical Foundation
==========================================

Unified library combining Linear Algebra and Convex Optimization.
"""

__version__ = "2.0.0"
__author__ = "ML Diaries Team"

# Core linear algebra components
try:
    from linear_algebra.core.matrix import Matrix
    from linear_algebra.core.vector import Vector
    from linear_algebra.ml_math.pca import PCA
except ImportError:
    pass

# Convex optimization components  
try:
    from convex_optimization.convex_optimization.first_order_methods.gradient_methods import GradientDescent, Adam
    from convex_optimization.numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer
except ImportError:
    pass

__all__ = ['Matrix', 'Vector', 'PCA', 'GradientDescent', 'Adam', 'ConditioningAnalyzer']

# ML Diaries: Complete Mathematical Foundation

A comprehensive mathematical library for machine learning, covering both **Linear Algebra** and **Convex Optimization** with academic rigor.

## 📚 Repository Structure

### 🔢 **Linear Algebra** (`linear_algebra/`)
Complete implementation of linear algebra fundamentals following Deisenroth, Faisal & Ong's "Mathematics for Machine Learning":

- **Core Components**: Matrix and Vector operations
- **Decompositions**: LU, QR, Cholesky, SVD, Eigenvalue
- **ML Applications**: PCA, regression, classification
- **Numerical Analysis**: Conditioning, stability, error analysis

### 📈 **Convex Optimization** (`convex_optimization/`)
Advanced convex optimization theory following Boyd & Vandenberghe standards:

- **Convex Sets & Functions**: Mathematical foundations
- **First-Order Methods**: Gradient descent variants, acceleration
- **Convergence Analysis**: Theoretical rates and bounds
- **Numerical Stability**: Floating-point analysis, conditioning

## 🚀 Quick Start

```python
# Linear Algebra
from linear_algebra.core.matrix import Matrix
from linear_algebra.core.vector import Vector
from linear_algebra.ml_math.pca import PCA

# Convex Optimization  
from convex_optimization.convex_optimization.first_order_methods.gradient_methods import Adam
from convex_optimization.numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer
```

## 🧪 Testing

```bash
# Test linear algebra components
python linear_algebra/tests/test_comprehensive.py

# Test convex optimization components  
python convex_optimization/tests/test_convex_optimization.py
python convex_optimization/tests/test_numerical_stability.py
```

## 📖 Educational Purpose

This repository serves as a complete mathematical foundation for:
- **Graduate-level machine learning courses**
- **Numerical analysis and optimization research**
- **Mathematical software development**
- **Academic reference implementation**

Both modules maintain the same level of mathematical rigor expected in academic settings while being practically useful for real applications.

## 📄 License

MIT License - See individual module licenses for details.

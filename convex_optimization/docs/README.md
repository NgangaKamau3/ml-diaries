# Advanced Mathematics Library

A rigorous implementation of **Convex Optimization Theory** and **Numerical Stability Analysis** following academic standards from Boyd & Vandenberghe and Higham.

## 🎯 Library Structure

### **Convex Optimization Theory** 📈

#### **Convex Sets** (`convex_sets/`)
- **Hyperplanes & Halfspaces** - Linear constraints and boundaries
- **Balls & Ellipsoids** - Euclidean and general norm balls  
- **Polyhedra** - Intersection of halfspaces
- **Simplex** - Probability simplex and generalizations
- **Convex Hull Operations** - Graham scan and projections

#### **Convex Functions** (`convex_functions/`)
- **Quadratic Functions** - f(x) = ½xᵀQx + cᵀx + d
- **Norm Functions** - Lp norms and their properties
- **Log-Sum-Exp** - Numerically stable implementation
- **Support Functions** - Dual characterization of sets
- **Conjugate Functions** - Fenchel conjugates
- **Proximal Operators** - Moreau envelopes and proximity

#### **First-Order Methods** (`first_order_methods/`)
- **Gradient Descent** - With exact and backtracking line search
- **Accelerated Methods** - Nesterov acceleration (O(1/k²) rate)
- **Adaptive Methods** - AdaGrad, RMSprop, Adam
- **Subgradient Methods** - For non-smooth optimization
- **Method Comparison** - Empirical performance analysis

#### **Convergence Analysis** (`convergence_analysis/`)
- **Linear Convergence** - Geometric rate analysis
- **Quadratic Convergence** - Newton-type methods
- **Sublinear Rates** - O(1/k) and O(1/√k) analysis
- **Theoretical Bounds** - Lipschitz constants and strong convexity
- **Convergence Certificates** - Verification of theoretical predictions

### **Numerical Stability Analysis** 🔬

#### **Floating-Point Arithmetic** (`floating_point/`)
- **Machine Epsilon** - Precision characterization
- **Catastrophic Cancellation** - Loss of significance analysis
- **Kahan Summation** - Compensated summation algorithms
- **ULP Distance** - Units in Last Place measurements
- **Stability Testing** - Algorithm perturbation analysis

#### **Matrix Conditioning** (`conditioning/`)
- **Condition Numbers** - Spectral and Frobenius norms
- **Perturbation Analysis** - Error amplification bounds
- **Iterative Refinement** - Accuracy improvement techniques
- **Backward Stability** - Backward error analysis
- **Ill-Conditioning Detection** - Automatic classification

#### **Error Propagation** (`error_propagation/`)
- **First-Order Analysis** - Taylor expansion error bounds
- **Accumulated Errors** - Rounding error accumulation
- **Sensitivity Analysis** - Input perturbation effects
- **Iterative Error Growth** - Error propagation in iterations
- **Condition-Based Bounds** - Theoretical error estimates

## 🚀 Usage Examples

### **Convex Optimization**
```python
from convex_optimization.convex_sets.sets import Ball, Simplex
from convex_optimization.first_order_methods.gradient_methods import Adam
from convex_optimization.convergence_analysis.rates import ConvergenceAnalyzer

# Define convex constraint set
ball = Ball(center=Vector([0, 0]), radius=1.0)

# Optimize with Adam
def objective(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return Vector([2*x[0], 2*x[1]])

adam = Adam(step_size=0.01)
result = adam.optimize(objective, gradient, Vector([0.5, 0.5]))

# Analyze convergence
analyzer = ConvergenceAnalyzer()
analysis = analyzer.analyze_linear_convergence(result.history, 0.0)
```

### **Numerical Stability**
```python
from numerical_stability.floating_point.arithmetic import FloatingPointAnalyzer
from numerical_stability.conditioning.matrix_conditioning import ConditioningAnalyzer

# Analyze floating-point issues
fp_analyzer = FloatingPointAnalyzer()
numbers = [1e10, 1.0, -1e10, 1.0]
stable_sum, info = fp_analyzer.stable_sum(numbers)

# Matrix conditioning analysis
cond_analyzer = ConditioningAnalyzer()
A = Matrix([[1, 1], [1, 1.000001]])  # Ill-conditioned
analysis = cond_analyzer.condition_number_analysis(A)
print(f"Condition number: {analysis['condition_number']:.2e}")
```

## 📊 Mathematical Rigor

### **Theoretical Foundations**
- **Complete proofs** for convergence rates
- **Precise definitions** of convexity and stability
- **Rigorous error bounds** with constants
- **Complexity analysis** for all algorithms

### **Numerical Properties**
- **IEEE 754 compliance** in floating-point analysis
- **Backward stability** guarantees where applicable
- **Condition number** characterization
- **Error propagation** tracking

### **Academic Standards**
- **Boyd & Vandenberghe** convex optimization theory
- **Higham** numerical stability principles
- **Nocedal & Wright** optimization algorithms
- **Trefethen & Bau** numerical linear algebra

## 🧪 Testing

Run comprehensive test suites:
```bash
python tests/test_convex_optimization.py
python tests/test_numerical_stability.py
```

## 📈 Key Algorithms

### **Optimization Methods**
- **Gradient Descent** with Armijo line search
- **Nesterov Acceleration** with optimal momentum
- **Adam** with bias correction
- **Subgradient Method** with diminishing step sizes

### **Stability Analysis**
- **Condition Number** computation via SVD
- **Kahan Summation** for accurate accumulation  
- **Iterative Refinement** for improved accuracy
- **Perturbation Bounds** using matrix norms

### **Convergence Rates**
- **Linear**: O(ρᵏ) with ρ ∈ (0,1)
- **Quadratic**: O(||xₖ - x*||²) near optimum
- **Sublinear**: O(1/k) for subgradient methods
- **Accelerated**: O(1/k²) for smooth functions

This library provides the mathematical foundation for advanced optimization and numerical analysis, with the same level of rigor expected in graduate-level coursework and research.
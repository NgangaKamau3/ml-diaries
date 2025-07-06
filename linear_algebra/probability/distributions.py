"""
Probability Distributions
========================

Gaussian and multivariate normal distributions for ML applications.
"""

import math
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.vector import Vector
from core.matrix import Matrix


class MultivariateNormal:
    """Multivariate normal distribution N(μ, Σ)."""
    
    def __init__(self, mean: Vector, covariance: Matrix):
        if mean.dimension != covariance.rows:
            raise ValueError("Mean and covariance dimensions must match")
        if not covariance.is_symmetric():
            raise ValueError("Covariance matrix must be symmetric")
        
        self.mean = mean.copy()
        self.covariance = covariance.copy()
        self.dimension = mean.dimension
        
        # Precompute for efficiency
        self._det_cov = None
        self._inv_cov = None
        self._log_det_cov = None
    
    def pdf(self, x: Vector) -> float:
        """Probability density function."""
        if x.dimension != self.dimension:
            raise ValueError("Input dimension mismatch")
        
        if self._inv_cov is None:
            self._inv_cov = self.covariance.inverse()
            self._det_cov = self.covariance.determinant()
        
        # Compute (x - μ)ᵀ Σ⁻¹ (x - μ)
        diff = x - self.mean
        quad_form = diff.dot(self._inv_cov * diff)
        
        # Normalization constant
        norm_const = 1.0 / math.sqrt((2 * math.pi) ** self.dimension * self._det_cov)
        
        return norm_const * math.exp(-0.5 * quad_form)
    
    def log_pdf(self, x: Vector) -> float:
        """Log probability density function."""
        if x.dimension != self.dimension:
            raise ValueError("Input dimension mismatch")
        
        if self._inv_cov is None:
            self._inv_cov = self.covariance.inverse()
        if self._log_det_cov is None:
            self._log_det_cov = math.log(self.covariance.determinant())
        
        diff = x - self.mean
        quad_form = diff.dot(self._inv_cov * diff)
        
        log_norm = -0.5 * (self.dimension * math.log(2 * math.pi) + self._log_det_cov)
        
        return log_norm - 0.5 * quad_form
    
    def sample(self) -> Vector:
        """Generate random sample (simplified)."""
        # This would require Cholesky decomposition for proper sampling
        # Simplified version using standard normal
        import random
        z = Vector([random.gauss(0, 1) for _ in range(self.dimension)])
        
        # Transform: x = μ + L*z where L is Cholesky factor
        from decompositions.cholesky import cholesky_decompose
        try:
            L = cholesky_decompose(self.covariance)
            return self.mean + L * z
        except ValueError:
            # Fallback for non-positive definite matrices
            return self.mean + z
    
    def mahalanobis_distance(self, x: Vector) -> float:
        """Mahalanobis distance from mean."""
        if self._inv_cov is None:
            self._inv_cov = self.covariance.inverse()
        
        diff = x - self.mean
        return math.sqrt(diff.dot(self._inv_cov * diff))


def gaussian_1d(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """1D Gaussian PDF."""
    return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)


def log_gaussian_1d(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """1D Gaussian log-PDF."""
    return -0.5 * math.log(2 * math.pi) - math.log(std) - 0.5 * ((x - mean) / std) ** 2


def kl_divergence_gaussian(mean1: Vector, cov1: Matrix, mean2: Vector, cov2: Matrix) -> float:
    """KL divergence between two multivariate Gaussians."""
    k = mean1.dimension
    
    # KL(N₁||N₂) = 0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - k + ln(|Σ₂|/|Σ₁|)]
    inv_cov2 = cov2.inverse()
    
    # Trace term
    trace_term = (inv_cov2 * cov1).trace()
    
    # Quadratic term
    mean_diff = mean2 - mean1
    quad_term = mean_diff.dot(inv_cov2 * mean_diff)
    
    # Log determinant term
    log_det_term = math.log(cov2.determinant() / cov1.determinant())
    
    return 0.5 * (trace_term + quad_term - k + log_det_term)
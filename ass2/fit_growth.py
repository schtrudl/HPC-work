#!/usr/bin/env python3
"""
Approximate the growth_lenia function with a polynomial using numpy.polyfit.
Uses variable transformation for better accuracy on narrow Gaussian.
"""

import numpy as np
import matplotlib.pyplot as plt

# Original growth_lenia function parameters
MU = 0.15
SIGMA = 0.015


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def growth_lenia(u):
    """Original growth function: -1 + 2 * gauss(u, 0.15, 0.015)"""
    return -1.0 + 2.0 * gauss(u, MU, SIGMA)


# Generate sample points
x_full = np.linspace(0, 1, 1000)
y_full = growth_lenia(x_full)

# The Gaussian is essentially 0 outside ~4 sigma from mu
# 4*sigma = 0.06, so active region is [0.09, 0.21]
# For better fit, use transformed variable: t = (u - mu) / sigma
# This normalizes the input range

# Active region bounds
ACTIVE_MIN = MU - 4 * SIGMA  # 0.09
ACTIVE_MAX = MU + 4 * SIGMA  # 0.21

# Fit in normalized coordinates for better numerical stability
x_fit = np.linspace(ACTIVE_MIN, ACTIVE_MAX, 500)
y_fit = growth_lenia(x_fit)

# Transform to normalized coordinates: t = (u - mu) / sigma
t_fit = (x_fit - MU) / SIGMA  # Range: [-4, 4]
# In this space: growth = -1 + 2*exp(-0.5*t^2)

# Try different polynomial degrees
degrees = [4, 6, 8, 10, 12, 14, 16, 18, 20]

print("Polynomial Approximation of growth_lenia(u)")
print("=" * 60)
print(f"Original: growth_lenia(u) = -1 + 2 * exp(-0.5 * ((u - {MU}) / {SIGMA})^2)")
print(f"Active region: [{ACTIVE_MIN:.3f}, {ACTIVE_MAX:.3f}]")
print()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

best_coeffs = None
best_degree = None
best_error = float("inf")

for degree in degrees:
    # Fit polynomial in normalized coordinates
    coeffs_norm = np.polyfit(t_fit, y_fit, degree)
    poly_norm = np.poly1d(coeffs_norm)

    # Calculate approximation
    y_approx = poly_norm(t_fit)

    # Calculate error
    max_error = np.max(np.abs(y_fit - y_approx))
    rmse = np.sqrt(np.mean((y_fit - y_approx) ** 2))

    print(f"Degree {degree:2d}: Max Error = {max_error:.2e}, RMSE = {rmse:.2e}")

    if max_error < best_error and max_error < 0.01:  # Want < 1% error
        best_error = max_error
        best_coeffs = coeffs_norm
        best_degree = degree

    # Plot in original coordinates
    t_full = (x_full - MU) / SIGMA
    y_approx_full = np.where(
        (x_full < ACTIVE_MIN) | (x_full > ACTIVE_MAX), -1.0, poly_norm(t_full)
    )
    axes[0].plot(x_full, y_approx_full, "--", label=f"Poly deg={degree}", alpha=0.7)

# Use degree 10 if no good fit found
if best_coeffs is None:
    best_degree = 10
    best_coeffs = np.polyfit(t_fit, y_fit, best_degree)

poly_best = np.poly1d(best_coeffs)

# Plot original function
axes[0].plot(x_full, y_full, "k-", linewidth=2, label="Original")
axes[0].set_xlabel("u")
axes[0].set_ylabel("growth_lenia(u)")
axes[0].set_title("Growth Function Polynomial Approximation")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.05, 0.25)

# Error plot for best fit
y_approx = poly_best(t_fit)
error = y_fit - y_approx
axes[1].plot(x_fit, error)
axes[1].set_xlabel("u")
axes[1].set_ylabel("Error")
axes[1].set_title(f"Approximation Error (degree {best_degree})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("growth_fit.png", dpi=150)
print()
print("Saved plot to growth_fit.png")

# Fit even-power polynomial (exploits Gaussian symmetry)
t2_fit = t_fit**2
coeffs_even = np.polyfit(t2_fit, y_fit, best_degree // 2)
poly_even = np.poly1d(coeffs_even)
max_err_even = np.max(np.abs(y_fit - poly_even(t2_fit)))

# Build compact Horner expression
horner = f"{coeffs_even[0]:.10e}f"
for c in coeffs_even[1:]:
    horner = f"({horner}) * t2 + ({c:.10e}f)"

print()
print("C Code (compact even-power polynomial):")
print("=" * 60)
print(f"""// Polynomial approximation of growth_lenia (max error: {max_err_even:.2e})
inline f32 growth_lenia(const f32 u) {{
    const f32 mu = {MU}f, sigma = {SIGMA}f;
    if (u < {ACTIVE_MIN}f || u > {ACTIVE_MAX}f) return -1.0f;
    const f32 t = (u - mu) / sigma, t2 = t * t;
    return {horner};
}}""")

plt.show()

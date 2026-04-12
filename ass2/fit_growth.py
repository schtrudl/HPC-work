#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

MU = 0.15
SIGMA = 0.015


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def growth_lenia(u):
    return -1.0 + 2.0 * gauss(u, MU, SIGMA)


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

# Try different even-power polynomial degrees
even_degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Polynomial Approximation of growth_lenia(u)")
print("=" * 60)
print(f"Original: growth_lenia(u) = -1 + 2 * exp(-0.5 * ((u - {MU}) / {SIGMA})^2)")
print(f"Active region: [{ACTIVE_MIN:.3f}, {ACTIVE_MAX:.3f}]")
print()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

best_coeffs = None
best_degree = None
best_error = float("inf")

# For even-power polynomial fitting
t2_fit = t_fit**2

for even_degree in even_degrees:
    # Fit even-power polynomial (exploits Gaussian symmetry)
    coeffs_even = np.polyfit(t2_fit, y_fit, even_degree)
    poly_even = np.poly1d(coeffs_even)
    y_approx_even = poly_even(t2_fit)

    max_error_even = np.max(np.abs(y_fit - y_approx_even))
    rmse_even = np.sqrt(np.mean((y_fit - y_approx_even) ** 2))

    print(
        f"Even degree {even_degree:2d}: Max Error = {max_error_even:.2e}, RMSE = {rmse_even:.2e}"
    )

    if max_error_even < best_error and max_error_even < 0.01:  # Want < 1% error
        best_error = max_error_even
        best_coeffs = coeffs_even
        best_degree = even_degree

    # Plot in original coordinates
    t_full = (x_full - MU) / SIGMA
    t2_full = t_full**2
    y_approx_full = np.where(
        (x_full < ACTIVE_MIN) | (x_full > ACTIVE_MAX), -1.0, poly_even(t2_full)
    )
    axes[0].plot(
        x_full, y_approx_full, "--", label=f"Even deg={even_degree}", alpha=0.7
    )

assert best_coeffs is not None, "No suitable polynomial found with error < 1%"

poly_best = np.poly1d(best_coeffs)

# Plot original function
axes[0].plot(x_full, y_full, "k-", linewidth=2, label="Original")
axes[0].set_xlabel("u")
axes[0].set_ylabel("growth_lenia(u)")
axes[0].set_title("Growth Function Polynomial Approximation (Even-Power)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.05, 0.25)

# Error plot for best fit (even-power polynomial)
y_approx = poly_best(t2_fit)
error = y_fit - y_approx
axes[1].plot(x_fit, error)
axes[1].set_xlabel("u")
axes[1].set_ylabel("Error")
axes[1].set_title(f"Approximation Error (even-power, degree {best_degree})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("growth_fit.png", dpi=150)
print()
print("Saved plot to growth_fit.png")

# Build compact Horner expression
horner = f"{best_coeffs[0]:.10e}f"
for c in best_coeffs[1:]:
    horner = f"({horner}) * t2 + ({c:.10e}f)"

print()
print("C Code:")
print("=" * 60)
print(f"""// Polynomial approximation of growth_lenia (max error: {best_error:.2e})
inline f32 growth_lenia(const f32 u) {{
    const f32 mu = {MU}f, sigma = {SIGMA}f;
    if (u < {ACTIVE_MIN}f || u > {ACTIVE_MAX}f) return -1.0f;
    const f32 t = (u - mu) / sigma, t2 = t * t;
    return {horner};
}}""")

plt.show()

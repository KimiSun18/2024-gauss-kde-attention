import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Given parameters
beta = 100
n = 10000
num_runs = 80  # Number of simulations

# Store all maxima across runs
all_maxima = []

for _ in range(num_runs):
    # Step 1: Generate X_i from a normal distribution N(0, 1)
    X_i = np.random.normal(0, 1, n)

    # Step 2: Define F'(t) and F''(t)
    def F_prime(t):
        return np.sum(beta * (t - X_i) * np.exp(-beta / 2 * (t - X_i) ** 2))

    def F_double_prime(t):
        return np.sum((1 - beta * (t - X_i) ** 2) * np.exp(-beta / 2 * (t - X_i) ** 2))

    # Step 3: Define the interval T and add focus near 0
    T_min = -np.sqrt(2 * np.log(n) - np.log(beta)) - 1
    T_max = np.sqrt(2 * np.log(n) - np.log(beta)) + 1

    # Increase grid resolution near 0 for better detection
    grid_around_zero = np.linspace(-0.3, 0.3, 10000)  # Higher resolution near 0
    grid_whole = np.linspace(T_min, T_max, 1000)  # Coarser grid for the whole range

    # Step 4: Find critical points (where F'(t) = 0) in the whole grid and near 0
    critical_points = []
    for grid in [grid_around_zero, grid_whole]:
        for i in range(len(grid) - 1):
            try:
                # Find where the derivative is zero
                root = brentq(F_prime, grid[i], grid[i+1])
                critical_points.append(root)
            except ValueError:
                pass  # No root in this interval

    # Step 5: Check which critical points are local maxima
    maxima = []
    for point in critical_points:
        if F_double_prime(point) < 0:  # Check if the second derivative is negative
            maxima.append(point)

    # Accumulate maxima from this run
    all_maxima.extend(maxima)

# Step 6: Plot the empirical distribution of the local maxima over multiple runs
plt.figure(figsize=(8, 6))
plt.hist(all_maxima, bins=50, density=True, edgecolor='lightgray', alpha=0.75, zorder=2)
plt.title(f"Empirical Distribution of Local Maxima", fontsize=14)
plt.xlabel("Local Maxima", fontsize=12)
plt.ylabel("Density", fontsize=12)

# Main grid (behind histogram) and minor grid (lighter sub-grid)
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.5, zorder=1)

# Save the prettier plot
plt.savefig(f"prettier_empirical_local_maxima_distribution_{num_runs}_runs.pdf")
plt.show()

# Step 7: Count the local maxima near t = 0
maxima_near_zero = [m for m in all_maxima if abs(m) < 0.5]
print(f"Number of local maxima near t = 0 (aggregated over {num_runs} runs): {len(maxima_near_zero)}")

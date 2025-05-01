import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX rendering with serif fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Given parameters
beta = 200
n = 10000

def kde(t, beta, n):
    s = 0
    for j in range(n):
        X_j = np.random.normal(0, 1)
        #s += np.sqrt(beta) / (n * np.sqrt(2*np.pi))*np.exp(-beta / 2 * (t - X_j) ** 2)
        s += np.sqrt(beta) / (n * np.sqrt(2*np.pi))*np.exp(-beta / 2 * (t - X_j) ** 2)*(1-beta*(t - X_j)**2)

    return s

# Generate x values
t_values = np.linspace(-10, 10, 500)

# Compute y values for the function
y_values = kde(t_values, beta, n)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the function using Mathematica Red color
plt.plot(t_values, y_values, color='red', alpha=0.75, linewidth=2.5)

# Add major grid with lighter styling
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray', zorder=0)

# Add minor grid (subgrid) with even lighter styling
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='lightgray', zorder=0)

# Add labels and use LaTeX rendering with serif font
plt.xlabel(r'$t$', fontsize=14)

# Add legend
#plt.legend(fontsize=12)

# Improve aesthetics
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlim(t_values.min(), t_values.max())
#plt.ylim(-0.005, np.max(y_values) + 0.05)

# Save the prettier plot
plt.savefig(f"kde_{beta}-{n}.pdf")
plt.show()
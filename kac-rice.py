import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX rendering with serif fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

b = 100
n = 10000

# Define the function to plot
def function(t):
    # ß = 300 -> -3.5
    # ß = 100 -> -2 
    return np.sqrt(b) * np.exp(-b**(-3/2) * n * t**2 * np.exp(-t**2/2) - 2) 

# Generate x values
x_values = np.linspace(-5, 5, 500)

# Compute y values for the function
y_values = function(x_values)

# Compute the interval bounds
interval_min = -np.sqrt(2 * np.log(n) - np.log(b))
interval_max = np.sqrt(2 * np.log(n) - np.log(b))

print(interval_min)
print(interval_max)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the function using Mathematica Red color
plt.plot(x_values, y_values, color='blue', alpha=0.75, linewidth=2, label=r'$f(x) = \sqrt{b} \cdot e^{-b^{-1.5} \cdot n^2 \cdot x^2 \cdot e^{-0.5 x^2}}$')

# Highlight the interval with a shaded region
plt.axvspan(interval_min, interval_max, color='lightgray', alpha=0.3, label=r'$\left[-\sqrt{2 \log n - \log b}, \sqrt{2 \log n - \log b}\right]$')

# Adjust the axis limits so the curve fits exactly within the frame
plt.xlim(x_values.min(), x_values.max())
plt.ylim(0, np.max(y_values) + 0.1)

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

# Save the plot as a PDF file
plt.savefig("graphly_with_highlighted_interval_b=%s_n=%s.pdf" % (b, n), format='pdf', bbox_inches='tight')

# Display the plot
plt.show()
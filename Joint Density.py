import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Parameters
n = 6500         # Number of samples
t = 3            # Mean of the normal distribution
beta = 81        # Value of beta

# Function g(z) = z * exp(-beta/2 * z^2)
def g(z, beta):
    return z * np.exp(-beta / 2 * z**2)

# Derivative of g(z): g'(z) = (1 - beta * z^2) * exp(-beta/2 * z^2)
def g_prime(z, beta):
    return (1 - beta * z**2) * np.exp(-beta / 2 * z**2)

# Generate n samples from N(t, 1)
X = np.random.normal(t, 1, n)

# Compute F and F' as the sum of g(X_i) and g'(X_i)
F = np.sum(g(X, beta))
F_prime = np.sum(g_prime(X, beta))

# Now create many (F, F') pairs to estimate the density
n_simulations = 10000  # Number of simulations to get a good density estimate
F_values = []
F_prime_values = []

for _ in range(n_simulations):
    X = np.random.normal(t, 1, n)  # Sample new X values
    F = np.sum(g(X, beta))         # Compute F
    F_prime = np.sum(g_prime(X, beta))  # Compute F'
    F_values.append(F)
    F_prime_values.append(F_prime)

# Convert to arrays
F_values = np.array(F_values)
F_prime_values = np.array(F_prime_values)

# Perform kernel density estimation on (F, F')
data = np.vstack([F_values, F_prime_values])
kde = gaussian_kde(data)

# Create a grid of points over which to evaluate the density
xmin, ymin = data.min(axis=1)
xmax, ymax = data.max(axis=1)
Xgrid, Ygrid = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
Z = np.reshape(kde(positions).T, Xgrid.shape)

# Plot the 2D density as a contour plot
plt.figure(figsize=(8, 6))
plt.contourf(Xgrid, Ygrid, Z, cmap='viridis')
plt.colorbar()
#plt.title("Density of (F, F')")
plt.xlabel('F')
plt.ylabel('F\'')

# Save the plot as PDF
plt.savefig("density_F_F_prime_b=%s_t=%s.pdf" % (beta, t))  # Save to local directory
plt.show()

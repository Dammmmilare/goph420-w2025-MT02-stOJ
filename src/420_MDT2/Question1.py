import numpy as np
import matplotlib.pyplot as plt

# Given data
masses = np.array([5, 5, 10, 10, 25, 25, 50, 50, 100, 100, 150, 150])
vf_over_g = np.array([0.106, 0.109, 0.217, 0.214, 0.540, 0.520, 1.08, 1.04, 2.09, 2.11, 3.20, 3.13])

# Construct the Z matrix and y vector
Z = masses.reshape(-1, 1)
y = vf_over_g.reshape(-1, 1)

# Least squares estimate: a = (Z^T Z)^-1 Z^T y
a_hat = np.linalg.inv(Z.T @ Z) @ Z.T @ y
a = a_hat[0, 0]
c_estimated = 1 / a

# Check if within design spec (47 ± 0.5)
design_c = 47.0
within_spec = (design_c - 0.5) <= c_estimated <= (design_c + 0.5)

# Predicted y values
y_pred = Z * a
residuals = y - y_pred
r_squared = 1 - (np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))

# Plotting
plt.scatter(masses, vf_over_g, label='Observed', color='blue')
plt.plot(masses, y_pred, label='Best Fit Line', color='red')
plt.xlabel('Mass (kg)')
plt.ylabel('vf / g (s)')
plt.title('Least Squares Fit for Drag Coefficient')
plt.legend()
plt.grid(True)
plt.show()

# Output results
print(f"Estimated drag coefficient c = {c_estimated:.3f} kg/s")
print(f"Within design spec? {'Yes' if within_spec else 'No'}")
print(f"Coefficient of determination r² = {r_squared:.4f}")

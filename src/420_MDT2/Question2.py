import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("Question_2_DATA_rho_vp.txt")
rho = data[:, 0]
Vp = data[:, 1]

# Part (a) - Plot original data
plt.figure()
plt.scatter(rho, Vp, color='blue', label='Observed Data')
plt.xlabel('Density (g/cm³)')
plt.ylabel('P-wave Velocity (m/s)')
plt.title('Vp vs Density')
plt.grid(True)
plt.legend()
plt.show()

# Part (b) - Transform the data
ln_Vp = np.log(Vp)
Z = np.column_stack((np.ones_like(rho), rho))  # [1, rho]

# Least squares solution
theta_hat = np.linalg.inv(Z.T @ Z) @ Z.T @ ln_Vp
a, k = theta_hat
V0 = np.exp(a)

# Predicted in log space and original space
ln_Vp_pred = Z @ theta_hat
Vp_pred = np.exp(ln_Vp_pred)

# Plot in transformed space
plt.figure()
plt.scatter(rho, ln_Vp, color='purple', label='ln(Vp) Observed')
plt.plot(rho, ln_Vp_pred, color='orange', label='Best Fit Line')
plt.xlabel('Density (g/cm³)')
plt.ylabel('ln(P-wave Velocity)')
plt.title('Transformed Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Plot in original space
plt.figure()
plt.scatter(rho, Vp, color='blue', label='Observed Vp')
plt.plot(rho, Vp_pred, color='red', label='Fitted Exponential')
plt.xlabel('Density (g/cm³)')
plt.ylabel('P-wave Velocity (m/s)')
plt.title('Vp vs Density with Fit')
plt.legend()
plt.grid(True)
plt.show()

# Part (d) - r^2 in transformed space
residuals = ln_Vp - ln_Vp_pred
r_squared = 1 - np.sum(residuals**2) / np.sum((ln_Vp - np.mean(ln_Vp))**2)

# Output results
print(f"Estimated V0 = {V0:.3f} m/s")
print(f"Estimated k  = {k:.6f}")
print(f"Coefficient of determination r² = {r_squared:.4f}")

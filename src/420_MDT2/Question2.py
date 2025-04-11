# Question 2 - Linearized Exponential Model

import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("c:/Users/joshu/Repos/Courses/goph420/goph420-w2025-MT02-stOJ/data/Question_2_DATA_rho_vp.txt")
rho = data[:, 0]
Vp = data[:, 1]

# Part (a) - Plot original data
plt.figure(figsize=(10,6))
plt.scatter(rho, Vp, color='green', label='Observed Data')
plt.xlabel('Density (g/cm³)')
plt.ylabel('P-wave Velocity (m/s)')
plt.title('Vp vs Density')
plt.grid(True)
plt.legend()
plt.show()

# Part (b) - Transform the data
#linearizing the model Vp = V0 * exp(k * rho) to ln(Vp) = ln(V0) + k * rho
# Taking the natural logarithm of Vp

ln_Vp = np.log(Vp)

#Constructing the Z matrix for linear regression
# Z = [1, rho] for the linear regression in transformed space

Z = np.column_stack((np.ones_like(rho), rho))  # [1, rho]

# Least squares solution
theta_hat = np.linalg.inv(Z.T @ Z) @ Z.T @ ln_Vp
a, k = theta_hat
V0 = np.exp(a)

# Predicted in log space and original space
ln_Vp_pred = Z @ theta_hat
Vp_pred = np.exp(ln_Vp_pred)

# Plot in transformed space
plt.figure(figsize=(10,6))
plt.scatter(rho, ln_Vp, color='purple', label='ln(Vp) Observed')
plt.plot(rho, ln_Vp_pred, color='orange', label='Best Fit Line')
plt.xlabel('Density (g/cm³)')
plt.ylabel('ln(P-wave Velocity)')
plt.title('Transformed Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Plot in original space
plt.figure(figsize=(10,6))
plt.scatter(rho, Vp, color='blue', label='Observed Vp')
plt.plot(rho, Vp_pred, color='red', label='Fitted Exponential')
plt.xlabel('Density (g/cm³)')
plt.ylabel('P-wave Velocity (m/s)')
plt.title('Vp vs Density with Fit')
plt.legend()
plt.grid(True)
plt.show()

# Part (d) - R^2 in transformed space
# Calculate R-squared

residuals = ln_Vp - ln_Vp_pred
r_squared = 1 - np.sum(residuals**2) / np.sum((ln_Vp - np.mean(ln_Vp))**2)

#Visualizing the residuals

plt.figure(figsize=(10,6))
plt.scatter(rho, residuals)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residuals (Transformed Space)")
plt.xlabel("Density")
plt.ylabel("Residual (lnVp - fit)")
plt.grid(True)
plt.show()

# Output results
print("\nQuestion 2 Answers:")
print(f"Estimated V0 = {V0:.3f} m/s")
print(f"Estimated k  = {k:.6f}")
print("\nQuestion 2d Answers:")
print(f"Coefficient of determination r² = {r_squared:.4f}")
print("Assumption check:")
if r_squared > 0.95:
    print("R² is high, indicating a strong exponential fit.")
else:
    print("R² is low — model may not fully capture variation in the data.")
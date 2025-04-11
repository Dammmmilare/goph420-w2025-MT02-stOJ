
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================
# Question 2 - Linearized Exponential Model
# =============================================

# Load data
data = np.loadtxt('Question_2_DATA_rho_vp.txt')
rho = data[:, 0]  # g/cm³
vp = data[:, 1]   # m/s

# Transform
log_vp = np.log(vp)

# Linear regression
slope, intercept, _, _, _ = stats.linregress(rho, log_vp)
k = slope
V0 = np.exp(intercept)

# Prediction
rho_range = np.linspace(min(rho), max(rho), 100)
vp_pred = V0 * np.exp(k * rho_range)

# Plot original
plt.figure(figsize=(10,6))
plt.scatter(rho, vp, label='Data')
plt.plot(rho_range, vp_pred, 'r-', label=f'Fit: V0={V0:.1f}, k={k:.4f}')
plt.xlabel('Density (g/cm³)'); plt.ylabel('Vp (m/s)')
plt.title('Question 2: Fit in Original Space')
plt.grid(); plt.legend(); plt.show()

# Plot transformed
plt.figure(figsize=(10,6))
plt.scatter(rho, log_vp, label='ln(Vp) Data')
plt.plot(rho_range, intercept + slope * rho_range, 'r-', label='Linear Fit')
plt.xlabel('Density (g/cm³)'); plt.ylabel('ln(Vp)')
plt.title('Question 2: Linear Fit in Transformed Space')
plt.grid(); plt.legend(); plt.show()

# R-squared
ss_total = np.sum((log_vp - np.mean(log_vp))**2)
ss_res = np.sum((log_vp - (intercept + slope * rho))**2)
r_squared = 1 - ss_res / ss_total

# Output
print(f"Estimated V0 = {V0:.2f} m/s")
print(f"Estimated k = {k:.4f}")
print(f"R² = {r_squared:.4f}")

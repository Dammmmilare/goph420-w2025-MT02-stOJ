import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# =============================================
# Question 1 - Linear Least Squares Regression
# =============================================

# Given data (mass in kg, terminal velocity in m/s)
mass = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
term_vel = np.array([10.45, 20.89, 31.34, 41.78, 52.23, 62.68, 73.12, 83.57, 94.01, 104.46])
g_earth = 9.81  # m/s²

# Transform: 1/v = (c/g)*(1/m)
y = 1 / term_vel
x = mass

# Linear regression through origin using stats.linregress
slope, _, _, _, _ = stats.linregress(x, y)
c = slope * g_earth

# Prediction
pred_term_vel = g_earth * mass / c

# Plot
plt.figure(figsize=(10,6))
plt.scatter(mass, term_vel, label='Observed Data')
plt.plot(mass, pred_term_vel, 'r-', label=f'Fit: c = {c:.2f} kg/s')
plt.xlabel('Mass (kg)'); plt.ylabel('Terminal Velocity (m/s)')
plt.legend(); plt.grid(); plt.title('Question 1: Terminal Velocity Fit')
plt.show()

# Residuals
residuals = term_vel - pred_term_vel

# Homoscedasticity and normality checks
_, pval_bp, _, _ = het_breuschpagan(residuals, sm.add_constant(mass))
_, pval_sw = shapiro(residuals)

# R-squared
ss_total = np.sum((term_vel - np.mean(term_vel))**2)
ss_res = np.sum(residuals**2)
r_squared = 1 - (ss_res / ss_total)

# Output
print(f"Estimated c = {c:.2f} kg/s")
print(f"Within design range? {46.5 <= c <= 47.5}")
print(f"Breusch-Pagan p-value = {pval_bp:.4f} (Homoscedasticity)")
print(f"Shapiro-Wilk p-value = {pval_sw:.4f} (Normality)")
print(f"R² = {r_squared:.6f}")

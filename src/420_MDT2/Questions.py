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

# Part a) Transform to linear form: 1/v = (c/g)*(1/m)
y = 1 / term_vel
x = mass

# Part b) Perform linear regression (force intercept=0)
slope, _, _, _, _ = stats.linregress(x, y)
c = slope * g_earth

print("Question 1b Results:")
print(f"Best fit c = {c:.2f} kg/s")
print(f"Target c = 47.0 ± 0.5 kg/s")
print(f"Within range? {46.5 <= c <= 47.5}")

# Part c) Plot best fit curve
pred_term_vel = g_earth * mass / c

plt.figure(figsize=(10,6))
plt.scatter(mass, term_vel, label='Data')
plt.plot(mass, pred_term_vel, 'r-', label=f'Fit (c={c:.2f} kg/s)')
plt.xlabel('Mass (kg)'); plt.ylabel('Terminal Velocity (m/s)')
plt.legend(); plt.grid(); plt.show()

# Residual analysis
residuals = term_vel - pred_term_vel

# Homoscedasticity check
_, pval_bp, _, _ = het_breuschpagan(residuals, sm.add_constant(mass))
print("\nQuestion 1c Homoscedasticity check:")
print(f"Breusch-Pagan p-value = {pval_bp:.4f}")

# Normality check
_, pval_sw = shapiro(residuals)
print("\nQuestion 1c Normality check:")
print(f"Shapiro-Wilk p-value = {pval_sw:.4f}")

# Part d) R-squared calculation
ss_total = np.sum((term_vel - np.mean(term_vel))**2)
ss_res = np.sum(residuals**2)
r_squared = 1 - (ss_res / ss_total)
print("\nQuestion 1d Results:")
print(f"R² = {r_squared:.6f}")


# =============================================
# Question 2 - Linearized Exponential Model
# =============================================

# Load data
data = np.loadtxt('Question_2_DATA_rho_vp.txt')
rho = data[:, 0]  # g/cm³
vp = data[:, 1]   # m/s

# Part a) Plot raw data
plt.figure(figsize=(10,6))
plt.scatter(rho, vp, alpha=0.6)
plt.xlabel('Density (g/cm³)'); plt.ylabel('Vp (m/s)')
plt.title('Question 2a: Raw Data'); plt.grid(); plt.show()

# Part b) Linearize Vp = V0*exp(k*ρ) -> ln(Vp) = ln(V0) + k*ρ
log_vp = np.log(vp)

plt.figure(figsize=(10,6))
plt.scatter(rho, log_vp, alpha=0.6)
plt.xlabel('Density (g/cm³)'); plt.ylabel('ln(Vp)')
plt.title('Question 2b: Transformed Data'); plt.grid(); plt.show()

# Part c) Perform linear regression
slope, intercept, _, _, _ = stats.linregress(rho, log_vp)
k = slope
V0 = np.exp(intercept)

print("\nQuestion 2c Results:")
print(f"V0 = {V0:.2f} m/s")
print(f"k = {k:.4f}")

# Plot fits
rho_range = np.linspace(min(rho), max(rho), 100)

# Original space
plt.figure(figsize=(10,6))
plt.scatter(rho, vp, alpha=0.6, label='Data')
plt.plot(rho_range, V0*np.exp(k*rho_range), 'r-',
         label=f'Fit: V0={V0:.1f}, k={k:.3f}')
plt.xlabel('Density (g/cm³)'); plt.ylabel('Vp (m/s)')
plt.title('Question 2c: Fit in Original Space')
plt.legend(); plt.grid(); plt.show()

# Transformed space
plt.figure(figsize=(10,6))
plt.scatter(rho, log_vp, alpha=0.6, label='Data')
plt.plot(rho_range, intercept + slope*rho_range, 'r-', label='Linear Fit')
plt.xlabel('Density (g/cm³)'); plt.ylabel('ln(Vp)')
plt.title('Question 2c: Fit in Transformed Space')
plt.legend(); plt.grid(); plt.show()

# Part d) R-squared for transformed model
ss_total = np.sum((log_vp - np.mean(log_vp))**2)
ss_res = np.sum((log_vp - (intercept + slope*rho))**2)
r_squared = 1 - (ss_res / ss_total)

print("\nQuestion 2d Results:")
print(f"R² = {r_squared:.4f}")


# =============================================
# Question 3 - Nonlinear Least Squares
# =============================================

def distance(a, x, y):
    return np.sqrt((a[0]-x)**2 + (a[1]-y)**2)

def jacobian(a, x0, y0, x1, y1):
    f0 = distance(a, x0, y0)
    f1 = distance(a, x1, y1)
    return np.array([[(a[0]-x0)/f0, (a[1]-y0)/f0],
                    [(a[0]-x1)/f1, (a[1]-y1)/f1]])

def objective(a, x0, y0, x1, y1, R0, R1):
    f0 = distance(a, x0, y0) - R0
    f1 = distance(a, x1, y1) - R1
    return f0**2 + f1**2

# Given parameters
x0, y0 = 0, 0
x1, y1 = 4, 4
R0 = 4
R1 = np.sqrt(5)

# Part b) Gradient Descent (h = -0.25)
print("\nQuestion 3b: Gradient Descent (h=-0.25)")
a = np.array([2.0, 4.0])
h = -0.25

print(f"{'Iter':<5}{'a_x':<10}{'a_y':<10}{'Error':<15}{'Rel Error':<15}")
for i in range(5):
    f0 = distance(a, x0, y0) - R0
    f1 = distance(a, x1, y1) - R1
    e = np.array([f0, f1])
    J = jacobian(a, x0, y0, x1, y1)
    g = -2 * J.T @ e
    a_new = a + h * g
    rel_error = np.linalg.norm(a_new - a)/np.linalg.norm(a)
    print(f"{i+1:<5}{a_new[0]:<10.6f}{a_new[1]:<10.6f}"
          f"{objective(a_new,x0,y0,x1,y1,R0,R1):<15.6f}{rel_error:<15.6f}")
    a = a_new

# Part c) Quasi-Newton (h = 1.0)
print("\nQuestion 3c: Quasi-Newton (h=1.0)")
a = np.array([4.0, 2.0])
h = 1.0

print(f"{'Iter':<5}{'a_x':<10}{'a_y':<10}{'Error':<15}{'Rel Error':<15}")
for i in range(5):
    f0 = distance(a, x0, y0) - R0
    f1 = distance(a, x1, y1) - R1
    e = np.array([f0, f1])
    J = jacobian(a, x0, y0, x1, y1)
    delta = np.linalg.inv(J.T @ J) @ J.T @ e
    a_new = a - h * delta
    rel_error = np.linalg.norm(a_new - a)/np.linalg.norm(a)
    print(f"{i+1:<5}{a_new[0]:<10.6f}{a_new[1]:<10.6f}"
          f"{objective(a_new,x0,y0,x1,y1,R0,R1):<15.6f}{rel_error:<15.6f}")
    a = a_new

# Part d) Method comparison
print("\nQuestion 3d: Comparison")
print("Quasi-Newton converges faster because:")
print("- Uses approximate Hessian (JᵀJ) for better step direction")
print("- Adapts to local curvature unlike gradient descent")
print("- Requires fewer iterations despite more computation per step")
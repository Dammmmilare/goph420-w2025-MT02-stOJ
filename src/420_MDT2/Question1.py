import numpy as np
import matplotlib.pyplot as plt

# Input data
# Masses (kg) and corresponding vf/g values
masses = np.array([5, 5, 10, 10, 25, 25, 50, 50, 100, 100, 150, 150])
vf_over_g = np.array([0.106, 0.109, 0.217, 0.214, 0.540, 0.520, 1.08, 1.04, 2.09, 2.11, 3.20, 3.13])


# Part a) Linear regression to find drag coefficient c
# Construct the Z matrix and y vector
Z = masses.reshape(-1, 1)
y = vf_over_g.reshape(-1, 1)


#part b) Calculate the drag coefficient c
# Least squares estimate: a = (Z^T Z)^-1 Z^T y
a_hat = np.linalg.inv(Z.T @ Z) @ Z.T @ y
a = a_hat[0, 0]
c_estimated = 1 / a

print("\nQuestion 1 Answers:")

# Check if within design spec (47 ± 0.5)
design_c = 47.0
within_spec = (design_c - 0.5) <= c_estimated <= (design_c + 0.5)
print("\nQuestion 1b Answers:")
print(f"Estimated drag coefficient c = {c_estimated:.3f} kg/s")
print(f"Within design spec? {'Yes' if within_spec else 'No'}")

# Predicted y values
y_pred = Z * a
residuals = y - y_pred
r_squared = 1 - (np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))


# Part c) Plotting the results
# Plotting
plt.scatter(masses, vf_over_g, label='Observed', color='blue')
plt.plot(masses, y_pred, label='Best Fit Line', color='red')
plt.xlabel('Mass (kg)')
plt.ylabel('vf / g (s)')
plt.title('Least Squares Fit for Drag Coefficient')
plt.legend()
plt.grid(True)
plt.show()
print("\nQuestion 1c Answers:")
print("\nAnswers for c is containd in the graph form the plot functions above")

# Part d) R-squared calculation
ss_total = np.sum((vf_over_g - np.mean(vf_over_g))**2)
ss_res = np.sum(residuals**2)
r_squared = 1 - (ss_res / ss_total)
print("\nQuestion 1d Answers:")
print(f"Value of R² = {r_squared:.4f}")
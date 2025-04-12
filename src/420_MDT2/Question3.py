# Question 3 - Nonlinear Least Squares
import numpy as np
import matplotlib.pyplot as plt 

# This code implements two optimization methods: Gradient Descent and Quasi-Newton method.
# It uses the following constants and functions to perform the optimization.

# Part (a) Constants
# The constants are defined as follows:
x0 = np.array([0.0, 0.0])
x1 = np.array([4.0, 4.0])
R0 = 4.0
R1 = np.sqrt(5)

# Distance function
def distance(a, x):
    return np.linalg.norm(a - x)

# Residual vector: difference from circle radii
def residual(a):
    return np.array([distance(a, x0) - R0,
                     distance(a, x1) - R1])

# Jacobian matrix: partial derivatives of distances
def jacobian(a):
    f0 = distance(a, x0)
    f1 = distance(a, x1)
    j0 = (a - x0) / f0
    j1 = (a - x1) / f1
    return np.vstack((j0, j1))

# Objective function: sum of squared residuals
def objective(a):
    e = residual(a)
    return np.sum(e**2)

print("Question 3a: steps in script form")
print("\n")
print("""# Distance function
def distance(a, x):
    return np.linalg.norm(a - x)

# Residual vector: difference from circle radii
def residual(a):
    return np.array([distance(a, x0) - R0,
                     distance(a, x1) - R1])

# Jacobian matrix: partial derivatives of distances
def jacobian(a):
    f0 = distance(a, x0)
    f1 = distance(a, x1)
    j0 = (a - x0) / f0
    j1 = (a - x1) / f1
    return np.vstack((j0, j1))

# Objective function: sum of squared residuals
def objective(a):
    e = residual(a)
    return np.sum(e**2)""")
print("\n")

# part (b) Gradient Descent
# This method uses the negative of the Jacobian matrix to update the solution.

def gradient_descent(a0, h, iterations):
    print("Question 3b: Gradient Descent (h = -0.25)")
    print(f"{'Iter':<5}{'a_x':<10}{'a_y':<10}{'Objective':<15}{'Rel Error':<15}")
    a = a0.copy()
    for i in range(iterations):
        Jk = jacobian(a)
        ek = residual(a)
        grad = -2 * Jk.T @ ek  # full gradient of the objective
        a_new = a + h * grad
        rel_error = np.linalg.norm(a_new - a) / np.linalg.norm(a)
        obj = objective(a_new)
        print(f"{i+1:<5}{a_new[0]:<10.6f}{a_new[1]:<10.6f}{obj:<15.6f}{rel_error:<15.6f}")
        a = a_new
    return a


#part (c) Quasi-Newton method
# This method uses the inverse of the Jacobian matrix to update the solution.
def quasi_newton(a0, h, iterations):
    print("\nQuestion 3c: Quasi-Newton (h = 1.0)")
    print(f"{'Iter':<5}{'a_x':<10}{'a_y':<10}{'Objective':<15}{'Rel Error':<15}")
    a = a0.copy()
    for i in range(iterations):
        Jk = jacobian(a)
        ek = residual(a)
        H = Jk.T @ Jk  # Approximate Hessian
        delta = np.linalg.inv(H) @ Jk.T @ ek
        a_new = a - h * delta
        rel_error = np.linalg.norm(a_new - a) / np.linalg.norm(a)
        obj = objective(a_new)
        print(f"{i+1:<5}{a_new[0]:<10.6f}{a_new[1]:<10.6f}{obj:<15.6f}{rel_error:<15.6f}")
        a = a_new
    return a

# Run Gradient Descent
a0_gd = np.array([2.0, 4.0])
gradient_descent(a0_gd, h=-0.25, iterations=5)

# Run Quasi-Newton
a0_qn = np.array([4.0, 2.0])
quasi_newton(a0_qn, h=1.0, iterations=5)

# Part (d) Comparison
print("\nQuestion 3d: Comparison Answers")
print("The Quasi-Newton method converges faster because:")

print("1. Achieves convergence in fewer steps, even though each step involves more calculations.")
print("2. Relies on an approximate second-derivative matrix (JᵀJ) to improve the optimization path")
print("3. Accounts for local changes in curvature, offering more adaptive updates than gradient descent")
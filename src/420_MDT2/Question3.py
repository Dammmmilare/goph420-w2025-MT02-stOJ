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

def f(a, xi):
    return np.linalg.norm(a - xi)

def J(a):
    f0 = f(a, x0)
    f1 = f(a, x1)
    j0 = (a - x0) / f0
    j1 = (a - x1) / f1
    return np.vstack((j0, j1))

def e(a):
    return np.array([f(a, x0) - R0, f(a, x1) - R1])
print("Question 3a: steps in script form")
print("\n")
print("""def f(a, xi):
    return np.linalg.norm(a - xi)

def J(a):
    f0 = f(a, x0)
    f1 = f(a, x1)
    j0 = (a - x0) / f0
    j1 = (a - x1) / f1
    return np.vstack((j0, j1))

def e(a):
    return np.array([f(a, x0) - R0, f(a, x1) - R1])""")
print("\n")

# part (b) Gradient Descent
# This method uses the negative of the Jacobian matrix to update the solution.

def gradient_descent(a0, h, iterations):
    a = a0.copy()
    print("Question 3b:Gradient Descent")
    for i in range(iterations):
        Jk = J(a)
        ek = e(a)
        delta = h * Jk.T @ ek
        a_new = a + delta
        rel_err = np.linalg.norm(a_new - a) / np.linalg.norm(a)
        print(f"Iter {i+1}: a = {a_new}, rel_error = {rel_err:.6f}")
        a = a_new
    return a


#part (c) Quasi-Newton method
# This method uses the inverse of the Jacobian matrix to update the solution.
def quasi_newton(a0, h, iterations):
    a = a0.copy()
    print("\nQuestion 3c: Quasi-Newton")
    for i in range(iterations):
        Jk = J(a)
        ek = e(a)
        H = Jk.T @ Jk
        delta = np.linalg.inv(H) @ Jk.T @ ek
        a_new = a - h * delta
        rel_err = np.linalg.norm(a_new - a) / np.linalg.norm(a)
        print(f"Iter {i+1}: a = {a_new}, rel_error = {rel_err:.6f}")
        a = a_new
    return a

# Run gradient descent
a0_gd = np.array([2.0, 4.0])
gradient_descent(a0_gd, h=-0.25, iterations=5)

# Run quasi-Newton
a0_qn = np.array([4.0, 2.0])
quasi_newton(a0_qn, h=1.0, iterations=5)

# Part (d) Comparison
print("\nQuestion 3d: Comparison Answers")
print("The Quasi-Newton method converges faster because:")

print("1. Achieves convergence in fewer steps, even though each step involves more calculations.")
print("2. Relies on an approximate second-derivative matrix (JᵀJ) to improve the optimization path")
print("3. Accounts for local changes in curvature, offering more adaptive updates than gradient descent")
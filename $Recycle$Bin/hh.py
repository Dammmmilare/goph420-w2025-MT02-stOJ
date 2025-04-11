# =============================================
# Question 3 - Nonlinear Least Squares
# =============================================
import numpy as np
import matplotlib.pyplot as plt

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
import numpy as np
import matplotlib.pyplot as plt

def quadratic_regression(x, y):
    # Fit a quadratic polynomial (degree 2)
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients

    # Print the coefficients
    print(f"Quadratic Equation: y = {a}x² + {b}x + {c}")

    # Compute the x-coordinate of the vertex
    x_min = -b / (2 * a)

    # Compute the y-coordinate of the vertex (minimum value)
    y_min = a * x_min**2 + b * x_min + c

    print(f"Minimum value at x = {x_min}, y = {y_min}")

    # Generate points for the quadratic curve
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = a * x_fit**2 + b * x_fit + c
    plt.plot(x_fit, y_fit, color='blue')
    plt.scatter([x_min], [y_min], color='black')
    return (x_min, y_min)

# Train loss data
x1 = np.array([72430, 441262, 3248302])
y1 = np.array([2.232, 2.1594, 2.2928])

x2 = np.array([441262, 3248302, 24397442])
y2 = np.array([1.5294, 1.4235, 2.1477])

x3 = np.array([3248302, 24397442, 201896110])
y3 = np.array([1.0111, 0.764, 2.1479])

data_dict = {
    "7.14E+11": (x1, y1),
    "4.35E+12": (x2, y2),
    "2.65E+13": (x3, y3),
}
opt_vals = {}
for budget, (x, y) in data_dict.items():
    plt.scatter(x, y, label=budget)
    opt_vals[budget] = quadratic_regression(x, y)

plt.title("IsoFLOP Profiles")
plt.xlabel("Parameters")
plt.ylabel("Training Loss")
plt.grid()
plt.legend()
plt.savefig('/home/sbiswas/class_projects/csc2541/CSC2541/isoflops.png')
plt.clf()

for budget, (x, y) in opt_vals.items():
    plt.scatter(budget, x)
plt.title("IsoFLOP Profiles")
plt.xlabel("FLOPs")
plt.ylabel("Parameters")
plt.savefig('/home/sbiswas/class_projects/csc2541/CSC2541/parameters.png')
plt.clf()

for budget, (x, y) in opt_vals.items():
    plt.scatter(budget, float(budget) / (x * 6.0))
plt.title("IsoFLOP Profiles")
plt.xlabel("FLOPs")
plt.ylabel("Tokens")
plt.savefig('/home/sbiswas/class_projects/csc2541/CSC2541/tokens.png')
plt.clf()
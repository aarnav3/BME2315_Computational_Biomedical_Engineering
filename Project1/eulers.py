import numpy as np
import matplotlib.pyplot as plt

def euler(f, t0, y0, h, n_steps):
    """
    Euler's Method for solving ODEs dy/dt = f(t, y)

    Parameters:
        f       : function f(t, y), derivative function
        t0      : initial time
        y0      : initial value of y at t0
        h       : step size
        n_steps : number of steps to take

    Returns:
        t_vals  : array of time values
        y_vals  : array of y values
    """
    t_vals = np.zeros(n_steps + 1)
    y_vals = np.zeros(n_steps + 1)

    # initial conditions
    t_vals[0] = t0
    y_vals[0] = y0

    for i in range(n_steps):
        y_vals[i+1] = y_vals[i] + h * f(t_vals[i], y_vals[i])
        t_vals[i+1] = t_vals[i] + h

    return t_vals, y_vals


# Example usage:
if __name__ == "__main__":
    # Define derivative function: dy/dt = 7*t^2 * (y-4)^(3/5)
    def f(t, y):
        return 7*t**2 * (y-4)**(3/5)

    # Parameters
    t0 = 0
    y0 = 4
    h = 0.05
    n_steps = 50

    # Run Euler's method
    t, y = euler(f, t0, y0, h, n_steps)

    # Print table header
    print(f"{'Step':<5}{'t':<10}{'y':<15}")
    print("-" * 30)

    # Print values
    for i in range(len(t)):
        print(f"{i:<5}{t[i]:<10.4f}{y[i]:<15.6f}")

    # Plot solution
    plt.plot(t, y, 'o-', label="Euler Approximation")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

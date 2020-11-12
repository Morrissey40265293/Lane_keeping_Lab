import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

v = 5
L = 2.3

u = -2. * np.pi / 180. # this turns the 2 degrees in radians - it is negative as rotation is clockwise

# Define the system dynamics as a function for equations 2.1a-c
def system_dynamics(t, z):
    theta = z[2]
    return [v * np.cos(theta),
            v * np.sin(theta),
            v * np.tan(u) / L]

t_final = 2
# initial condition is : z(0) = [x(0), y(0), theta(0)]
# Initial Variables, Angles and Resolution
θ_initial = 5. * np.pi / 180.  # 5 degrees in radians
z_initial = [0., 0.3, θ_initial]


# Simulate the dynamical system
solution = solve_ivp(system_dynamics,
                     [0, t_final],
                     z_initial,
                     t_eval = np.linspace(0, t_final, 1000))

times = solution.t
x_trajectory = solution.y[0]
y_trajectory = solution.y[1]
θ_trajectory = solution.y[2]

plt.plot(x_trajectory, y_trajectory)
plt.xlabel("x trajectory (m)")
plt.ylabel("y trajectory (m)")
plt.title("The Trajectory of a Vehicle over a Period of Time")
plt.show()


# Plot the graphs
plt.plot(times, x_trajectory.T)                      # x(t) against time graph
plt.grid()                                          # .T Transposes the vector
plt.xlabel('Time (s)')
plt.ylabel('X - Trajectory (m)')
plt.show()

plt.plot(times, y_trajectory.T)                      # y(t) against time graph
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Y - Trajectory (m)')
plt.show()

plt.plot(time, θ_trajectory.T)                  # Theta against time graph
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('θ-Trajectory (rad)')

plt.show()

plt.plot(x_trajectory, y_trajectory)                # y(t) against x(t) graph
plt.grid()
plt.xlabel('X - Trajectory (m)')
plt.ylabel('Y - Trajectory (m)')
plt.show()
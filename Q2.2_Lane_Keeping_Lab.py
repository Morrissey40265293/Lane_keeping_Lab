import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class car:

    def __init__(self, length=2.3, velocity=5, x=0, y=0, theta=0, disturbance=0):
        """
               :param length: The length between the two axles of the car
               :param velocity: The velocity of the car (m/s)
               :param disturbance: The additive disturbance (rad)
               :param x: The x-position of the car (m)
               :param y: The y-position of the car (m)
        """
        self.__length = length
        self.__velocity = velocity
        self.__x = x
        self.__y = y
        self.__theta = theta
        self.__disturbance = disturbance

    # Simulate the motion of the car from t = 0 to t = 0 + dt.
    def move(self, steering_angle, dt):
        """
                :param steering_angle: The steering angle of the car (rad)
                :param dt: dt is a time that is added to 0 s to produce the final time of the simulation (s)
                :return:
         """

        # Define the system dynamics as a function for equations 3.11a-c
        def system_dynamics(t, z):
            theta = z[2]
            return [self.__velocity * np.cos(theta),
                    self.__velocity * np.sin(theta),
                    self.__velocity * np.tan(steering_angle + self.__disturbance) / self.__length]

        z_initial = [self.__x, self.__y, self.__theta] # Starting from z_initial = [self.x, self.y, self.pose]

        # Simulate the dynamical system
        solution = solve_ivp(system_dynamics,
                             [0, dt],
                             z_initial)
        self.__x = solution.y[0][-1]
        self.__y = solution.y[1][-1]
        self.__theta = solution.y[2][-1]

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def theta(self):
        return self.__theta

    def length(self):
        return self.__length

    def velocity(self):
        return self.__velocity


class PIDController:

    def __init__(self, kp, ki, kd, ts):
        """
        :param kp: The proportional gain
        :param ki: The integral gain
        :param kd: The derivative gain
        :param ts: The sampling time
        """
        self.__kp = kp
        self.__ki = ki * ts
        self.__kd = kd / ts
        self.__ts = ts
        self.__previous_error = None

    def contol(self, y, y_set_point=0):
        """
        :param y: The y-position of the car
        :param y_set_point: The desired y-position of the car
        :return:
        """
        error = y_set_point - y  # Calculates the control error
        control_action = self.__kp * error

        if self.__previous_error is not None:
            control_action += self.__kd * (error - self.__previous_error)

        self.__previous_error = error # Means that next time, the previous error is needed
        return control_action

sampling_freq = 40
t_final = 20         # t [0, 50]
t_sampling = 1 / sampling_freq
num_points = t_final * sampling_freq
initial_y =0.3
initial_disturbance = 1. * np.pi / 180.  # 1 ° in radians, counter-clockwise direction therefore positive

pid = PIDController(kp=0.2, ki=0.0, kd=0, ts=t_sampling)
pid2 = PIDController(kp=0.2, ki=0.0, kd=0.1, ts=t_sampling)
pid3 = PIDController(kp=0.2, ki=0.0, kd=0.3, ts=t_sampling)
pid4 = PIDController(kp=0.2, ki=0.0, kd=0.9, ts=t_sampling)
malcolm = car(y=initial_y, disturbance=initial_disturbance)
malcolm2 = car(y=initial_y, disturbance=initial_disturbance)
malcolm3 = car(y=initial_y, disturbance=initial_disturbance)
malcolm4 = car(y=initial_y, disturbance=initial_disturbance)
y_cache = np.array([malcolm.y()])  # Inserted current first value of y into the cache
y_cache2 = np.array([malcolm2.y()])
y_cache3 = np.array([malcolm3.y()])
y_cache4 = np.array([malcolm4.y()])
x_cache = np.array([malcolm.x()])  # Inserted current first value of x into the cache
x_cache2 = np.array([malcolm2.x()])
x_cache3 = np.array([malcolm3.x()])
x_cache4 = np.array([malcolm4.x()])

for k in range(num_points):
    # Simulation of vehicle with kp = 0.2 and kd = 0.0
    control_action = pid.contol(y=malcolm.y())
    malcolm.move(control_action, t_sampling)
    y_cache = np.append(y_cache, malcolm.y())
    x_cache = np.append(x_cache, malcolm.x())

    # Simulation of vehicle with kp = 0.2 and kd = 0.1
    control_action2 = pid2.contol(y=malcolm2.y())
    malcolm2.move(control_action2, t_sampling)
    y_cache2 = np.append(y_cache2, malcolm2.y())
    x_cache2 = np.append(x_cache2, malcolm2.x())

    # Simulation of vehicle with kp = 0.1 and kd = 0.3
    control_action3 = pid3.contol(y=malcolm3.y())
    malcolm3.move(control_action3, t_sampling)
    y_cache3 = np.append(y_cache3, malcolm3.y())
    x_cache3 = np.append(x_cache3, malcolm3.x())

    # Simulation of vehicle with kp = 0.1 and kd = 0.9
    control_action4 = pid4.contol(y=malcolm4.y())
    malcolm4.move(control_action4, t_sampling)
    y_cache4 = np.append(y_cache4, malcolm4.y())
    x_cache4 = np.append(x_cache4, malcolm4.x())


t_span = t_sampling * np.arange(num_points + 1)
plt.plot(x_cache, y_cache, label="Kd = 0.0")
plt.plot(x_cache2, y_cache2, label="Kd = 0.1")
plt.plot(x_cache3, y_cache3, label="Kd = 0.3")
plt.plot(x_cache4, y_cache4, label="Kd = 0.9")
plt.xlabel("x trajectory (m)")
plt.ylabel("y trajectory (m)")
plt.grid()
plt.title("The Trajectory of a Vehicle controlled by a PD Controller")
plt.legend()
plt.show()
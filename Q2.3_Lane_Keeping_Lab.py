import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Car:

    def __init__(self, length=2.3, velocity=5., disturbance=0, x=0., y=0., pose=0.):
        """
        :param length: The length between the two axles of the car
        :param velocity: The velocity of the car (m/s)
        :param disturbance: The additive disturbance (rad)
        :param x: The x-position of the car (m)
        :param y: The y-position of the car (m)
        :param pose: The angle of the car from the y-setpoint (rad)
        """
        self.__length = length
        self.__velocity = velocity
        self.__disturbance = disturbance
        self.__x = x
        self.__y = y
        self.__pose = pose

    # Simulate the motion of the car from t = 0 to t = 0 + dt.
    def move(self, steering_angle, dt):
        """
        :param steering_angle: The steering angle of the car (rad)
        :param dt: dt is a time that is added to 0 s to produce the final time of the simulation (s)
        :return:
        """

        # Define the system dynamics as a function for equations 3.11a-c
        def bicycle_model(t, z):
            """
                       [v * cos(theta)]
             g(t,z) =  [v * sin(theta)]
                       [v * tan(u + w)/L]
            :param t: Time (s)
            :param z: An array that stores equations for x, y and pose
            :return:
            """
            theta = z[2]
            return [self.__velocity*np.cos(theta),
                    self.__velocity*np.sin(theta),
                    self.__velocity*np.tan(steering_angle+self.__disturbance)/self.__length]

        z_initial = [self.__x, self.__y, self.__pose]   # Starting from z_initial = [self.x, self.y, self.pose]
        solution = solve_ivp(bicycle_model,
                             [0, dt],
                             z_initial)
        self.__x = solution.y[0][-1]
        self.__y = solution.y[1][-1]
        self.__pose = solution.y[2][-1]

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def theta(self):
        return self.__pose


class PidController:

    def __init__(self, kp, kd, ki, ts):
        """
        :param kp: The proportional gain
        :param kd: The derivative gain
        :param ki: The integral gain
        :param ts: The sampling time
        """
        self.__kp = kp
        self.__kd = kd/ts
        self.__ki = ki*ts
        self.__ts = ts
        self.__previous_error = None                    # None i.e. 'Not defined yet'
        self.__sum_errors = 0.0
        self.control_action = 0.                        # Control action (steering angle) for steering_cache

    def control(self, y, y_set_point=0):
        """
        :param y: The y-position of the car
        :param y_set_point: The desired y-position of the car
        :return:
        """
        error = y_set_point - y                         # Calculates the control error
        control_action = self.__kp*error                # P control

        if self.__previous_error is not None:
            control_action += self.__kd*(error - self.__previous_error)  # D control

        control_action += self.__ki*self.__sum_errors   # I Control

        self.__sum_errors += error
        self.__previous_error = error                   # Means that next time we need the previous error
        self.control_action = control_action            # For steering_cache
        return control_action


# Initial Variables, Angles, Sampling rate and Ticks
sampling_rate = 40                                      # Sampling rate in Hz
t_final = 10                                            # t [0, 50]
x_initial = 0
y_initial = 0.3                                         # 0.3 m = 30 cm
theta_initial = np.deg2rad(0)                           # 0 ° in radians,
disturbance_initial = np.deg2rad(1)                     # 1 ° in radians, counter-clockwise direction therefore positive
sampling_period = 1/sampling_rate                       # Sampling period in s
ticks = sampling_rate*t_final                           # 40 Hz x 50 s = 2000

# Simulation of vehicle with kp = 0.9, kd = 0.5 and ki = 0.01
murphy = Car(x=x_initial, y=y_initial, pose=theta_initial, disturbance=disturbance_initial)
pid_1 = PidController(kp=0.9, kd=0.5, ki=0.01, ts=sampling_period)
y_cache = np.array([murphy.y()])  # Inserted current first value of y into the cache
x_cache = np.array([murphy.x()])  # Inserted current first value of x into the cache
steering_cache = np.array([pid_1.control_action])

for k in range(ticks):

    control_action = pid_1.control(murphy.y())
    murphy.move(control_action, sampling_period)
    y_cache = np.vstack((y_cache, [murphy.y()]))
    x_cache = np.vstack((x_cache, [murphy.x()]))
    steering_cache = np.vstack((steering_cache, [pid_1.control_action]))

    # Simulation of vehicle with kp = 0.9, kd = 0.5 and ki = 0.1. These are the ideal parameters.
    murphy_2 = Car(x=x_initial, y=y_initial, pose=theta_initial, disturbance=disturbance_initial)
    pid_2 = PidController(kp=0.9, kd=0.5, ki=0.1, ts=sampling_period)
    y_cache_2 = np.array([murphy_2.y()])
    x_cache_2 = np.array([murphy_2.x()])
    steering_cache_2 = np.array([pid_2.control_action])


for k in range(ticks):
    control_action_2 = pid_2.control(murphy_2.y())
    murphy_2.move(control_action_2, sampling_period)
    y_cache_2 = np.vstack((y_cache_2, [murphy_2.y()]))
    x_cache_2 = np.vstack((x_cache_2, [murphy_2.x()]))
    steering_cache_2 = np.vstack((steering_cache_2, [pid_2.control_action]))

    # Simulation of vehicle with kp = 0.9, kd = 0.5 and ki = 0.7
    murphy_3 = Car(x=x_initial, y=y_initial, pose=theta_initial, disturbance=disturbance_initial)
    pid_3 = PidController(kp=0.9, kd=0.5, ki=0.7, ts=sampling_period)
    y_cache_3 = np.array([murphy_3.y()])
    x_cache_3 = np.array([murphy_3.x()])
    steering_cache_3 = np.array([pid_3.control_action])


for k in range(ticks):
    control_action_3 = pid_3.control(murphy_3.y())
    murphy_3.move(control_action_3, sampling_period)
    y_cache_3 = np.vstack((y_cache_3, [murphy_3.y()]))
    x_cache_3 = np.vstack((x_cache_3, [murphy_3.x()]))
    steering_cache_3 = np.vstack((steering_cache_3, [pid_3.control_action]))

# Plot the graphs of the (x, y) Trajectories
plt.plot(x_cache, y_cache, label="K$_i$ = 0.01")
plt.plot(x_cache_2, y_cache_2, label="K$_i$ = 0.1")
plt.plot(x_cache_3, y_cache_3, label="K$_i$ = 0.7")
plt.grid()
plt.xlabel('X - Trajectory (m)')
plt.ylabel('Y - Trajectory (m)')
plt.legend()
plt.savefig("Question2Part3YAgainstXKpKdKi.svg", format="svg")
plt.show()

# Plot graphs of u(t) against time
t_span = sampling_period * np.arange(ticks + 1)
plt.plot(t_span, steering_cache, label="K$_i$ = 0.01")
plt.plot(t_span, steering_cache_2, label="K$_i$ = 0.1")
plt.plot(t_span, steering_cache_3, label="K$_i$ = 0.7")
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Steering Angle (rad)')
plt.legend()

plt.show()
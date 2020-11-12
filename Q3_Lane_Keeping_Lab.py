import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Car:

    def __init__(self, length=1, velocity=1., x=0., y=0., pose=0.):
        """
        :param length: The length between the two axles of the car
        :param velocity: The velocity of the car (m/s)
        :param x: The x-position of the car (m)
        :param y: The y-position of the car (m)
        :param pose: The angle of the car from the y-setpoint (rad)
        """
        self.__length = length
        self.__velocity = velocity
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

        # Define the system dynamics as a function for equations 4.1a-c
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
            return [self.__velocity * np.cos(theta),
                    self.__velocity * np.sin(theta),
                    self.__velocity * np.tan(steering_angle) / self.__length]

        z_initial = [self.__x, self.__y, self.__pose]  # Starting from z_initial = [self.x, self.y, self.pose]
        solution = solve_ivp(bicycle_model,
                             [0, dt],
                             z_initial,
                             t_eval=np.linspace(0, dt, num_points))
        self.__x = solution.y[0][-1]
        self.__y = solution.y[1][-1]
        self.__pose = solution.y[2][-1]

        return solution

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def theta(self):
        return self.__pose


# Initial variables, sampling rate and ticks
num_points = 1000
t_final = 1000                          # t [0, 1000]
steering_angle_initial = np.deg2rad(2)  # 2 ° in radians
velocity_initial1 = 1                   # Velocity in m/s
velocity_initial2 = 5
velocity_initial3 = 15
length_initial1 = 1                     # Length in m
length_initial2 = 2
length_initial3 = 3
theta_initial1 = np.deg2rad(0)          # theta in radians
theta_initial2 = np.deg2rad(45)
theta_initial3 = np.deg2rad(90)

murphy_velocity1 = Car(velocity=velocity_initial1)
murphy_velocity2 = Car(velocity=velocity_initial2)
murphy_velocity3 = Car(velocity=velocity_initial3)

murphy_length1 = Car(length=length_initial1)
murphy_length2 = Car(length=length_initial2)
murphy_length3 = Car(length=length_initial3)

murphy_theta1 = Car(pose=theta_initial1)
murphy_theta2 = Car(pose=theta_initial2)
murphy_theta3 = Car(pose=theta_initial3)

velocity_solution1 = murphy_velocity1.move(steering_angle_initial, t_final)
velocity_solution2 = murphy_velocity2.move(steering_angle_initial, t_final)
velocity_solution3 = murphy_velocity3.move(steering_angle_initial, t_final)

length_solution1 = murphy_length1.move(steering_angle_initial, t_final)
length_solution2 = murphy_length2.move(steering_angle_initial, t_final)
length_solution3 = murphy_length3.move(steering_angle_initial, t_final)

theta_solution1 = murphy_theta1.move(steering_angle_initial, t_final)
theta_solution2 = murphy_theta2.move(steering_angle_initial, t_final)
theta_solution3 = murphy_theta3.move(steering_angle_initial, t_final)

# Plot the graphs, showing effect of velocity, length and theta
plt.plot(velocity_solution1.y[0], velocity_solution1.y[1].T, label="Velocity = 1 m/s")
plt.plot(velocity_solution2.y[0], velocity_solution2.y[1].T, label="Velocity = 5 m/s")
plt.plot(velocity_solution3.y[0], velocity_solution3.y[1].T, label="Velocity = 15 m/s")
plt.grid()
plt.xlabel('X - Trajectory (m)')
plt.ylabel('T - Trajectory (m)')
plt.legend()
plt.savefig("Question3VelocityXAgainstTime.svg", format="svg")
plt.show()

plt.plot(length_solution1.y[0], length_solution1.y[1].T, label="Length = 1 m")
plt.plot(length_solution2.y[0], length_solution2.y[1].T, label="Length = 2 m")
plt.plot(length_solution3.y[0], length_solution3.y[1].T, label="Length = 3 m")
plt.grid()
plt.xlabel('X - Trajectory (m)')
plt.ylabel('Y - Trajectory (m)')
plt.legend()
plt.savefig("Question3LengthXAgainstTime.svg", format="svg")
plt.show()

plt.plot(theta_solution1.y[0], theta_solution1.y[1].T, label="Theta = 0 °")
plt.plot(theta_solution2.y[0], theta_solution2.y[1].T, label="Theta = 45 °")
plt.plot(theta_solution3.y[0], theta_solution3.y[1].T, label="Theta = 90 °")
plt.grid()
plt.xlabel('X - Trajectory (m)')
plt.ylabel('Y - Trajectory (m)')
plt.legend()
plt.savefig("Question3ThetaXAgainstTime.svg", format="svg")
plt.show()
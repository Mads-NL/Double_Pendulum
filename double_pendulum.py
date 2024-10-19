from manim import *
config.disable_caching = True
import numpy as np
from numba import jit

L1 = 1.0
L2 = 1.0
m1 = 1.0
m2 = 1.0
M = m1 + m2
g = 9.81
T = 15
dt = 0.015
NT = int(T/dt)
theta2Step = 0.01
theta1_initial = np.pi - 0.01
theta1Vel_initial = 0
theta2_initial = np.pi
theta2Vel_initial = 0

@jit
def compute_accelerations(theta1, theta1Vel, theta2, theta2Vel):
    """
    Compute the angular accelerations of a double pendulum system.

    Args:
        theta1 (float): Angle of the first pendulum (in radians).
        theta1Vel (float): Angular velocity of the first pendulum.
        theta2 (float): Angle of the second pendulum (in radians).
        theta2Vel (float): Angular velocity of the second pendulum.

    Returns:
        tuple: (theta1Accel, theta2Accel) where:
            - theta1Accel (float): Angular acceleration of the first pendulum.
            - theta2Accel (float): Angular acceleration of the second pendulum.
    """
    deltaTheta = theta1-theta2
    alpha = m1 + m2*np.sin(deltaTheta)*np.sin(deltaTheta)
    
    theta1Accel = ( -np.sin(deltaTheta) * (m2*L1*theta1Vel*theta1Vel*np.cos(deltaTheta) + m2*L2*theta2Vel*theta2Vel) - g*(M*np.sin(theta1) - m2*np.sin(theta2)*np.cos(deltaTheta)) ) / (L1*alpha)
    theta2Accel = ( np.sin(deltaTheta) * (M*L1*theta1Vel*theta1Vel + m2*L2*theta2Vel*theta2Vel*np.cos(deltaTheta)) + g*(M*np.sin(theta1)*np.cos(deltaTheta) - M*np.sin(theta2)) ) / (L2*alpha)

    return theta1Accel, theta2Accel

@jit
def rk4_step(theta1, theta1Vel, theta2, theta2Vel):
    """
    Perform a single RK4 step for a double pendulum system to update angles and velocities.

    Args:
        theta1 (float): Angle of the first pendulum (in radians).
        theta1Vel (float): Angular velocity of the first pendulum.
        theta2 (float): Angle of the second pendulum (in radians).
        theta2Vel (float): Angular velocity of the second pendulum.

    Returns:
        tuple: (theta1, theta1Vel, theta2, theta2Vel) where:
            - theta1 (float): Updated angle of the first pendulum.
            - theta1Vel (float): Updated angular velocity of the first pendulum.
            - theta2 (float): Updated angle of the second pendulum.
            - theta2Vel (float): Updated angular velocity of the second pendulum.
    """
    theta1Accel1, theta2Accel1 = compute_accelerations(theta1, theta1Vel, theta2, theta2Vel)
    k1_theta1 = theta1Vel 
    k1_theta2 = theta2Vel
    k1_theta1Vel = theta1Accel1 
    k1_theta2Vel = theta2Accel1 

    theta1_half = theta1 + 0.5 * k1_theta1 * dt 
    theta2_half = theta2 + 0.5 * k1_theta2 * dt 
    theta1Vel_half = theta1Vel + 0.5 * k1_theta1Vel * dt 
    theta2Vel_half = theta2Vel + 0.5 * k1_theta2Vel * dt 
    theta1Accel2, theta2Accel2 = compute_accelerations(theta1_half, theta1Vel_half, theta2_half, theta2Vel_half)
    k2_theta1 = theta1Vel_half 
    k2_theta2 = theta2Vel_half
    k2_theta1Vel = theta1Accel2 
    k2_theta2Vel = theta2Accel2 

    theta1_half = theta1 + 0.5 * k2_theta1 * dt
    theta2_half = theta2 + 0.5 * k2_theta2 * dt
    theta1Vel_half = theta1Vel + 0.5 * k2_theta1Vel * dt
    theta2Vel_half = theta2Vel + 0.5 * k2_theta2Vel * dt
    theta1Accel3, theta2Accel3 = compute_accelerations(theta1_half, theta1Vel_half, theta2_half, theta2Vel_half)
    k3_theta1 = theta1Vel_half
    k3_theta2 = theta2Vel_half
    k3_theta1Vel = theta1Accel3
    k3_theta2Vel = theta2Accel3

    theta1_end = theta1 + k3_theta1 * dt
    theta2_end = theta2 + k3_theta2 * dt
    theta1Vel_end = theta1Vel + k3_theta1Vel * dt
    theta2Vel_end = theta2Vel + k3_theta2Vel * dt
    theta1Accel4, theta2Accel4 = compute_accelerations(theta1_end, theta1Vel_end, theta2_end, theta2Vel_end)
    k4_theta1 = theta1Vel_end
    k4_theta2 = theta2Vel_end
    k4_theta1Vel = theta1Accel4
    k4_theta2Vel = theta2Accel4

    theta1 += (k1_theta1 + 2 * k2_theta1 + 2 * k3_theta1 + k4_theta1) * dt / 6
    theta2 += (k1_theta2 + 2 * k2_theta2 + 2 * k3_theta2 + k4_theta2) * dt / 6
    theta1Vel += (k1_theta1Vel + 2 * k2_theta1Vel + 2 * k3_theta1Vel + k4_theta1Vel) * dt / 6
    theta2Vel += (k1_theta2Vel + 2 * k2_theta2Vel + 2 * k3_theta2Vel + k4_theta2Vel) * dt / 6

    return theta1, theta1Vel, theta2, theta2Vel

@jit
def calculate_coordinates(theta1, theta2, L1, L2):
    """
    Calculate the positions of the pendulums' ends based on their angles.

    Args:
        theta1 (float): Angle of the first pendulum (in radians).
        theta2 (float): Angle of the second pendulum (in radians).
        L1 (float): Length of the first pendulum.
        L2 (float): Length of the second pendulum.

    Returns:
        tuple: ((x1, y1), (x2, y2)) where:
            - (x1, y1): Coordinates of the end of the first pendulum.
            - (x2, y2): Coordinates of the end of the second pendulum.
    """
    x1 = L1 * np.sin(theta1)  
    y1 = -L1 * np.cos(theta1) 
    x2 = x1 + L2 * np.sin(theta2)  
    y2 = y1 - L2 * np.cos(theta2)   
    return (x1, y1), (x2, y2)

@jit
def simulate_pendulum(theta1, theta1Vel, theta2, theta2Vel):
    """
    Simulate the motion of a double pendulum using RK4 integration.

    Args:
        theta1 (float): Initial angle of the first pendulum (in radians).
        theta1Vel (float): Initial angular velocity of the first pendulum.
        theta2 (float): Initial angle of the second pendulum (in radians).
        theta2Vel (float): Initial angular velocity of the second pendulum.

    Returns:
        tuple: (theta1Vals, theta1VelVals, theta2Vals, theta2VelVals, Pos1, Pos2) where:
            - theta1Vals (list): List of angles for the first pendulum over time.
            - theta1VelVals (list): List of angular velocities for the first pendulum over time.
            - theta2Vals (list): List of angles for the second pendulum over time.
            - theta2VelVals (list): List of angular velocities for the second pendulum over time.
            - Pos1 (list): List of 3D coordinates for the first pendulum end over time.
            - Pos2 (list): List of 3D coordinates for the second pendulum end over time.
    """
    theta1Vals = [0.0 for _ in range(NT)]
    theta2Vals = [0.0 for _ in range(NT)]
    theta1VelVals = [0.0 for _ in range(NT)]
    theta2VelVals = [0.0 for _ in range(NT)]
    Pos1 = [[0.0,0.0,0.0] for _ in range(NT)]
    Pos2 = [[0.0,0.0,0.0] for _ in range(NT)]

    theta1Vals[0] = theta1 
    theta2Vals[0] = theta2
    theta1VelVals[0] = theta1Vel 
    theta2VelVals[0] = theta2Vel

    (x1, y1), (x2, y2) = calculate_coordinates(theta1, theta2, L1, L2)
    Pos1[0] = [x1,y1,0] 
    Pos2[0] = [x2,y2,0]

    for i in range(1, NT):
        theta1, theta1Vel, theta2, theta2Vel = rk4_step(theta1, theta1Vel, theta2, theta2Vel)
        theta1Vals[i] = theta1 
        theta2Vals[i] = theta2
        theta1VelVals[i] = theta1Vel 
        theta2VelVals[i] = theta2Vel

        (x1, y1), (x2, y2) = calculate_coordinates(theta1, theta2, L1, L2)
        Pos1[i] = [x1,y1,0] 
        Pos2[i] = [x2,y2,0]

    return theta1Vals, theta1VelVals, theta2Vals, theta2VelVals, Pos1, Pos2 

def sim(NPend=11):
    """
    Simulate multiple double pendulums with slightly varied initial conditions.

    Args:
        NPend (int): Number of pendulums to simulate. Defaults to 11.

    Returns:
        list: A list of results for each pendulum simulation, where each result is a tuple
              containing (theta1Vals, theta1VelVals, theta2Vals, theta2VelVals, Pos1, Pos2).
    """
    initial_conditions = []

    initial_conditions.append((theta1_initial, theta1Vel_initial, theta2_initial, theta2Vel_initial))

    if NPend > 1:
        NOther = int((NPend-1)/2)
        for i in range(-NOther, NOther+1):
            if i != 0:
                theta2_new = theta2_initial + i*theta2Step 
                initial_conditions.append((theta1_initial, theta1Vel_initial, theta2_new, theta2Vel_initial))

    results = []
    for cond in initial_conditions:
        result = simulate_pendulum(*cond)
        results.append(result)

    return results

class Single(Scene):
    def construct(self):
        simResults = sim(NPend=1)

        d_eq = MathTex(r"\delta \theta = \theta_1 - \theta_2")
        a_eq = MathTex(r"\alpha = m_1 + m_2 \sin^2 (\delta \theta)")
        t1_eq = MathTex(
            r"\ddot{\theta}_1 = \frac{-\sin(\delta\theta) \left(m_2 L_1 \dot{\theta}_1^2 \cos(\delta\theta) + m_2 L_2 \dot{\theta}_2^2\right) - g \left(M \sin(\theta_1) - m_2 \sin(\theta_2) \cos(\delta\theta)\right)}{L_1 \alpha}"
        )
        t2_eq = MathTex(
            r"\ddot{\theta}_2 = \frac{\sin(\delta\theta) \left(M L_1 \dot{\theta}_1^2 + m_2 L_2 \dot{\theta}_2^2 \cos(\delta\theta)\right) + g \left(M \sin(\theta_1) \cos(\delta\theta) - M \sin(\theta_2)\right)}{L_2 \alpha}"
        )

        # Arrange them vertically
        equations = VGroup(d_eq, a_eq, t1_eq, t2_eq)
        equations.arrange(DOWN, aligned_edge=LEFT)
        equations.scale(0.5)

        equations.to_edge(UL)

        for eq in equations:
            self.play(Write(eq))

        p1 = simResults[0][-2]
        dot1 = Dot(point=p1[0], color=WHITE)
        dot2 = Dot(point=simResults[0][-1][0], color=WHITE)

        path = TracedPath(dot2.get_center, dissipating_time=1.5, stroke_opacity=[1])

        line1 = Line(ORIGIN, p1[0], color=WHITE)
        line2 = Line(p1[0], simResults[0][-1][0], color=WHITE)

        pendulum_group = Group(dot1, dot2, line1, line2)
        self.add(pendulum_group)
        self.add(path)

        self.i = 0
        max_steps = len(p1)

        def update_pendulum(mobj, dt):
            if self.i < max_steps: 
                line1.put_start_and_end_on(ORIGIN, p1[self.i])
                line2.put_start_and_end_on(p1[self.i], simResults[0][-1][self.i])
                dot1.move_to(p1[self.i])
                dot2.move_to(simResults[0][-1][self.i])
                self.i += 1

        pendulum_group.add_updater(update_pendulum)

        self.wait(max_steps * dt)

        pendulum_group.remove_updater(update_pendulum)

        self.wait(1)
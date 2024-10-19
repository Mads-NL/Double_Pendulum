# Double Pendulum Simulation with Manim

This project simulates the motion of a double pendulum using Manim and the RK4 numerical integration method. The double pendulum exhibits chaotic behavior, providing a compelling example of non-linear dynamics.

## Key Functions

- **compute_accelerations**: Calculates the angular accelerations of the pendulum.
- **rk4_step**: Performs a single RK4 step to update angles and velocities.
- **calculate_coordinates**: Computes the (x, y) coordinates of the pendulum's ends.
- **simulate_pendulum**: Simulates the motion of the double pendulum over time.
- **sim**: Simulates multiple double pendulums with slightly varied initial conditions.

## Methods Used

- **Runge-Kutta 4th Order (RK4)**: A numerical integration method used to solve ordinary differential equations, which is essential for simulating the pendulum's motion.
- **NumPy**: Utilized for mathematical computations and efficient array manipulations.
- **Numba**: An optimization library that speeds up the calculations through Just-In-Time (JIT) compilation, enhancing performance in numerical simulations.


## Demo

[![Watch the video](https://img.youtube.com/vi/f0Cs_SV4BaI/0.jpg)](https://youtu.be/f0Cs_SV4BaI)
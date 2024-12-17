import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =========================================================
# Toggle whether to use DOB or not
use_DOB = True

# System and Observer Parameters
m = 1.0    # True mass
c = 0.5    # True damping
k = 2.0    # True stiffness

# Nominal model parameters for DOB
m_nom = 1.0
c_nom = 0.5
k_nom = 2.0

dt = 0.01
tau = 0.1   # Q-filter time constant for DOB (larger for smoother estimate)
noise_std = 0.001  

# PD controller gains (reduced from 50.0 and 10.0 to 20 and 5)
Kp = 120.0
Kd = 16.0

# Filtering factors for velocity/acceleration
alpha = 0.7
beta = 0.7

def reference_position(t):
    # Sine wave between 0.5 and 1.0
    return 0.75 + 0.25 * np.sin(np.pi * t)

def disturbance(t):
    # Disturbance starts after t=2s: smaller amplitude to reduce chaos
    if t < 2.0:
        return 0.0
    else:
        random_noise = 0.1 * np.random.randn()
        return 3.3 * np.sin(3 * t) + random_noise

def measure_position(x_true):
    noise = np.random.randn() * noise_std
    return x_true + noise

def disturbance_observer(d_est, x_ddot_meas, x_meas, x_dot_meas, u):
    # Use measured/estimated states only (x_ddot_meas, x_dot_meas, x_meas)
    d_raw = m_nom * x_ddot_meas + c_nom * x_dot_meas + k_nom * x_meas - u
    d_est_dot = (d_raw - d_est) / tau
    d_est_new = d_est + d_est_dot * dt
    return d_est_new

# Simulation State
x = 0.0
x_dot = 0.0
d_est = 0.0
current_time = 0.0
last_x_meas = None
last_x_dot_meas = 0.0
x_dot_meas = 0.0
x_ddot_meas = 0.0

time_data = []
r_data = []
x_meas_data = []
d_true_data = []
d_est_data = []
u_data = []

# =========================================================
# Figure and Subplots Setup
# =========================================================
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])

ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.5, 0.5)
ax1.set_aspect('equal')
ax1.set_title("Mass-Spring-Damper System")
ax1.get_yaxis().set_visible(False)

wall_x = [-0.1, -0.1]
wall_y = [-0.2, 0.2]
ax1.plot(wall_x, wall_y, 'k', linewidth=5)

mass_width = 0.1
mass_height = 0.1
mass_patch = ax1.add_patch(plt.Rectangle((x - mass_width/2, -mass_height/2),
                                         mass_width, mass_height, fc='b'))
spring_line, = ax1.plot([], [], 'k-', lw=2)

ax3.set_title("Real-Time Position Tracking")
ax3.set_xlim(0, 5)
ax3.set_ylim(0.4, 1.1)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Position (m)")
line_r_small, = ax3.plot([], [], 'm-', label='Ref (r)')
line_x_meas_small, = ax3.plot([], [], 'b-', label='x_meas')
ax3.legend(loc='upper right')

ax2.set_xlim(0, 5)
ax2.set_ylim(-14.0, 14.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Values")

line_d_true, = ax2.plot([], [], 'r-', label='True Disturbance (d)')
line_d_est, = ax2.plot([], [], 'g--', label='Estimated Disturbance (d_est)')
line_u, = ax2.plot([], [], 'c-', label='Control Input (u)')
ax2.legend(loc='upper right')

def spring_shape(x_start, x_end, n_coils=5, amplitude=0.05):
    length = x_end - x_start
    coil_points = []
    for i in range(n_coils*2 + 1):
        frac = i/(n_coils*2)
        xp = x_start + length*frac
        yp = 0
        if i % 2 == 1:
            yp = amplitude if (i//2)%2==0 else -amplitude
        coil_points.append((xp, yp))
    return zip(*coil_points)

def init():

    line_d_true.set_data([], [])
    line_d_est.set_data([], [])
    line_u.set_data([], [])
    line_r_small.set_data([], [])
    line_x_meas_small.set_data([], [])
    return (mass_patch, spring_line,
            line_d_true, line_d_est, line_u, line_r_small, line_x_meas_small)

def update(frame):
    global x, x_dot, d_est, current_time
    global last_x_meas, last_x_dot_meas, x_dot_meas, x_ddot_meas

    current_time += dt
    t = current_time

    # Get reference and disturbance
    r = reference_position(t)
    d = disturbance(t)

    # Measure position (simulate sensor)
    x_meas = measure_position(x)

    # Estimate velocity and acceleration from position measurements
    if last_x_meas is None:
        x_dot_meas = 0.0
        x_ddot_meas = 0.0
    else:
        new_x_dot_meas = (x_meas - last_x_meas)/dt
        new_x_ddot_meas = (new_x_dot_meas - last_x_dot_meas)/dt

        # Filter velocity and acceleration
        x_dot_meas = alpha*new_x_dot_meas + (1-alpha)*x_dot_meas
        x_ddot_meas = beta*new_x_ddot_meas + (1-beta)*x_ddot_meas

    # PD control
    e = r - x_meas
    u_control = Kp*e - Kd*x_dot_meas

    # Use DOB to adjust input (uncomment for DOB effect)
    u = u_control - d_est

    # Update true plant states using true x and x_dot
    x_ddot = (u + d - c*x_dot - k*x)/m
    x_dot += x_ddot*dt
    x += x_dot*dt

    # Update DOB with measured acceleration, not true acceleration
    d_est = disturbance_observer(d_est, x_ddot_meas, x_meas, x_dot_meas, u)

    # Update memory
    last_x_meas = x_meas
    last_x_dot_meas = x_dot_meas

    # Store data
    time_data.append(t)
    r_data.append(r)
    x_meas_data.append(x_meas)
    d_true_data.append(d)
    d_est_data.append(d_est)
    u_data.append(u)

    # Update animation and plots
    mass_patch.set_x(x - mass_width/2)
    sx, sy = spring_shape(-0.1, x - mass_width/2, n_coils=5, amplitude=0.05)
    spring_line.set_data(sx, sy)

    #line_r.set_data(time_data, r_data)
    #line_x_meas.set_data(time_data, x_meas_data)
    line_d_true.set_data(time_data, d_true_data)
    line_d_est.set_data(time_data, d_est_data)
    line_u.set_data(time_data, u_data)

    line_r_small.set_data(time_data, r_data)
    line_x_meas_small.set_data(time_data, x_meas_data)

    if t > 5:
        ax2.set_xlim(t-5, t)
        ax3.set_xlim(t-5, t)

    return (mass_patch, spring_line, 
            line_d_true, line_d_est, line_u, line_r_small, line_x_meas_small)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=10, blit=True)
plt.tight_layout()
plt.show()

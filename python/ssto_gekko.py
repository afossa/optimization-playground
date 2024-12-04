"""
Constant-Thrust Lunar Ascent Trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

m = GEKKO()

# options
m.options.IMODE = 6  # dynamic simultaneous control mode
m.options.NODES = 3  # number of collocation nodes per time segment
m.options.MV_TYPE = 0  # zeroth-order hold control
m.options.SOLVER = 3  # IPOPT solver
m.options.SCALING = 1  # automatic scaling

m.solver_options = ["max_iter 1000"]

# physical constants
mu = 4.9028e12  # Moon standard gravitational parameter [m**3/s**2]
Rm = 1737.4e3  # Moon radius [m]
g0 = 9.80665  # sea-level gravity [m/s**2]
Isp = 450.0  # specific impulse [s]
twr = 2.1  # thrust/weight ratio [-]

# scaling parameters
lc = Rm
tc = np.sqrt((Rm * Rm * Rm) / mu)
vc = lc / tc

# scaled quantities
rf = 1.05  # target orbit radius [-]
vf = np.sqrt(1.0 / rf)  # target tangential velocity [-]

# time grid
nt = 30
tv = np.linspace(0.0, 1.0, nt)
m.time = tv

# scale time
tof = m.FV(0.5, lb=0.25, ub=0.75)  # fixed (i.e., scalar) variable
tof.STATUS = 1  # set as optimization variable

# constants
mdot = -twr / (Isp * g0 / vc)
thr = m.Const(twr)

# state variables
# r, theta, u, v, m = model.Array(model.Var, 5)
r = m.Var(value=np.linspace(1.0, rf, nt), lb=1.0)
theta = m.Var(value=np.linspace(0.0, np.pi / 18.0, nt), lb=0.0)
u = m.Var(value=np.linspace(0.0, 0.0, nt))
v = m.Var(value=np.linspace(0.0, vf, nt))
mass = m.Var(value=np.linspace(1.0, 0.7, nt), lb=0.0)

# control variables
alpha = m.MV(value=np.linspace(np.pi / 3.0, -np.pi / 6.0, nt), lb=-np.pi / 2.0)
alpha.STATUS = 1  # set as optimization variable
# alpha.DCOST = 0.01
# alpha.DMAX = 0.5

# Equations of Motion
m.Equation(r.dt() == tof * u)
m.Equation(theta.dt() == tof * (v / r))
m.Equation(u.dt() == tof * (-1.0 / (r * r) + (v * v) / r + twr / mass * m.sin(alpha)))
m.Equation(v.dt() == tof * (-(u * v) / r + twr / mass * m.cos(alpha)))
m.Equation(mass.dt() == tof * mdot)

# Terminal Constraints
m.fix_final(r, rf)
# m.fix_final(u, 0.0)
m.fix_final(v, vf)

# objective
p = np.zeros(nt)
p[-1] = 1.0
final = m.Param(value=p)

# m.Minimize(-mass * final)
m.Minimize(tof * final)

# solve problem
m.solve()

print(f"\nTime of Flight: {tc * tof.VALUE[-1]:.3f} s")
print(f"Final Mass:     {mass.VALUE[-1]:.3f} kg")

# %% plot
_, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

ax[0, 0].plot(tc * tof.VALUE * tv, Rm / 1e3 * np.asarray(r.value), marker="o")
ax[0, 0].grid(True)
ax[0, 0].set_xlabel(r"$t$ [s]")
ax[0, 0].set_ylabel(r"$r$ [km]")

ax[0, 1].plot(tc * tof.VALUE * tv, 180.0 / np.pi * np.asarray(theta.value), marker="o")
ax[0, 1].grid(True)
ax[0, 1].set_xlabel(r"$t$ [s]")
ax[0, 1].set_ylabel(r"$\theta$ [deg]")

ax[1, 0].plot(tc * tof.VALUE * tv, vc / 1e3 * np.asarray(u.value), marker="o")
ax[1, 0].grid(True)
ax[1, 0].set_xlabel(r"$t$ [s]")
ax[1, 0].set_ylabel(r"$u$ [km/s]")

ax[1, 1].plot(tc * tof.VALUE * tv, vc / 1e3 * np.asarray(v.value), marker="o")
ax[1, 1].grid(True)
ax[1, 1].set_xlabel(r"$t$ [s]")
ax[1, 1].set_ylabel(r"$v$ [km/s]")

_, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(
    tc * tof.VALUE * tv,
    180.0 / np.pi * np.asarray(alpha.value),
    color="tab:red",
    marker="o",
)
ax.grid(True)
ax.set_xlabel(r"$t$ [s]")
ax.set_ylabel(r"$\alpha$ [deg]")

plt.show()

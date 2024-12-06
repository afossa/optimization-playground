"""

Earth-Mars transfer taken from G. Lantoine and R. P. Russell, “A Hybrid
Differential Dynamic Programming Algorithm for Constrained Optimal Control
Problems. Part 2: Application,” J Optim Theory Appl, vol. 154, no. 2,
pp. 418–442, Aug. 2012, doi: 10.1007/s10957-012-0038-1.

"""

# %% imports

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gekko import GEKKO

# %% input parameters

# physical constants
gm_d = 1.3271244004127939e11 # Sun gravitational parameter [km^3/s^2]
au_d = 149597870.7 # astronomical unit [km]
d2s = 86400.0 # seconds per day [s]
g0 = 9.80665/1e3 # standard gravity [km/s^2]

# spacecraft parameters
thr_d = 0.5/1e3 # thrust magnitude [kg*km/s^2]
Isp_d = 2000.0 # specific impulse [s]
vex_d = Isp_d*g0 # exhaust velocity [m/s]
m0_d = 1000.0 # initial mass [kg]
m_dry_d = 500.0 # dry mass [kg]

# transfer parameters
tof_d = 348.79*d2s # time of flight [s]
N = 300  # number of stages

# scaling
lc = au_d # scale length [km]
tc = np.sqrt(lc**3/gm_d) # scale time [s]
vc = lc/tc # scale velocity [km/s]
mc = thr_d/vex_d*tc # scale mass such that the mass derivative is one at full thrust [kg]
rv = [lc,lc,lc,vc,vc,vc] # scale state [km, km, km, km/s, km/s, km/s]

# parameters in scaled units
gm = 1.0 # gravitational parameter [-]
thr = thr_d / (mc*vc/tc) # thrust magnitude [-]
vex = vex_d / vc # exhaust velocity [-]

# time span in scaled units
tof = tof_d / tc # time of flight [-]
t0 = 0.0 # initial time [-]
tf = t0 + tof # final time [-]
tv = np.linspace(t0, tf, N + 1) # time vector [-]

# %% initial guess from Lambert solution

# parse raw data
pth = Path(__file__).parent
data = np.loadtxt(os.path.join(pth, "earth_mars_lantoine_guess.txt"))

tv_l = data[:,0] # time vector [-]
rvv_l = data[:,1:7].T # spacecraft state vector [-]
rv_ert = data[:,7:10].T # Earth position during transfer [-]
rv_mrs = data[:,10:].T # Mars position during transfer [-]

uv_l = rvv_l[3:,:] / np.linalg.norm(rvv_l[3:,:], axis=0)

# boundary conditions in scaled units
m0 = m0_d / mc # initial mass [-]
m_dry = m_dry_d / mc # dry mass [-]
rv0 = rvv_l[:,0] # spacecraft state at departure [-]
rvf = rvv_l[:,-1] # spacecraft state at arrival [-]

# %% optimal control problem formulation

# model initialization
mdl = GEKKO()

# model options
mdl.options.IMODE = 6  # dynamic simultaneous control mode
mdl.options.NODES = 3  # number of collocation nodes per time segment
mdl.options.MV_TYPE = 0  # zeroth-order hold control
mdl.options.SOLVER = 3  # IPOPT solver
mdl.options.SCALING = 1  # automatic scaling

# solver options
mdl.solver_options = ["max_iter 3000"]
mdl.solver_options = ["print_user_options yes"]
mdl.solver_options = ["print_frequency_iter 10"]

# time grid
mdl.time = tv

# state variables
r = mdl.Array(mdl.SV, 3)
v = mdl.Array(mdl.SV, 3)
m = mdl.SV(value=np.linspace(m0, m_dry, N + 1), lb=m_dry, ub=m0)
for i in range(3):
    r[i].value = np.interp(tv, tv_l, rvv_l[i, :])
    v[i].value = np.interp(tv, tv_l, rvv_l[i + 3, :])

# control variables
u = mdl.Array(mdl.MV, 3, fixed_initial=False)
f = mdl.MV(value=np.zeros(N + 1), lb=0.0, ub=thr, fixed_initial=False)
for i in range(3):
    u[i].value = np.zeros(N + 1)
    u[i].STATUS = 1

# dynamics
for i in range(3):
    mdl.Equation(r[i].dt() == v[i])
    mdl.Equation(v[i].dt() == -gm / (r[0]**2 + r[1]**2 + r[2]**2)**(1.5) * r[i] + u[i] / m)
mdl.Equation(m.dt() == -f / vex)

# control constraints
mdl.Equation(u[0]**2 + u[1]**2 + u[2]**2 <= f**2)

# boundary conditions
for i in range(3):
    mdl.fix_final(r[i], rvf[i])
    mdl.fix_final(v[i], rvf[i + 3])

# objective
p = np.zeros(N + 1)
p[-1] = 1.0
final = mdl.Param(value=p)

# mdl.Minimize(-m * final)
mdl.Minimize(0.5 * mdl.integral(f**2))

# %% problem solution
try:
    mdl.solve(disp=True, debug=True)
except Exception as e:
    print(e)

# %% plots
_, ax = plt.subplots(1, 1)
ax.scatter(0.0, 0.0, color="gold", label="Sun")
ax.plot(rvv_l[0, :], rvv_l[1, :], color="blue", label="Lambert")
ax.plot(np.asarray(r[0].value), np.asarray(r[1].value), color="red", label="optimal")
ax.plot(rv_ert[0, :], rv_ert[1, :], color="tab:green", label="Earth")
ax.plot(rv_mrs[0, :], rv_mrs[1, :], color="tab:orange", label="Mars")
ax.grid(True)
ax.legend(loc=0)
ax.set_aspect("equal")
plt.show()
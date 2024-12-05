"""

Earth-Mars transfer taken from G. Lantoine and R. P. Russell, “A Hybrid
Differential Dynamic Programming Algorithm for Constrained Optimal Control
Problems. Part 2: Application,” J Optim Theory Appl, vol. 154, no. 2,
pp. 418–442, Aug. 2012, doi: 10.1007/s10957-012-0038-1.

"""

# %% imports

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

# %% input parameters

# physical constants
gm_d = 1.3271244004127939e11 # Sun gravitational parameter [km^3/s^2]
au_d = 149597870.7 # astronomical unit [km]
d2s = 86400.0 # seconds per day [s]
g0 = 9.80665 # standard gravity [m/s^2]

# spacecraft parameters
thr_d = 0.5 # thrust magnitude [N]
Isp_d = 2000.0 # specific impulse [s]
vex_d = Isp_d*g0 # exhaust velocity [m/s]
m0_d = 1000.0 # initial mass [kg]
m_dry_d = 500.0 # dry mass [kg]

# transfer parameters
tof_d = 348.79*d2s # time of flight [days]
N = 40 # number of stages

# scaling
lc = au_d # scale length [km]
tc = np.sqrt(lc**3/gm_d) # scale time [s]
vc = lc/tc # scale velocity [km/s]
mc = thr_d/vex_d*tc # scale mass such that the mass derivative is one at full thrust [kg]
rv = [lc,lc,lc,vc,vc,vc] # scale state [km, km, km, km/s, km/s, km/s]

# parameters in scaled units
gm = 1.0 # gravitational parameter [scaled]
thr = thr_d / (1e3*mc*vc/tc) # thrust magnitude [scaled]
vex = vex_d / (1e3*vc) # exhaust velocity [scaled]

# time span in scaled units
t0 = 0.0 # initial time [scaled]
tf = tof_d / tc # final time [scaled]
tof = tf - t0 # time of flight [scaled]
h = tof / N # time step [scaled]

# %% initial guess from Lambert solution

# parse raw data
data = np.loadtxt("earth_mars_lantoine_guess.txt")

tv_l = data[:,0] # time vector [scaled]
rvv_l = data[:,1:7].T # spacecraft state vector [scaled]
rv_ert = data[:,7:10].T # Earth position during transfer [scaled]
rv_mrs = data[:,10:].T # Mars position during transfer [scaled]

uv_l = rvv_l[3:,:] / np.linalg.norm(rvv_l[3:,:], axis=0)

# boundary conditions in scaled units
m0 = m0_d / mc # initial mass [scaled]
m_dry = m_dry_d / mc # dry mass [scaled]
rv0 = rvv_l[:,0] # spacecraft state at departure [scaled]
rvf = rvv_l[:,-1] # spacecraft state at arrival [scaled]

# %% optimal control problem formulation

# model initialization
mdl = GEKKO()

# model options
mdl.options.IMODE = 6  # dynamic simultaneous control mode
mdl.options.NODES = 3  # number of collocation nodes per time segment
mdl.options.MV_TYPE = 0  # zeroth-order hold control
mdl.options.SOLVER = 3  # IPOPT solver
mdl.options.SCALING = 1  # automatic scaling

# state variables
r = mdl.Array(mdl.SV, 3)
v = mdl.Array(mdl.SV, 3)
m = mdl.SV(value=np.linspace(m0, m_dry, N + 1), lb=m_dry, ub=m0)
for i in range(3):
    r[i].value = rvv_l[i, :]
    v[i].value = rvv_l[i + 3, :]

# control variables
u = mdl.Array(mdl.MV, 3)
t = mdl.MV(value=np.zeros(N + 1), lb=0.0, ub=thr)
for i in range(3):
    u[i].value = uv_l[i, :]
    u[i].STATUS = 1
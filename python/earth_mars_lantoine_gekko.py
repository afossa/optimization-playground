"""

Earth-Mars transfer taken from G. Lantoine and R. P. Russell, “A Hybrid
Differential Dynamic Programming Algorithm for Constrained Optimal Control
Problems. Part 2: Application,” J Optim Theory Appl, vol. 154, no. 2,
pp. 418–442, Aug. 2012, doi: 10.1007/s10957-012-0038-1.

"""

# %% imports

import os
import numpy as np
import scipy.integrate as spi
import spiceypy as sp
from gekko import GEKKO

import matplotlib.pyplot as plt

from pathlib import Path
from lambert import lambert_solver

# %% SPICE kernels

pth = Path(__file__).parent.parent
sp.furnsh(os.path.join(pth, "kernels/naif0012.tls"))
sp.furnsh(os.path.join(pth, "kernels/gm_de440.tpc"))
sp.furnsh(os.path.join(pth, "kernels/de440.bsp"))

# %% input parameters

# problem type: 0 = energy optimal, 1 = fuel optimal
obj_id = 0
obj_nm = "energy" if obj_id == 0 else "fuel"

# physical constants
gm_d = sp.bodvcd(10, "GM", 1)[1][0]  # Sun gravitational parameter [km^3/s^2]
au_d = 149597870.7  # astronomical unit [km]
d2s = 86400.0  # seconds per day [s]
g0 = 9.80665 / 1e3  # standard gravity [km/s^2]

# spacecraft parameters
thr_max_d = 0.5 / 1e3  # maximum thrust magnitude [kg*km/s^2]
isp_d = 2000.0  # specific impulse [s]
vex_d = isp_d * g0  # exhaust velocity [m/s]
m0_d = 1000.0  # initial mass [kg]
m_dry_d = 500.0  # dry mass [kg]

# transfer parameters
tof_d = 348.79 * d2s  # time of flight [s]
nb_sg = 40  # number of segments
order = 3  # transcription order
et0_d = sp.str2et("2007 APR 10 12:00:00.000")  # initial epoch [s]
etf_d = et0_d + tof_d  # final epoch [s]

# boundary conditions, Earth at departure, Mars at arrival [km, km/s]
rv0_ert_d = sp.spkgeo(3, et0_d, "ECLIPJ2000", 10)[0]
rvf_mrs_d = sp.spkgeo(4, etf_d, "ECLIPJ2000", 10)[0]

# scaling
lc = au_d  # scale length [km]
tc = np.sqrt(lc**3 / gm_d)  # scale time [s]
vc = lc / tc  # scale velocity [km/s]
rv = [lc, lc, lc, vc, vc, vc]  # scale state [km, km, km, km/s, km/s, km/s]

# scale mass such that the mass derivative is one at full thrust [kg]
mc = thr_max_d / vex_d * tc

# parameters in scaled units
gm = 1.0  # gravitational parameter [-]
thr_max = thr_max_d / (mc * vc / tc)  # maximum thrust magnitude [-]
vex = vex_d / vc  # exhaust velocity [-]

# boundary conditions in scaled units
m0 = m0_d / mc  # initial mass [-]
m_dry = m_dry_d / mc  # dry mass [-]
rv0 = rv0_ert_d / rv  # spacecraft state at departure [-]
rvf = rvf_mrs_d / rv  # spacecraft state at arrival [-]

# time span in scaled units
tof = tof_d / tc  # time of flight [-]
t0 = 0.0  # initial time [-]
tf = t0 + tof  # final time [-]

# %% initial guess from Lambert solution

# Lambert problem solution
sol_l = lambert_solver(rv0[0:3], rvf[0:3], tof, gm, 0)

# Lambert Δv
dv0_l = sol_l[0][0] - rv0[3:6]
dvf_l = rvf[3:6] - sol_l[1][0]

# Lambert arc sampling
rv0_l = np.concatenate((rv0[0:3], sol_l[0][0]))
tv_l = np.linspace(t0, tf, nb_sg + 1)


def kepler(_, y, mu):
    """Differential equations for the Keplerian motion."""
    r = np.linalg.norm(y[0:3])
    return np.concatenate((y[3:6], -mu / r**3 * y[0:3]))


sol_l = spi.solve_ivp(
    kepler, [t0, tf], rv0_l, method="DOP853", t_eval=tv_l, args=(gm,)
)
rvv_l = sol_l.y

# fix boundary conditions
rvv_l[:, 0] = rv0
rvv_l[:, -1] = rvf

# tangential thrust direction
uv_l = rvv_l[3:6, :] / np.linalg.norm(rvv_l[3:6, :], axis=0)

# Earth and Mars positions sampling
etv_d = et0_d + tv_l * tc
rv_ert = np.empty((3, etv_d.size))
rv_mrs = np.empty((3, etv_d.size))

for i, et in enumerate(etv_d):
    rv_ert[:, i] = sp.spkgps(3, et, "ECLIPJ2000", 10)[0] / lc
    rv_mrs[:, i] = sp.spkgps(4, et, "ECLIPJ2000", 10)[0] / lc

# %% optimal control problem formulation

# model initialization
mdl = GEKKO()

# model options
mdl.options.IMODE = 6  # dynamic simultaneous control mode
mdl.options.NODES = 5  # number of collocation nodes per time segment
mdl.options.MV_TYPE = 0  # zeroth-order hold control
mdl.options.SOLVER = 3  # IPOPT solver
mdl.options.SCALING = 1  # automatic scaling

# solver options
mdl.solver_options = ["max_iter 1000"]
mdl.solver_options = ["linear_solver ma57"]
mdl.solver_options = ["print_user_options yes"]
mdl.solver_options = ["print_frequency_iter 10"]

# time grid
tv = np.linspace(t0, tf, nb_sg + 1)
mdl.time = tv

# state variables
r = mdl.Array(mdl.SV, 3)
v = mdl.Array(mdl.SV, 3)
m = mdl.SV(value=np.linspace(m0, m_dry, nb_sg + 1), lb=m_dry, ub=m0)
for i in range(3):
    r[i].value = np.interp(tv, tv_l, rvv_l[i, :])
    v[i].value = np.interp(tv, tv_l, rvv_l[i + 3, :])

# control variables
thr = mdl.MV(value=np.zeros(nb_sg + 1), lb=0.0, ub=thr_max, fixed_initial=False)
thr.STATUS = 1
u = mdl.Array(mdl.MV, 3, fixed_initial=False)
for i in range(3):
    u[i].value = np.zeros(nb_sg + 1)
    u[i].STATUS = 1

# dynamics
for i in range(3):
    mdl.Equation(r[i].dt() == v[i])
    mdl.Equation(v[i].dt() == -gm / (r[0]**2 + r[1]**2 + r[2]**2)**(1.5) * r[i] + u[i] / m)
mdl.Equation(m.dt() == -thr / vex)

# control constraints
mdl.Equation(u[0]**2 + u[1]**2 + u[2]**2 <= thr**2)

# boundary conditions
for i in range(3):
    mdl.fix_final(r[i], rvf[i])
    mdl.fix_final(v[i], rvf[i + 3])

# objective
p = np.zeros(nb_sg + 1)
p[-1] = 1.0
final = mdl.Param(value=p)

if obj_id == 0:
    mdl.Minimize(0.5 * mdl.integral(thr**2))
else:
    mdl.Minimize(-m * final)

# %% problem solution
try:
    mdl.solve(disp=True, debug=True)
except Exception as e:
    print(e)

xv_o = np.asarray(r[0].value)
yv_o = np.asarray(r[1].value)
mv_o = np.asarray(m.value)

print(f"Problem type:   {obj_nm:s} optimal")
print(f"Remaining fuel: {mc * (mv_o[-1] - m_dry):.3f} kg")

# %% plots

_, ax = plt.subplots(1, 1)
ax.scatter(0.0, 0.0, color="gold", label="Sun")
ax.plot(rvv_l[0, :], rvv_l[1, :], color="blue", label="Lambert")
ax.plot(xv_o, yv_o, color="red", label="optimal")
ax.plot(rv_ert[0, :], rv_ert[1, :], color="tab:green", label="Earth")
ax.plot(rv_mrs[0, :], rv_mrs[1, :], color="tab:orange", label="Mars")
ax.grid(True)
ax.legend(loc=0)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$ [AU]")
ax.set_ylabel(r"$y$ [AU]")
ax.set_title("Earth-Mars transfer")

plt.show()

# %%

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
import dymos as dm
import openmdao.api as om
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
nb_sg = 120  # number of segments
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
class ODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("mu", types=float)
        self.options.declare("vex", types=float)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("x", val=np.zeros(nn), units="km")
        self.add_input("y", val=np.zeros(nn), units="km")
        self.add_input("z", val=np.zeros(nn), units="km")
        self.add_input("vx", val=np.zeros(nn), units="km/s")
        self.add_input("vy", val=np.zeros(nn), units="km/s")
        self.add_input("vz", val=np.zeros(nn), units="km/s")
        self.add_input("m", val=np.zeros(nn), units="kg")
        self.add_input("thr", val=np.zeros(nn), units="kg*km/s/s")
        self.add_input("ux", val=np.zeros(nn), units="kg*km/s/s")
        self.add_input("uy", val=np.zeros(nn), units="kg*km/s/s")
        self.add_input("uz", val=np.zeros(nn), units="kg*km/s/s")

        self.add_output("xdot", val=np.zeros(nn), units="km/s")
        self.add_output("ydot", val=np.zeros(nn), units="km/s")
        self.add_output("zdot", val=np.zeros(nn), units="km/s")
        self.add_output("vxdot", val=np.zeros(nn), units="km/s/s")
        self.add_output("vydot", val=np.zeros(nn), units="km/s/s")
        self.add_output("vzdot", val=np.zeros(nn), units="km/s/s")
        self.add_output("mdot", val=np.zeros(nn), units="kg/s")

        self.add_output("endot", val=np.zeros(nn), units="kg**2*km**2/s**4")

        self.declare_coloring(wrt="*", method="cs", show_sparsity=False)

    def compute(self, inputs, outputs):
        mu = self.options["mu"]
        vex = self.options["vex"]

        x = inputs["x"]
        y = inputs["y"]
        z = inputs["z"]
        vx = inputs["vx"]
        vy = inputs["vy"]
        vz = inputs["vz"]
        m = inputs["m"]

        thr = inputs["thr"]
        ux = inputs["ux"]
        uy = inputs["uy"]
        uz = inputs["uz"]

        c = -mu / (x**2 + y**2 + z**2) ** (1.5)

        outputs["xdot"] = vx
        outputs["ydot"] = vy
        outputs["zdot"] = vz
        outputs["vxdot"] = c * x + ux / m
        outputs["vydot"] = c * y + uy / m
        outputs["vzdot"] = c * z + uz / m

        outputs["mdot"] = -thr / vex
        outputs["endot"] = 0.5 * thr**2


# optimal control problem
pbm = om.Problem(model=om.Group(), reports=None)
pbm.model.linear_solver = om.DirectSolver()
pbm.driver = om.pyOptSparseDriver()

pbm.driver.options["optimizer"] = "IPOPT"
pbm.driver.options["print_results"] = False
pbm.driver.declare_coloring(show_summary=False, show_sparsity=False)

pbm.driver.opt_settings["tol"] = 1e-12
pbm.driver.opt_settings["acceptable_tol"] = 1e-10
pbm.driver.opt_settings["linear_solver"] = "ma57"
pbm.driver.opt_settings["nlp_scaling_method"] = "gradient-based"
pbm.driver.opt_settings["ma57_automatic_scaling"] = "yes"
pbm.driver.opt_settings["print_level"] = 5
pbm.driver.opt_settings["print_frequency_iter"] = 10
pbm.driver.opt_settings["print_user_options"] = "yes"
pbm.driver.opt_settings["print_timing_statistics"] = "yes"

# trajectory and phase
traj = dm.Trajectory()
phase = dm.Phase(
    ode_class=ODE,
    ode_init_kwargs={"mu": gm, "vex": vex},
    transcription=dm.GaussLobatto(
        num_segments=nb_sg, order=order, compressed=True
    ),
)

traj.add_phase("phase0", phase)
pbm.model.add_subsystem("traj", traj)

# time, state, and control variables
phase.set_time_options(fix_initial=True, fix_duration=True, units="s")

phase.add_state(
    "x", fix_initial=True, fix_final=True, rate_source="xdot", units="km"
)
phase.add_state(
    "y", fix_initial=True, fix_final=True, rate_source="ydot", units="km"
)
phase.add_state(
    "z", fix_initial=True, fix_final=True, rate_source="zdot", units="km"
)
phase.add_state(
    "vx", fix_initial=True, fix_final=True, rate_source="vxdot", units="km/s"
)
phase.add_state(
    "vy", fix_initial=True, fix_final=True, rate_source="vydot", units="km/s"
)
phase.add_state(
    "vz", fix_initial=True, fix_final=True, rate_source="vzdot", units="km/s"
)
phase.add_state(
    "m", fix_initial=True, lower=m_dry, upper=m0, rate_source="mdot", units="kg"
)
phase.add_state(
    "en", fix_initial=True, rate_source="endot", units="kg**2*km**2/s**3"
)

phase.add_control(
    "thr", lower=0.0, upper=thr_max, targets=["thr"], units="kg*km/s/s"
)
phase.add_control("ux", targets=["ux"], units="kg*km/s/s")
phase.add_control("uy", targets=["uy"], units="kg*km/s/s")
phase.add_control("uz", targets=["uz"], units="kg*km/s/s")

# path constraints
phase.add_path_constraint(
    "pc=ux**2 + uy**2 + uz**2 - thr**2", upper=0.0, units="kg**2*km**2/s**4"
)

# objective function
if obj_id == 0:
    phase.add_objective("en", loc="final")
else:
    phase.add_objective("m", loc="final", scaler=-1.0)

# problem setup
pbm.setup(check=True)

# initial guess
phase.set_time_val(initial=t0, duration=tof)
phase.set_state_val("x", rvv_l[0, :], tv_l)
phase.set_state_val("y", rvv_l[1, :], tv_l)
phase.set_state_val("z", rvv_l[2, :], tv_l)
phase.set_state_val("vx", rvv_l[3, :], tv_l)
phase.set_state_val("vy", rvv_l[4, :], tv_l)
phase.set_state_val("vz", rvv_l[5, :], tv_l)
phase.set_state_val("m", [m0, m_dry])
phase.set_state_val("en", [0.0, 0.0])
phase.set_control_val("thr", [0.0, 0.0])
phase.set_control_val("ux", [0.0, 0.0])
phase.set_control_val("uy", [0.0, 0.0])
phase.set_control_val("uz", [0.0, 0.0])

# %% problem solution and explicit simulation
dm.run_problem(
    pbm,
    run_driver=True,
    simulate=True,
    solution_record_file=None,
    simulation_record_file=None,
)

tv_o = pbm.get_val("traj.phases.phase0.timeseries.time")

mv_o = pbm.get_val("traj.phases.phase0.timeseries.m")
xv_o = pbm.get_val("traj.phases.phase0.timeseries.x")
yv_o = pbm.get_val("traj.phases.phase0.timeseries.y")

uxv_o = pbm.get_val("traj.phases.phase0.timeseries.ux")
uyv_o = pbm.get_val("traj.phases.phase0.timeseries.uy")
uzv_o = pbm.get_val("traj.phases.phase0.timeseries.uz")
thr_o = pbm.get_val("traj.phases.phase0.timeseries.thr")
uvm_o = np.sqrt(uxv_o**2 + uyv_o**2 + uzv_o**2)

env_o = pbm.get_val("traj.phases.phase0.timeseries.en")
env_i = spi.cumulative_trapezoid(
    0.5 * thr_o.ravel() ** 2, tv_o.ravel(), initial=0.0
)

print(f"Problem type:   {obj_nm:s} optimal")
print(f"Remaining fuel: {mc * (mv_o[-1, -1] - m_dry):.3f} kg")

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

_, ax2 = plt.subplots(1, 1)
ax2.plot(
    (tc / d2s) * tv_o,
    (1e3 * mc * vc / tc) * thr_o,
    color="red",
    label=r"$T(t)$",
)
ax2.plot(
    (tc / d2s) * tv_o,
    (1e3 * mc * vc / tc) * uvm_o,
    color="blue",
    linestyle="--",
    label=r"$||u(t)||_2$",
)
ax2.grid(True)
ax2.legend(loc=0)
ax2.set_xlabel(r"$t$ [days]")
ax2.set_ylabel(r"$T$ [N]")
ax2.set_title("Thrust profile")

_, ax3 = plt.subplots(1, 1)
ax3.plot(
    (tc / d2s) * tv_o,
    ((mc**2) * (lc**2) / (tc**3)) * env_o,
    color="red",
    label="state variable",
)
ax3.plot(
    (tc / d2s) * tv_o,
    ((mc**2) * (lc**2) / (tc**3)) * env_i,
    color="blue",
    linestyle="--",
    label="quadrature",
)
ax3.grid(True)
ax3.legend(loc=0)
ax3.set_xlabel(r"$t$ [days]")
ax3.set_ylabel(r"$\Gamma$ [$kg^2\cdot km^2/s^3$]")
ax3.set_title(r"Energy integral $\Gamma=\dfrac{1}{2}\int T^2(t)\text{d}t$")

plt.show()

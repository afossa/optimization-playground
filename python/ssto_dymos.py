"""
Single Stage to Orbit (SSTO) Trajectory.
"""

import dymos as dm
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om


class ODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("mu", types=float)
        self.options.declare("thr", types=float)
        self.options.declare("mdot", types=float)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("r", val=np.zeros(nn), units="m")
        self.add_input("theta", val=np.zeros(nn), units="rad")
        self.add_input("u", val=np.zeros(nn), units="m/s")
        self.add_input("v", val=np.zeros(nn), units="m/s")
        self.add_input("m", val=np.zeros(nn), units="kg")
        self.add_input("alpha", val=np.zeros(nn), units="rad")

        self.add_output("rdot", val=np.zeros(nn), units="m/s")
        self.add_output("thetadot", val=np.zeros(nn), units="rad/s")
        self.add_output("udot", val=np.zeros(nn), units="m/s/s")
        self.add_output("vdot", val=np.zeros(nn), units="m/s/s")
        self.add_output("mdot", val=np.zeros(nn), units="kg/s")

        self.declare_coloring(wrt="*", method="cs", show_sparsity=False)

    def compute(self, inputs, outputs):
        mu = self.options["mu"]
        thr = self.options["thr"]

        r = inputs["r"]
        u = inputs["u"]
        v = inputs["v"]
        m = inputs["m"]
        alpha = inputs["alpha"]

        outputs["rdot"] = u
        outputs["thetadot"] = v / r
        outputs["udot"] = -mu / (r * r) + (v * v) / r + thr / m * np.sin(alpha)
        outputs["vdot"] = -(u * v) / r + thr / m * np.cos(alpha)
        outputs["mdot"] = self.options["mdot"]


# %% Parameters
nb_seg = 20
order = 3

gm = 4.9028e12
rm = 1737.4e3

tc = np.sqrt(rm**3 / gm)
vc = rm / tc

twr = 2.1
isp = 450.0
g0 = 9.80665
mdot = -twr / (isp * g0 / vc)

tof = 0.5
rf = 1.05
vf = np.sqrt(1.0 / rf)

# %% Optimal Control Problem
pbm = om.Problem(model=om.Group(), reports=None)
pbm.model.linear_solver = om.DirectSolver()
pbm.driver = om.pyOptSparseDriver()

pbm.driver.options["optimizer"] = "IPOPT"
pbm.driver.options["print_results"] = False
pbm.driver.declare_coloring(show_summary=False, show_sparsity=False)

pbm.driver.opt_settings["tol"] = 1e-10
pbm.driver.opt_settings["acceptable_tol"] = 1e-8
pbm.driver.opt_settings["print_level"] = 5
pbm.driver.opt_settings["output_file"] = ""
pbm.driver.opt_settings["print_timing_statistics"] = "yes"
pbm.driver.opt_settings["linear_solver"] = "ma57"
pbm.driver.opt_settings["nlp_scaling_method"] = "gradient-based"
pbm.driver.opt_settings["ma57_automatic_scaling"] = "yes"

# %% Phase and Trajectory
traj = dm.Trajectory()
phase = dm.Phase(
    ode_class=ODE,
    ode_init_kwargs={"mu": 1.0, "thr": twr, "mdot": mdot},
    transcription=dm.GaussLobatto(num_segments=nb_seg, order=order, compressed=True),
)

traj.add_phase("phase0", phase)
pbm.model.add_subsystem("traj", traj)

# %% Initial Conditions
phase.set_time_options(fix_initial=True, duration_bounds=(0.0, 4 * tof), units="s")

phase.add_state("r", fix_initial=True, lower=0.0, rate_source="rdot", units="m")
phase.add_state(
    "theta", fix_initial=True, lower=0.0, rate_source="thetadot", units="rad"
)
phase.add_state("u", fix_initial=True, lower=0.0, rate_source="udot", units="m/s")
phase.add_state("v", fix_initial=True, lower=0.0, rate_source="vdot", units="m/s")
phase.add_state("m", fix_initial=True, lower=0.0, rate_source="mdot", units="kg")

phase.add_control("alpha", lower=-np.pi, upper=np.pi, targets=["alpha"], units="rad")

# %% Terminal Constraints
phase.add_boundary_constraint("r", loc="final", equals=rf, linear=True, units="m")
phase.add_boundary_constraint("u", loc="final", equals=0.0, linear=True, units="m/s")
phase.add_boundary_constraint("v", loc="final", equals=vf, linear=True, units="m/s")

# %% Objective
phase.add_objective("time", loc="final")

# %% Setup and Initial Guess
pbm.setup(check=True)

pbm.set_val("traj.phase0.t_initial", 0.0)
pbm.set_val("traj.phase0.t_duration", tof)
pbm.set_val("traj.phase0.states:r", phase.interp("r", [1.0, rf]))
pbm.set_val("traj.phase0.states:theta", phase.interp("theta", [0.0, np.pi / 18.0]))
pbm.set_val("traj.phase0.states:u", phase.interp("u", [0.0, 0.0]))
pbm.set_val("traj.phase0.states:v", phase.interp("v", [0.0, vf]))
pbm.set_val("traj.phase0.states:m", phase.interp("m", [1.0, 0.7]))
pbm.set_val("traj.phase0.controls:alpha", phase.interp("alpha", [np.pi / 3.0, 0.0]))

# %% Solve the Problem
dm.run_problem(
    pbm, simulate=True, solution_record_file=None, simulation_record_file=None
)

tv = tc * pbm.get_val("traj.phase0.timeseries.time")
mv = pbm.get_val("traj.phases.phase0.timeseries.m")
print(f"\nTime of Flight: {tv[-1, -1]:.3f} s")
print(f"Final Mass:     {mv[-1, -1]:.3f} kg")

# %% Explicit simulation
exp_out = traj.simulate(record_file=None)

# %% Plots
_, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
plt.suptitle("State Variables Timeseries")

ax[0, 0].plot(
    tv,
    rm / 1e3 * pbm.get_val("traj.phase0.timeseries.r"),
    marker="o",
    ms=4,
    linestyle="None",
    label="solution",
)
ax[0, 0].plot(
    tc * exp_out.get_val("traj.phase0.timeseries.time"),
    rm / 1e3 * exp_out.get_val("traj.phase0.timeseries.r"),
    marker=None,
    linestyle="-",
    label="simulation",
)
ax[0, 0].grid(True)
ax[0, 0].set_xlabel(r"$t$ [s]")
ax[0, 0].set_ylabel(r"$r$ [km]")

ax[0, 1].plot(
    tv,
    180.0 / np.pi * pbm.get_val("traj.phase0.timeseries.theta"),
    marker="o",
    ms=4,
    linestyle="None",
    label="solution",
)
ax[0, 1].plot(
    tc * exp_out.get_val("traj.phase0.timeseries.time"),
    180.0 / np.pi * exp_out.get_val("traj.phase0.timeseries.theta"),
    marker=None,
    linestyle="-",
    label="simulation",
)
ax[0, 1].grid(True)
ax[0, 1].set_xlabel(r"$t$ [s]")
ax[0, 1].set_ylabel(r"$\theta$ [deg]")

ax[1, 0].plot(
    tv,
    vc / 1e3 * pbm.get_val("traj.phase0.timeseries.u"),
    marker="o",
    ms=4,
    linestyle="None",
    label="solution",
)
ax[1, 0].plot(
    tc * exp_out.get_val("traj.phase0.timeseries.time"),
    vc / 1e3 * exp_out.get_val("traj.phase0.timeseries.u"),
    marker=None,
    linestyle="-",
    label="simulation",
)
ax[1, 0].grid(True)
ax[1, 0].set_xlabel(r"$t$ [s]")
ax[1, 0].set_ylabel(r"$u$ [km/s]")

ax[1, 1].plot(
    tv,
    vc / 1e3 * pbm.get_val("traj.phase0.timeseries.v"),
    marker="o",
    ms=4,
    linestyle="None",
    label="solution",
)
ax[1, 1].plot(
    tc * exp_out.get_val("traj.phase0.timeseries.time"),
    vc / 1e3 * exp_out.get_val("traj.phase0.timeseries.v"),
    marker=None,
    linestyle="-",
    label="simulation",
)
ax[1, 1].grid(True)
ax[1, 1].set_xlabel(r"$t$ [s]")
ax[1, 1].set_ylabel(r"$v$ [km/s]")


_, ax = plt.subplots(1, 1, constrained_layout=True)
plt.suptitle("Control Variables Timeseries")
ax.plot(
    tv,
    180.0 / np.pi * pbm.get_val("traj.phase0.timeseries.alpha"),
    color="tab:red",
    marker="o",
    ms=4,
    linestyle="-",
    label="solution",
)
ax.grid(True)
ax.set_xlabel(r"$t$ [s]")
ax.set_ylabel(r"$\alpha$ [deg]")

plt.show()

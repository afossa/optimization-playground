"""
Forbes spiral from A. E. Petropoulos and J. A. Sims, “A Review of Some Exact
Solutions to the Planar Equations of Motion of a Thrusting Spacecraft,” in
Proceedings of the 2nd International Symposium on Low-Thrust Trajectory,
Toulouse, France, Jun. 2002, pp. 1–14. [Online]. Available:
https://corpora.tika.apache.org/base/docs/govdocs1/970/970012.pdf
"""

import json
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import spiceypy as spice


class PolarTrajectory:
    """Class PolarTrajectory displays a planar trajectory in polar coordinates.

    Parameters
    ----------
    title : str
        Title on top of the figure
    r_dep : float
        Initial orbit radius [km]
    r_arr : float
        Final orbit radius [km]
    r_body : float
        Central body radius [km]
    r_vec : ndarray
        Transfer trajectory radius time series from analytic solution [km]
    ang_vec: ndarray
        Transfer trajectory angle time series from analytic solution [rad]
    r_vec_exp : ndarray or None, optional
        Transfer trajectory radius time series from numerical propagation [km] or None.
        Default is None
    ang_vec_exp: ndarray or None, optional
        Transfer trajectory angle time series from numerical propagation [rad] or None.
        Default is None

    Attributes
    ----------
    title : str
        Title on top of the figure
    r_dep : float
        Initial orbit radius [km]
    r_arr : float
        Final orbit radius [km]
    r_body : float
        Central body radius [km]
    r_vec : ndarray
        Transfer trajectory radius time series from analytic solution [km]
    ang_vec: ndarray
        Transfer trajectory angle time series from analytic solution [rad]
    r_vec_exp : ndarray or None
        Transfer trajectory radius time series from numerical propagation [km] or None
    ang_vec_exp: ndarray or None
        Transfer trajectory angle time series from numerical propagation [rad] or None

    """

    def __init__(
        self, title, r_dep, r_arr, r_body, r_vec, ang_vec, r_vec_exp=None, ang_vec_exp=None
    ):
        """Initializes PolarTrajectory."""

        self.title = title
        self.r_dep = r_dep
        self.r_arr = r_arr
        self.r_body = r_body
        self.r_vec = r_vec
        self.ang_vec = ang_vec
        self.r_vec_exp = r_vec_exp
        self.ang_vec_exp = ang_vec_exp

    def plot(self):
        """Plots the transfer trajectory."""

        # planet surface, initial and target orbits points
        alpha = np.linspace(0, 2 * np.pi, 1000)

        x_body = self.r_body * np.cos(alpha)
        y_body = self.r_body * np.sin(alpha)

        x_dep = self.r_dep * np.cos(alpha)
        y_dep = self.r_dep * np.sin(alpha)

        x_arr = self.r_arr * np.cos(alpha)
        y_arr = self.r_arr * np.sin(alpha)

        # transfer trajectory points
        x_vec = self.r_vec * np.cos(self.ang_vec)
        y_vec = self.r_vec * np.sin(self.ang_vec)

        # figure
        _, axs = plt.subplots(constrained_layout=True)

        axs.plot(x_body, y_body, label="Moon surface")
        axs.plot(x_dep, y_dep, label="Initial orbit")
        axs.plot(x_arr, y_arr, label="Target orbit")
        axs.plot(x_vec, y_vec, label="Transfer trajectory")

        axs.scatter(x_vec[0], y_vec[0], color="k", label="Departure point", zorder=10)
        axs.scatter(x_vec[-1], y_vec[-1], color="r", label="Insertion point", zorder=15)

        if (self.r_vec_exp is not None) and (self.ang_vec_exp is not None):
            x_vec_exp = self.r_vec_exp * np.cos(self.ang_vec_exp)
            y_vec_exp = self.r_vec_exp * np.sin(self.ang_vec_exp)
            axs.plot(x_vec_exp, y_vec_exp, ".", color="k", label="Explicit simulation")

        axs.set_aspect("equal")
        axs.grid()
        axs.legend(bbox_to_anchor=(1, 1), loc=2)
        axs.set_xlabel("x (km)")
        axs.set_ylabel("y (km)")
        axs.set_title(self.title)
        axs.tick_params(axis="x", rotation=60)


class PolarTimeSeries:
    """Class PolarTimeSeries displays the time series of states and control variables for
    a planar transfer trajectory in polar coordinates.

    Parameters
    ----------
    title : str
        Title on top of the figure
    time: ndarray
        Time vector [s]
    a_sol : object
        Time series from analytic solution as ``[r, theta, u, v]`` [km, rad, km/s, km/s]
    num_sol : object
        Time series from numerical simulation as ``[r, theta, u, v]`` [km, rad, km/s, km/s]
    mass : ndarray
        Spacecraft mass time series [kg]
    thrust : ndarray
        Thrust magnitude time series [N]


    Attributes
    ----------
    title : str
        Title on top of the figure
    time: ndarray
        Time vector [s]
    a_sol : object
        Time series from analytic solution as ``[r, theta, u, v]`` [km, rad, km/s, km/s]
    num_sol : object
        Time series from numerical simulation as ``[r, theta, u, v]`` [km, rad, km/s, km/s]
    mass : ndarray
        Spacecraft mass time series [kg]
    thrust : ndarray
        Thrust magnitude time series [N]

    """

    def __init__(self, title, time, a_sol, num_sol, mass, thrust):
        """Initializes PolarTimeSeries class."""

        self.title = title
        self.time = time
        self.a_sol = a_sol
        self.num_sol = num_sol
        self.mass = mass
        self.thrust = thrust

    def plot(self):
        """Plots the time series."""

        fig, axs = plt.subplots(2, 3, constrained_layout=True)
        fig.suptitle(self.title)

        # radius (km)
        axs[0, 0].plot(self.time, self.a_sol[0, :], color="b", label="analytic")
        axs[0, 0].plot(self.time, self.num_sol[0, :], "--", color="r", label="numerical")
        axs[0, 0].set_xlabel("time (s)")
        axs[0, 0].set_ylabel("r (km)")
        axs[0, 0].set_title("Radius")
        axs[0, 0].grid()
        # axs[0, 0].legend(loc=0)

        # angle (deg)
        axs[0, 1].plot(
            self.time, self.a_sol[1, :] * 180 / np.pi, color="b", label="analytic"
        )
        axs[0, 1].plot(
            self.time, self.num_sol[1, :] * 180 / np.pi, "--", color="r", label="numerical"
        )
        axs[0, 1].set_xlabel("time (s)")
        axs[0, 1].set_ylabel("theta (deg)")
        axs[0, 1].set_title("Angle")
        axs[0, 1].grid()
        # axs[0, 1].legend(loc=0)

        # flight path angle (deg)
        axs[0, 2].plot(self.time, self.thrust, "--", color="r", label="numerical")
        axs[0, 2].set_xlabel("time (s)")
        axs[0, 2].set_ylabel("thrust (N)")
        axs[0, 2].set_title("Thrust magnitude")
        axs[0, 2].grid()
        # axs[0, 2].legend(loc=0)

        # radial velocity (km/s)
        axs[1, 0].plot(self.time, self.a_sol[2, :], color="b", label="analytic")
        axs[1, 0].plot(self.time, self.num_sol[2, :], "--", color="r", label="numerical")
        axs[1, 0].set_xlabel("time (s)")
        axs[1, 0].set_ylabel("u (km/s)")
        axs[1, 0].set_title("Radial velocity")
        axs[1, 0].grid()
        # axs[1, 0].legend(loc=0)

        # tangential velocity (km/s)
        axs[1, 1].plot(self.time, self.a_sol[3, :], color="b", label="analytic")
        axs[1, 1].plot(self.time, self.num_sol[3, :], "--", color="r", label="numerical")
        axs[1, 1].set_xlabel("time (s)")
        axs[1, 1].set_ylabel("v (km/s)")
        axs[1, 1].set_title("Tangential velocity")
        axs[1, 1].grid()
        # axs[1, 1].legend(loc=0)

        # mass (kg)
        axs[1, 2].plot(self.time, self.mass, "--", color="r", label="numerical")
        axs[1, 2].set_xlabel("time (s)")
        axs[1, 2].set_ylabel("mass (kg)")
        axs[1, 2].set_title("Mass")
        axs[1, 2].grid()
        # axs[1, 2].legend(loc=0)


class Spacecraft:
    """Spacecraft class defines the attributes for an electric propelled spacecraft.

    Parameters
    ----------
    mass0 : float
        Initial spacecraft mass [kg]
    thrust_max : float
        Maximum thrust magnitude [N]
    isp : float
        Specific impulse [s]
    mass_dry : float, optional
        Dry mass [kg]. Default is ``m0/100``
    thrust_min : float, optional
        Minimum thrust magnitude [N]. Default is 0.0
    g0 : float, optional
        Standard gravity acceleration [m/s^2]. Default is 9.80665
    scale : float, optional
        Scale factor for exhaust velocity [km/s]. Default is 1e-3

    Attributes
    ----------
    mass0 : float
        Initial spacecraft mass [kg]
    thrust_max : float
        Maximum thrust magnitude [N]
    isp : float
        Specific impulse [s]
    vex : float
        Exhaust velocity [m/s]
    mass_dry : float
        Dry mass [kg]
    thrust_min : float
        Minimum thrust magnitude [N]
    scale : float
        Scale factor for exhaust velocity [km/s]

    """

    def __init__(
        self, mass0, thrust_max, isp, mass_dry=None, thrust_min=0.0, g0=9.80665, scale=1e-3
    ):
        """Initializes Spacecraft class."""

        self.mass0 = float(mass0)
        self.thrust_max = float(thrust_max)
        self.isp = float(isp)
        self.vex = g0 * isp
        self.scale = scale

        if mass_dry is None:
            self.mass_dry = self.mass0 / 100
        else:
            self.mass_dry = mass_dry

        self.thrust_min = thrust_min

    def __str__(self):
        """Prints the Spacecraft object attributes."""

        lines = [
            "\n{:^40s}".format("Spacecraft characteristics:"),
            "\n{:<25s}{:>12.3f}{:>3s}".format("Initial mass:", self.mass0, "kg"),
            "{:<25s}{:>12.3f}{:>3s}".format("Dry mass:", self.mass_dry, "kg"),
            "{:<25s}{:>12.3f}{:>2s}".format("Max thrust:", self.thrust_max, "N"),
            "{:<25s}{:>12.3f}{:>2s}".format("Min thrust:", self.thrust_min, "N"),
            "{:<25s}{:>12.3f}{:>2s}".format("Specific impulse:", self.isp, "s"),
            "{:<25s}{:>12.3f}{:>2s}".format("Exhaust velocity:", self.vex, " m/s"),
        ]

        printed = "\n".join(lines)

        return printed


class TangentialThrust:
    """TangentialThrust computes a sub-optimal solution for a low-thrust transfer
    trajectory between two circular and coplanar orbits in the 2BP framework.
    The thrust direction is assumed to be always equal to the flight path angle.

    Parameters
    ----------
    spacecraft : Spacecraft
        `Spacecraft` object
    r_init : float
        Initial orbit radius [km]
    r_final : float
        Final orbit radius [km]
    fpa_init : float
        Initial flight path angle [rad]
    gm_r2bp : float
        Central body standard gravitational parameter [km^3/s^2]
    t_init : float, optional
        Initial time [s]. Default is 0.0
    theta_init : float, optional
        Initial angle [rad]. Default is 0.0

    Attributes
    ----------
    spacecraft : Spacecraft
        `Spacecraft` object
    r_init : float
        Initial orbit radius [km]
    r_final : float
        Final orbit radius [km]
    fpa_init : float
        Initial flight path angle [rad]
    tan_fpa_init : float
        Tangent of `fpa_init` [-]
    gm_r2bp : float
        Central body standard gravitational parameter [km^3/s^2]
    t_init : float
        Initial time [s]
    theta_init : float
        Initial angle [rad]
    theta_final : float
        Final angle [rad]
    tof : float
        Time of flight [s]
    t_final : float
        Final time [s]
    t_vec : ndarray
        Time vector [s]
    theta_vec : ndarray
        Angle time series [rad]
    r_vec : ndarray
        Radius time series [km]
    u_vec : ndarray
        Radial velocity time series [km/s]
    v_vec : ndarray
        Tangential velocity time series [km/s]
    fpa_vec : ndarray
        Flight path angle and thrust direction time series [rad]
    acc_vec : ndarray
        Acceleration magnitude time series [km/s^2]
    mass_vec : ndarray
        Spacecraft mass time series [kg]
    thrust_vec : ndarray
        Thrust magnitude time series [N]
    a_sol : ndarray
        Analytical solution time series as ``[r, theta, u, v, fpa, a]``
    num_sol : ndarray
        Numerical solution time series as ``[r, theta, u, v, m]``

    """

    def __init__(
        self, spacecraft, r_init, r_final, fpa_init, gm_r2bp, t_init=0.0, theta_init=0.0
    ):
        """Initializes TangentialThrust class."""

        self.spacecraft = spacecraft
        self.r_init = r_init
        self.r_final = r_final
        self.fpa_init = fpa_init
        self.tan_fpa_init = np.tan(self.fpa_init)
        self.gm_r2bp = gm_r2bp
        self.t_init = t_init
        self.theta_init = theta_init

        # final angle
        self.theta_final = (
            self.theta_init + (1.0 - self.r_init / self.r_final) / self.tan_fpa_init
        )

        # time of flight and final time
        sol = spi.solve_ivp(
            lambda t, y: self.dt_dtheta(y, t),
            t_span=[self.theta_init, self.theta_final],
            y0=[0.0],
            method="DOP853",
            rtol=1e-14,
            atol=1e-14,
        )
        # time, _ = odeint(self.dt_dtheta, y0=[0.0], t=[self.theta_init, self.theta_final],
        #                  full_output=True, rtol=dft.rtol, atol=dft.atol)

        self.tof = sol.y[-1, -1]
        self.t_final = self.t_init + self.tof

        # initialization
        self.t_vec = self.theta_vec = self.r_vec = self.u_vec = self.v_vec = (
            self.fpa_vec
        ) = self.acc_vec = self.mass_vec = self.thrust_vec = self.a_sol = self.num_sol = (
            None
        )

    def compute_time_series(self, **kwargs):
        """Computes the time series of angle, radius, flight path angle, tangential velocity,
        radial velocity, spacecraft mass and thrust magnitude on a given time vector `time` or
        number of points `nb_points`.

        Other Parameters
        ----------------
        **kwargs :
        time : ndarray
            Time vector in which the time series are computed, must be monotonic increasing [s]
        nb_points : int
            Number of points equally spaced in time in which the time series are computed [-]

        """

        # time vector in which the solution is computed
        if "time" in kwargs:
            time = kwargs["time"]
            time = np.sort(np.asarray(time))
            if np.allclose(
                np.array([self.t_init, self.t_final]), np.array([time[0], time[-1]])
            ):
                self.t_vec = time
            else:
                raise ValueError("Elements in time must lie in [t0, tf]")

        elif "nb_points" in kwargs:
            nb_points = kwargs["nb_points"]
            if isinstance(nb_points, int) and nb_points > 1:
                self.t_vec = np.linspace(self.t_init, self.t_final, nb_points)
            else:
                raise TypeError("Number of points must be a positive integer nb_points > 1")

        else:
            raise Exception(
                "Must provide one between time vector time or number of points " "nb_points"
            )

        # analytic solution and numerical integration
        self.analytic_solution()
        self.simulate()

        check = np.allclose(self.a_sol[:4, :], self.num_sol[:4, :], rtol=1e-3, atol=1e-3)
        print(
            "{:<50s}{:<30s}".format(
                "Match between analytic and numerical solutions", str(check)
            )
        )

        # mass and thrust magnitude time series
        self.mass_vec = self.num_sol[4, :]
        self.thrust_vec = self.acc_vec * self.mass_vec / self.spacecraft.scale

    def analytic_solution(self):
        """Analytic solution for the time series of ``[r, theta, u, v, fpa, a]``."""

        sol = spi.solve_ivp(
            self.theta_dot,
            t_span=(self.t_init, self.t_final),
            y0=[self.theta_init],
            method="DOP853",
            t_eval=self.t_vec,
            rtol=1e-14,
            atol=1e-14,
        )

        # theta, sol = odeint(self.theta_dot, y0=[self.theta_init], t=self.t_vec,
        #                     full_output=True, rtol=dft.rtol, atol=dft.atol, tfirst=True)

        self.theta_vec = sol.y.ravel()
        self.r_vec = self.r_theta(self.theta_vec)
        self.v_vec = self.r_vec * self.theta_dot(self.t_vec, self.theta_vec)
        self.u_vec = self.tan_fpa_theta(self.theta_vec) * self.v_vec
        self.fpa_vec = np.arctan(self.tan_fpa_theta(self.theta_vec))
        self.acc_vec = (
            self.gm_r2bp
            * self.tan_fpa_init**2
            / (self.r_init**2 * 2 * np.sin(self.fpa_vec))
        )
        self.a_sol = np.vstack(
            (self.r_vec, self.theta_vec, self.u_vec, self.v_vec, self.fpa_vec, self.acc_vec)
        )

        print("{:<50s}{:<30s}".format("Computing analytic solution", sol["message"]))

    def simulate(self):
        """Explicitly simulates the transfer trajectory."""

        # initial conditions
        state0 = [
            self.r_init,
            self.theta_init,
            self.u_vec[0],
            self.v_vec[0],
            self.spacecraft.mass0,
        ]

        sol = spi.solve_ivp(
            self.odes,
            t_span=(self.t_init, self.t_final),
            y0=state0,
            method="DOP853",
            t_eval=self.t_vec,
            rtol=1e-14,
            atol=1e-14,
        )

        # states, sol = odeint(self.odes, y0=state0, t=self.t_vec, full_output=True,
        #                      rtol=dft.rtol, atol=dft.atol, tfirst=True)

        self.num_sol = sol.y

        print("{:<50s}{:<30s}".format("Simulating trajectory", sol["message"]))

    def r_theta(self, theta):
        """Defines an explicit expression for the radius as a function of the angle theta.

        Parameters
        ----------
        theta : float or ndarray
            Angle [rad]

        Returns
        -------
        radius : float or ndarray
            Radius [km]

        """

        radius = self.r_init / (1.0 - (theta - self.theta_init) * self.tan_fpa_init)

        return radius

    def dr_dtheta(self, theta):
        """Defines an explicit expression for the first derivative
        of the radius with respect to the angle theta.

        Parameters
        ----------
        theta : float or ndarray
            Angle [rad]

        Returns
        -------
        dr_dtheta : float or ndarray
            Radius rate [km/s]

        """

        dr_dtheta = (self.r_theta(theta) ** 2) * (self.tan_fpa_init / self.r_init)

        return dr_dtheta

    def tan_fpa_theta(self, theta):
        """Defines an explicit expression for the tangent of
        the flight path angle as a function of the angle theta.

        Parameters
        ----------
        theta : float or ndarray
            Angle [rad]

        Returns
        -------
        tan_fpa : float or ndarray
            Tangent of the flight path angle [-]

        """

        tan_fpa = self.r_theta(theta) * (self.tan_fpa_init / self.r_init)

        return tan_fpa

    def theta_dot(self, _, theta):
        """Defines an explicit expression for the first
        derivative of theta with respect to time.

        Parameters
        ----------
        _ : float
            Time handled internally by integrator [s]
        theta : float
            Angle [rad]

        Returns
        -------
        theta_dot : float
            Angle rate [rad/s]

        """

        theta_dot = (self.gm_r2bp**0.5) * (self.r_theta(theta) ** -1.5)

        return theta_dot

    def dt_dtheta(self, time, theta):
        """Defines an explicit expression for the first derivative of the time with respect
        to the angle.

        Parameters
        ----------
        time : float or ndarray
            Time [s]
        theta : float or ndarray
            Angle [rad]

        Returns
        -------
        dt_dtheta : float or ndarray
            First derivative of `time` wrt `theta` [rad/s]

        """

        dt_dtheta = 1.0 / self.theta_dot(time, theta)

        return dt_dtheta

    def odes(self, _, state):
        """Defines the ODEs for a planar transfer trajectory with tangential thrust
        in polar coordinates.

        Parameters
        ----------
        _ : float
            Time handled implicitly by integrator [s]
        state : ndarray
            States variables as ``[r, theta, u, v, m]``

        Returns
        -------
        states_dot : ndarray
            States variables rates as ``[r_dot, theta_dot, u_dot, v_dot, mass_dot]``

        """

        speed = (state[2] ** 2 + state[3] ** 2) ** 0.5  # velocity magnitude [km/s]
        acc = (self.gm_r2bp / self.r_init**2) * (
            np.tan(self.fpa_init) ** 2 / 2.0
        )  # thrust acceleration [km/s^2]

        r_dot = state[2]
        theta_dot = state[3] / state[0]
        u_dot = -self.gm_r2bp / state[0] ** 2 + state[3] ** 2 / state[0] + acc
        v_dot = -state[2] * state[3] / state[0] + acc * (state[3] / state[2])
        mass_dot = (
            -(acc / (self.spacecraft.vex * self.spacecraft.scale))
            * state[4]
            * (speed / state[2])
        )

        states_dot = [r_dot, theta_dot, u_dot, v_dot, mass_dot]

        return states_dot


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, o):
        """Return a serializable data type."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Spacecraft):
            return o.__dict__
        return json.JSONEncoder.default(self, o)


if __name__ == "__main__":
    import os
    from pathlib import Path

    # load SPICE kernels
    pth = Path(__file__).parent.parent
    spice.furnsh(os.path.join(pth, "kernels/gm_de440.tpc"))

    # spacecraft object
    sc = Spacecraft(1000, 20.0, 2000)
    print(sc)

    # spiral object
    gm = spice.bodvcd(301, "GM", 1)[1][0]
    tt = TangentialThrust(sc, 2000, 5000, np.pi / 180, gm)

    # write data to JSON file
    # tt.compute_time_series(nb_points=2)
    # with open("forbes_spiral.json", "w") as file:
    #     json.dump(tt.__dict__, file, indent=4, cls=CustomEncoder)

    # timeseries and trajectory plots
    tt.compute_time_series(nb_points=2000)
    title = "Tangential thrust approximation"
    pt = PolarTimeSeries(
        title, tt.t_vec, tt.a_sol[:4, :], tt.num_sol[:4, :], tt.mass_vec, tt.thrust_vec
    )
    pp = PolarTrajectory(title, tt.r_init, tt.r_final, 1738.1, tt.r_vec, tt.theta_vec)
    pt.plot()
    pp.plot()

    plt.show()

    spice.kclear()

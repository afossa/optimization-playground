"""
De-orbit burn and powered descent and landing on Europa.
"""

import json
import numpy as np
import scipy.integrate as si
import openmdao.api as om
import dymos as dm
import matplotlib as mp
import matplotlib.pyplot as plt


# %% Dymos Ordinary Differential Equations (ODEs)

class ThrustODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('ve', types=float)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('thrust', val=np.zeros(nn), desc='thrust', units='N')
        self.add_input('ve', val=self.options['ve'] * np.ones(nn), desc='exhaust velocity', units='m/s')

        # outputs (ODE left-hand side)
        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('thetadot', val=np.zeros(nn), desc='angular rate', units='rad/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('vdot', val=np.zeros(nn), desc='tangential acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')

        # self.declare_coloring(wrt='*', method='cs', show_sparsity=False)

        # partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='thetadot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='thetadot', wrt='v', rows=ar, cols=ar)

        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='vdot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='ve', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        outputs['rdot'] = u
        outputs['thetadot'] = v / r
        outputs['udot'] = -self.options['mu'] / r ** 2 + v ** 2 / r + (thrust / m) * sin_alpha
        outputs['vdot'] = -u * v / r + (thrust / m) * cos_alpha
        outputs['mdot'] = -thrust / inputs['ve']

    def compute_partials(self, inputs, jacobian):
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        jacobian['thetadot', 'r'] = -v / r ** 2
        jacobian['thetadot', 'v'] = 1.0 / r

        jacobian['udot', 'r'] = 2.0 * self.options['mu'] / r ** 3 - (v / r) ** 2
        jacobian['udot', 'v'] = 2.0 * v / r
        jacobian['udot', 'm'] = -(thrust / m ** 2) * sin_alpha
        jacobian['udot', 'alpha'] = (thrust / m) * cos_alpha
        jacobian['udot', 'thrust'] = sin_alpha / m

        jacobian['vdot', 'r'] = u * v / r ** 2
        jacobian['vdot', 'u'] = -v / r
        jacobian['vdot', 'v'] = -u / r
        jacobian['vdot', 'm'] = -(thrust / m ** 2) * cos_alpha
        jacobian['vdot', 'alpha'] = -(thrust / m) * sin_alpha
        jacobian['vdot', 'thrust'] = cos_alpha / m

        jacobian['mdot', 'thrust'] = -1.0 / inputs['ve']
        jacobian['mdot', 've'] = thrust / inputs['ve'] ** 2


class SafeAlt(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('rp', types=float)
        self.options.declare('hs', types=float)
        self.options.declare('s', types=float)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('theta', val=np.zeros(nn), desc='true anomaly', units='rad')

        self.add_output('rs', val=np.zeros(nn), desc='minimum safe radius')
        self.add_output('d', val=np.zeros(nn), desc='distance from minimum safe radius')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rs', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='d', wrt='r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='d', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rp = self.options['rp']
        hs = self.options['hs']
        s = self.options['s']

        r = inputs['r']
        theta = inputs['theta']

        rs = rp + hs * rp * theta / (rp * theta + hs / s)

        outputs['rs'] = rs
        outputs['d'] = r - rs

    def compute_partials(self, inputs, jacobian):
        rp = self.options['rp']
        hs = self.options['hs']
        s = self.options['s']

        theta = inputs['theta']

        drs = hs ** 2 * rp * s / (hs + s * rp * theta) ** 2

        jacobian['rs', 'theta'] = drs
        jacobian['d', 'theta'] = -drs


class VLandODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('ve', types=float)
        self.options.declare('rp', types=float)
        self.options.declare('hs', types=float)
        self.options.declare('s', types=float)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='odes',
                           subsys=ThrustODE(num_nodes=nn, mu=self.options['mu'], ve=self.options['ve']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 've'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='safe_alt',
                           subsys=SafeAlt(num_nodes=nn, rp=self.options['rp'],
                                          hs=self.options['hs'], s=self.options['s']),
                           promotes_inputs=['r', 'theta'],
                           promotes_outputs=['rs', 'd'])


# %% Classes for initial guess computation

class PowConstRadius:
    """Powered trajectory at constant radius. """

    def __init__(self, _mu, _rc, _v0, _v1, _thr, _mdt):
        """Initialization. """
        self.mu = _mu
        self.rc = _rc
        self.v0 = _v0
        self.v1 = _v1
        self.thr = _thr
        self.mdt = _mdt
        self.sgn = 1.0 if _v1 > _v0 else -1.0

    def get_final(self, _t0, _m0):
        """Estimates the time of flight, swept angle and final mass. """
        sol1 = si.odeint(lambda v, t: self._dt_dv(v, t, _t0, _m0), _t0, [self.v0, self.v1],
                         rtol=3e-14, atol=1e-14, tfirst=True, full_output=True)
        sol2 = si.odeint(lambda t, x: self._dx_dt(t, x, _t0, _m0), [0.0, self.v0], sol1[0].ravel(),
                         rtol=3e-14, atol=1e-14, tfirst=True, full_output=True)
        _dt = sol1[0][-1, 0] - sol1[0][0, 0]
        _df = sol2[0][-1, 0] - sol2[0][0, 0]
        assert np.isclose(self.v1, sol2[0][-1, 1], rtol=3e-12, atol=1e-12)
        return _dt, _df, _m0 + self.mdt * _dt, sol1, sol2

    def get_traj(self, _tv, _t0, _m0, _f0):
        """Computes the states and control variables timeseries on the provided time grid. """
        _tg = _tv if _tv[0] == _t0 else np.concatenate(([_t0], _tv))
        sol = si.odeint(lambda t, x: self._dx_dt(t, x, _t0, _m0), [_f0, self.v0], _tg,
                        rtol=3e-14, atol=1e-14, tfirst=True, full_output=True)
        _xv = sol[0] if _tv[0] == _t0 else sol[0][1:, :]
        if _tv[-1] != _t0:
            _xv[-1, 1] = self.v1
        _sv = np.empty((_tv.size, 5))
        _sv[:, 0] = self.rc
        _sv[:, 1] = _xv[:, 0]
        _sv[:, 2] = 0.0
        _sv[:, 3] = _xv[:, 1]
        _sv[:, 4] = _m0 + self.mdt * (_tv - _t0)
        _cv = np.empty((_tv.size, 2))
        _cv[:, 0] = self._alpha(_tv, _xv[:, 1], _t0, _m0)
        _cv[:, 1] = self.thr
        return _sv, _cv

    def _alpha(self, _tv, _vv, _t0, _m0):
        """Computes the thrust direction. """
        _av = np.arctan2(self.mu / self.rc ** 2 - _vv ** 2 / self.rc, self._dv_dt(_tv, _vv, _t0, _m0))
        _av[_av < -np.pi / 2.0] = _av[_av < -np.pi / 2.0] + 2.0 * np.pi
        return _av

    def _dv_dt(self, _t, _v, _t0, _m0):
        """ODE for tangential velocity at constant radius. """
        return self.sgn * np.sqrt((self.thr / (_m0 + self.mdt * (_t - _t0))) ** 2 -
                                  (self.mu / self.rc ** 2 - _v ** 2 / self.rc) ** 2)

    def _dt_dv(self, _v, _t, _t0, _m0):
        """ODE for time as function of tangential velocity at constant radius. """
        return 1.0 / self._dv_dt(_t, _v, _t0, _m0)

    def _dx_dt(self, _t, _x, _t0, _m0):
        """ODE for swept angle and tangential velocity at constant radius. """
        return [_x[1] / self.rc, self._dv_dt(_t, _x[1], _t0, _m0)]


class KeplerOrbit:
    """Keplerian orbit class. """

    def __init__(self, _mu, _ra, _rp):
        """Initialization. """
        self.mu = _mu
        self.ra = _ra
        self.rp = _rp
        self._compute()

    def get_period(self):
        """Returns the orbit period. """
        return 2.0 * np.pi / np.sqrt(self.mu) * self.a ** 1.5

    def get_traj(self, _tv, _t0, _m0, _f0, _a0, _f1):
        """Computes the states and control variables timeseries on the provided time grid. """
        _tg = _tv if _tv[0] == _t0 else np.concatenate(([_t0], _tv))
        sol = si.odeint(lambda t, f: self._df_dt(t, f), _f1, _tg,
                        rtol=3e-14, atol=1e-14, tfirst=True, full_output=True)
        _fv = sol[0].ravel() if _tv[0] == _t0 else sol[0][1:, :].ravel()
        _sv = np.empty((_tv.size, 5))
        _sv[:, 0] = self.p / (1.0 + self.e * np.cos(_fv))
        _sv[:, 1] = _fv - _f1 + _f0
        _sv[:, 2] = self.mu / self.h * self.e * np.sin(_fv)
        _sv[:, 3] = self.mu / self.h * (1.0 + self.e * np.cos(_fv))
        _sv[:, 4] = _m0
        _cv = np.empty((_tv.size, 2))
        _cv[:, 0] = _a0
        _cv[:, 1] = 0.0
        return _sv, _cv

    def _compute(self):
        """Computes the transfer characteristics. """
        self.a = 0.5 * (self.ra + self.rp)
        self.e = (self.ra - self.rp) / (self.ra + self.rp)
        self.p = self.a * (1.0 - self.e * self.e)
        self.h = np.sqrt(self.mu * self.p)
        self.va = np.sqrt(self.mu / self.ra * (1.0 - self.e))
        self.vp = np.sqrt(self.mu / self.rp * (1.0 + self.e))

    def _df_dt(self, _t, _f):
        """ODE for true anomaly. """
        return self.mu ** 2 / self.h ** 3 * (1.0 + self.e * np.cos(_f)) ** 2


# %% Classes for serialization

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for Numpy data types. """

    def default(self, o):
        """Return a serializable data type. """
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float64):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


# %% Simulation parameters

# NLP settings
nb_seg = 300
order = 3
opti = 'IPOPT'

# physical parameters
MU = 3188.82e9  # Europa's standard gravitational parameter [m^3/s^2]
RE = 1565e3  # Europa's mean radius [m]
g0 = 9.80665  # standard sea-level gravity [m/s^2]

# spacecraft characteristics
THR = 12e3  # thrust force [N]
ISP = 230.0  # specific impulse [s]
MDRY = 3529.0  # spacecraft dry mass [kg]
MWET = MDRY / 0.39  # spacecraft wet mass [kg]

# initial orbit
RP = RE + 100e3  # parking orbit periapsis radius [m]
RA = RE + 14455e3  # parking orbit apoapsis radius [m]

# vertical landing constraint
HS = 10e3  # safe altitude [m]
sl = -10.  # constraint slope [-]

# characteristic quantities
lc = RE  # characteristic length [m]
tc = np.sqrt(RE ** 3 / MU)  # characteristic time [s]
vc = np.sqrt(MU / RE)  # characteristic speed [m/s]
ac = MU / (RE * RE)  # characteristic acceleration [m/s^2]
mc = MDRY  # characteristic mass [kg]

# normalized physical parameters
mu = 1.0  # Europa's standard gravitational parameter [-]
re = 1.0  # Europa's mean radius [-]

# normalized spacecraft characteristics
thr = THR / (mc * ac)  # thrust force [-]
ve = ISP * g0 / vc  # exhaust velocity [-]
mdt = - thr / ve  # mass flow rate [-]
m0 = MWET / mc  # initial mass [-]

# initial orbit object
dep = KeplerOrbit(mu, RA / lc, RP / lc)

# final constraints
rf = re  # final radius [-]
uf = 0.0  # final radial velocity [-]
vf = 0.0  # final tangential velocity [-]

# %% Initial guess computation

# Hohmann transfer
hoh = KeplerOrbit(mu, dep.ra, rf)
dt1 = 0.5 * hoh.get_period()  # Hohmann transfer time of flight [-]
df1 = np.pi  # swept angle [-]

# de-orbit burn
pw0 = PowConstRadius(mu, hoh.ra, dep.va, hoh.va, thr, mdt)
dt0, df0, m1, _, _ = pw0.get_final(0.0, m0)  # burn duration, swept angle and final mass [-]

# landing burn
pw2 = PowConstRadius(mu, rf, hoh.vp, 0.0, thr, mdt)
dt2, df2, m2, _, _ = pw2.get_final(0.0, m1)  # burn duration, swept angle and final mass [-]
assert m2 > MDRY / mc

# total time of flight and swept angle
dtt = dt0 + dt1 + dt2
dft = df0 + df1 + df2

# %% Optimal Control Problem (OCP) initialization

pbm = om.Problem(model=om.Group(), reports=None)
pbm.model.linear_solver = om.DirectSolver()
pbm.driver = om.pyOptSparseDriver()

pbm.driver.options['optimizer'] = opti
pbm.driver.options['print_results'] = False
pbm.driver.declare_coloring(show_summary=False, show_sparsity=False)

# IPOPT options
if opti == 'IPOPT':
    pbm.driver.opt_settings['tol'] = 1e-8
    pbm.driver.opt_settings['acceptable_tol'] = 1e-8
    pbm.driver.opt_settings['linear_solver'] = 'ma57'
    pbm.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    pbm.driver.opt_settings['ma57_automatic_scaling'] = 'yes'
    pbm.driver.opt_settings['print_level'] = 5
    pbm.driver.opt_settings['output_file'] = ''
    pbm.driver.opt_settings['print_timing_statistics'] = 'yes'
elif opti == 'SNOPT':
    pbm.driver.opt_settings['Major feasibility tolerance'] = 1e-9
    pbm.driver.opt_settings['Major optimality tolerance'] = 1e-9
    pbm.driver.opt_settings['Minor feasibility tolerance'] = 1e-9

# %% Phases and Trajectory initialization

ode_kwargs = {'mu': mu, 've': ve, 'rp': re, 'hs': HS / lc, 's': sl}

traj = dm.Trajectory()
phase0 = dm.Phase(ode_class=VLandODE, ode_init_kwargs=ode_kwargs,
                  transcription=dm.GaussLobatto(num_segments=nb_seg, order=order, compressed=True))
traj.add_phase('phase0', phase0)
pbm.model.add_subsystem('traj', traj)

# %% Boundary Conditions (BCs)

# time options
phase0.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(0.0, 2 * dtt),
                        duration_ref0=0.0, duration_ref=dtt, units='s')

# state variables
phase0.add_state('r', fix_initial=True, fix_final=True, lower=rf, ref0=rf, ref=dep.ra,
                 rate_source='rdot', units='m')
phase0.add_state('theta', fix_initial=False, fix_final=True, upper=0.0, ref0=0.0, ref=dft,
                 rate_source='thetadot', units='rad')
phase0.add_state('u', fix_initial=True, fix_final=True, upper=0.0, ref0=0.0, ref=dep.va,
                 rate_source='udot', units='m/s')
phase0.add_state('v', fix_initial=True, fix_final=True, lower=0.0, ref0=0.0, ref=dep.va,
                 rate_source='vdot', units='m/s')
phase0.add_state('m', fix_initial=False, fix_final=True, lower=MDRY / mc, ref0=MDRY / mc, ref=m0,
                 rate_source='mdot', units='kg')

# control variables
phase0.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                   rate2_continuity=False, lower=0.0, upper=2.0 * np.pi, ref0=0.0, ref=2.0 * np.pi)
phase0.add_control('thrust', fix_initial=False, fix_final=False, continuity=False,
                   rate_continuity=False, rate2_continuity=False, lower=0.0, upper=thr, ref0=0.0, ref=thr)

# vertical landing constraint
phase0.add_path_constraint('d', lower=0.0, ref0=0.0, ref=HS / re)
phase0.add_timeseries_output('rs')

# parameters
phase0.add_parameter('ve', val=ve, units='m/s', opt=False, desc='exhaust velocity')

# objective function
phase0.add_objective('m', loc='final', scaler=-1.0)

pbm.setup(check=True)

# %% Initial guess

# time grid
pbm.set_val('traj.phase0.t_initial', 0.0)
pbm.set_val('traj.phase0.t_duration', dtt)
pbm.run_model()

tgt = pbm.get_val('traj.phase0.timeseries.time').ravel()
tg0 = tgt[tgt <= dt0]
tg1 = tgt[tgt > dt0]
tg1 = tg1[tg1 < (dt0 + dt1)]
tg2 = tgt[tgt >= (dt0 + dt1)]

# de-orbit burn
sg0, cg0 = pw0.get_traj(tg0, 0.0, m0, 0.0)

# ballistic phase
sg1, cg1 = hoh.get_traj(tg1, dt0, m1, df0, cg0[-1, 0], np.pi)

# landing burn
sg2, cg2 = pw2.get_traj(tg2, dt0 + dt1, m1, df0 + df1)

# complete guess
sgt = np.concatenate((sg0, sg1, sg2), axis=0)
cgt = np.concatenate((cg0, cg1, cg2), axis=0)

# rescale final mass
sgt[:, -1] /= sgt[-1, -1]

snt = phase0.options['transcription'].grid_data.subset_node_indices['state_input']
cnt = phase0.options['transcription'].grid_data.subset_node_indices['control_input']

pbm.set_val('traj.phase0.states:r', np.take(sgt[:, 0], snt))
pbm.set_val('traj.phase0.states:theta', np.take(sgt[:, 1] - dft, snt))
pbm.set_val('traj.phase0.states:u', np.take(sgt[:, 2], snt))
pbm.set_val('traj.phase0.states:v', np.take(sgt[:, 3], snt))
pbm.set_val('traj.phase0.states:m', np.take(sgt[:, 4], snt))

pbm.set_val('traj.phase0.controls:alpha', np.take(cgt[:, 0], cnt))
pbm.set_val('traj.phase0.controls:thrust', np.take(cgt[:, 1], cnt))

# %% OCP solution

pbm.run_model()
dm.run_problem(pbm, simulate=False, solution_record_file=None, simulation_record_file=None)

print(f"Initial mass: {pbm.get_val('traj.phase0.timeseries.m')[0, 0] * mc} kg")
print(f"Final mass:   {pbm.get_val('traj.phase0.timeseries.m')[-1, 0] * mc} kg")

# %% Explicit integration

exp = traj.simulate(record_file=None)

# %% Store simulation data to JSON file

tvi, idx = np.unique(pbm.get_val('traj.phase0.timeseries.time'), return_index=True)
tvi = tc * tvi
svi = np.empty((tvi.size, 5))
svi[:, 0] = lc * 1e-3 * np.take(pbm.get_val('traj.phase0.timeseries.r'), idx)
svi[:, 1] = 180.0 / np.pi * np.take(pbm.get_val('traj.phase0.timeseries.theta'), idx)
svi[:, 2] = vc * 1e-3 * np.take(pbm.get_val('traj.phase0.timeseries.u'), idx)
svi[:, 3] = vc * 1e-3 * np.take(pbm.get_val('traj.phase0.timeseries.v'), idx)
svi[:, 4] = mc * np.take(pbm.get_val('traj.phase0.timeseries.m'), idx)

out = {
    'europa': {'mu': MU * 1e-9, 'mean_radius': RE * 1e-3},
    'parking_orbit': {'periapsis_radius': RP * 1e-3, 'apoapsis_radius': RA * 1e-3},
    'spacecraft': {'thrust': THR, 'isp': ISP, 'dry_mass': MDRY},
    'vertical_constraint': {'minimum_altitude': HS * 1e-3, 'slope': sl},
    'transcription': {'segments': nb_seg, 'order': order},
    'solution': {'time': tvi,
                 'states': {'radius': svi[:, 0], 'swept_angle': svi[:, 1],
                            'radial_velocity': svi[:, 2], 'tangential_velocity': svi[:, 3],
                            'mass': svi[:, 4]}}
}

with open('europa_landing.json', 'w') as fid:
    json.dump(out, fid, cls=NumpyEncoder, indent=4)

# header = "time [s], radius [km], angle [deg], radial velocity [km/s], tangential velocity [km/s], mass [kg]"
# np.savetxt('europa_landing.txt', np.concatenate((tvi.reshape(tvi.size, 1), svi), axis=1), header=header)

# %% Plots

tve = tc * exp.get_val('traj.phase0.timeseries.time')

_, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
plt.suptitle('State Variables Timeseries')

ax[0, 0].plot(tvi, svi[:, 0], marker='o', ms=4, linestyle='None')
ax[0, 0].plot(tve, lc * 1e-3 * exp.get_val('traj.phase0.timeseries.r'),
              marker=None, linestyle='-')
ax[0, 0].grid(True)
ax[0, 0].set_xlabel(r'$t$ [s]')
ax[0, 0].set_ylabel(r'$r$ [km]')

ax[0, 1].plot(tvi, svi[:, 1], marker='o', ms=4, linestyle='-')
ax[0, 1].plot(tve, 180.0 / np.pi * exp.get_val('traj.phase0.timeseries.theta'),
              marker=None, linestyle='-')
ax[0, 1].grid(True)
ax[0, 1].set_xlabel(r'$t$ [s]')
ax[0, 1].set_ylabel(r'$\theta$ [deg]')

ax[1, 0].plot(tvi, svi[:, 2], marker='o', ms=4, linestyle='-')
ax[1, 0].plot(tve, vc * 1e-3 * exp.get_val('traj.phase0.timeseries.u'),
              marker=None, linestyle='-')
ax[1, 0].grid(True)
ax[1, 0].set_xlabel(r'$t$ [s]')
ax[1, 0].set_ylabel(r'$u$ [km/s]')

ax[1, 1].plot(tvi, svi[:, 3], marker='o', ms=4, linestyle='-')
ax[1, 1].plot(tve, vc * 1e-3 * exp.get_val('traj.phase0.timeseries.v'),
              marker=None, linestyle='-')
ax[1, 1].grid(True)
ax[1, 1].set_xlabel(r'$t$ [s]')
ax[1, 1].set_ylabel(r'$v$ [km/s]')

_, ax = plt.subplots(1, 1, constrained_layout=True)
plt.suptitle('Spacecraft Mass Timeseries')

ax.plot(tvi, svi[:, 4], marker='o', ms=4, linestyle='-')
ax.grid(True)
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$m$ [kg]')

_, ax = plt.subplots(2, 1, constrained_layout=True)
plt.suptitle('Control Variables Timeseries')

ax[0].plot(tvi, 180.0 / np.pi * np.take(pbm.get_val('traj.phase0.timeseries.alpha'), idx),
           color='tab:red', marker='o', ms=4, linestyle='-')
ax[0].grid(True)
ax[0].set_xlabel(r'$t$ [s]')
ax[0].set_ylabel(r'$\alpha$ [deg]')

ax[1].plot(tvi, mc * ac * np.take(pbm.get_val('traj.phase0.timeseries.thrust'), idx),
           color='tab:red', marker='o', ms=4, linestyle='-')
ax[1].grid(True)
ax[1].set_xlabel(r'$t$ [s]')
ax[1].set_ylabel(r'$T$ [N]')

plt.show()

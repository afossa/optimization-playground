"""
Part of this code has been adapted from https://github.com/naoyaozaki/LambertProblem.jl
maintained by Naoya Ozaki and released under the MIT License.

MIT License

Copyright (c) 2022 Naoya Ozaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings
import numpy as np


def lambert_solver(r1,r2,tof,mu,multi_revs=0,is_retrograde=False):
    """
    Solves the Lambert's problem, i.e. computes the initial and final velocity
    vectors for given initial and final position vectors and time of flight.

    For N maximum number of multiple revolutions, the total number of solutions
    is 2N+1. The output velocity vectors are organized in two numpy arrays of
    shape (2N+1, 3).

    Parameters
    ----------
    r1 : numpy.ndarray
        Initial position vector.
    r2 : numpy.ndarray
        Final position vector.
    tof : float
        Time of flight.
    mu : float
        Gravitational parameter.
    multi_revs : int, optional
        Maximum number of multiple revolutions to compute. The default is 0.
    is_retrograde : bool, optional
        Flag to indicate if the motion is retrograde. The default is False.

    Returns
    -------
    numpy.ndarray
        Initial velocity vectors.
    numpy.ndarray
        Final velocity vectors.
    int
        Number of solutions.

    """

    # 0. Sanity Check
    if tof <= 0:
        raise ValueError("Time of flight must be positive!")
    if mu <= 0:
        raise ValueError("Gravity parameter must be positive!")

    # 1. Preparation
    # Variables used in Lambert's Problem
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    c = np.linalg.norm(r2 - r1)
    s = 0.5 * (r1_norm + r2_norm + c)
    lbd2 = 1.0 - c / s
    tof_ndim = np.sqrt(2.0 * mu / (s**3)) * tof

    # Basis vectors
    ivec_r1 = r1 / r1_norm
    ivec_r2 = r2 / r2_norm
    ivec_h = np.cross(ivec_r1, ivec_r2)
    ivec_h = ivec_h / np.linalg.norm(ivec_h) # Angular momentum vector
    # NOTE: ivec_h cannot be defined if r1 // r2

    if ivec_h[2] == 0.0:
        raise ValueError("The angular momentum vector has no z component, impossible to define automatically clock or counterclockwise!")
    elif ivec_h[2] < 0.0:
        lbd = -np.sqrt(lbd2)
        ivec_t1 = -np.cross(ivec_h, ivec_r1)
        ivec_t2 = -np.cross(ivec_h, ivec_r2)
    else:
        lbd = np.sqrt(lbd2)
        ivec_t1 = np.cross(ivec_h, ivec_r1)
        ivec_t2 = np.cross(ivec_h, ivec_r2)

    if is_retrograde: # Retrograde Motion
        lbd = -lbd
        ivec_t1 = -ivec_t1
        ivec_t2 = -ivec_t2

    # 2. Calculate x
    _, x_all = _find_xy(lbd, tof_ndim, multi_revs)

    # 3. Calculate Terminal Velocities
    vvec_1_all = np.zeros((len(x_all), 3))
    vvec_2_all = np.zeros((len(x_all), 3))

    gamma = np.sqrt(0.5 * mu * s)
    rho = (r1_norm - r2_norm) / c
    sigma = np.sqrt(1.0 - rho**2)

    for (i, x) in enumerate(x_all):
        # Calculate Velocity Norm
        y = np.sqrt(1.0 - lbd**2 * (1.0 - x**2))
        v_r1 = gamma * ((lbd * y - x) - rho * (lbd * y + x)) / r1_norm
        v_r2 = -gamma * ((lbd * y - x) + rho * (lbd * y + x)) / r2_norm
        v_t1 = gamma * sigma * (y + lbd * x) / r1_norm
        v_t2 = gamma * sigma * (y + lbd * x) / r2_norm

        # Calculate Velocity Vectors
        vvec_1_all[i, :] = (v_r1 * ivec_r1 + v_t1 * ivec_t1)
        vvec_2_all[i, :] = (v_r2 * ivec_r2 + v_t2 * ivec_t2)

    # Output
    return vvec_1_all, vvec_2_all, len(x_all)


def _x2tof(x, lbd, m_max, dx_battin=0.01, dx_lagrange=0.2):
    dx = abs(x - 1.0)

    if (dx > dx_battin) and (dx < dx_lagrange):
        # Use Lagrange TOF Expression
        a = 1.0 / (1.0 - x**2) # Non dimensional semi-major axis
        if a > 0: # Ellipse
            alpha = 2.0 * np.arccos(x)
            beta = 2.0 * np.arcsin(lbd / np.sqrt(a))
            tof = 0.5 * a**(3 / 2) * ((alpha - np.sin(alpha)) - (beta - np.sin(beta)) + 2.0 * m_max * np.pi) # Non dimensinal TOF
        else: # Hyperbola
            alpha = 2.0 * np.arccosh(x)
            beta = 2.0 * np.arcsinh(lbd / np.sqrt(-a))
            tof = -0.5 * (-a)**(3 / 2) * ((alpha - np.sinh(alpha)) - (beta - np.sinh(beta))) # Non dimensinal TOF

    elif (dx < dx_battin):
        # Use Battin Series TOF Expression
        rho = abs(x**2 - 1.0)
        z = np.sqrt(1.0 + lbd**2 * (x**2 - 1.0))
        η = z - lbd * x
        s1 = 0.5 * (1.0 - lbd - x * η)
        q = 4.0 / 3.0 * _hypergeometric_2F1(s1)
        tof = (η**3 * q + 4.0 * lbd * η) / 2.0 + m_max * np.pi / (rho**1.5)

    else:
        # Use Lancaster TOF Expression
        e = x**2 - 1.0
        z = np.sqrt(1.0 + lbd**2 * e)
        y = np.sqrt(abs(e))
        if (e < 0.0):
            d = m_max * np.pi + np.arccos(x * z - lbd * e)
        else:
            d_temp = y * (z - lbd * x) + (x * z - lbd * e)
            if d_temp > 0.0:
                d = np.log(d_temp)
            else:
                warnings.warn("Fail to Calculate TOF using Lancaster TOF Expression.")
                return np.nan

        tof = (x - lbd * z - d / y) / e

    return tof


def _hypergeometric_2F1(z, a=3.0, b=1.0, c=2.5, tol=1.0e-11):
    # Initilization
    sj, cj, sj1, cj1, err = 1.0, 1.0, 0.0, 0.0, 1.0

    # Iteration
    j = 0
    while (err > tol):
        cj1 = cj * (a + j) * (b + j) / (c + j) * z / (j + 1)
        sj1 = sj + cj1
        err = abs(cj1)
        sj, cj = sj1, cj1
        j += 1
        if j > 1000:
            raise RuntimeError("Hypergeometric def Reaches Maximum Iteration.")

    return sj


def _find_xy(lbd, tof, m_multi_revs):
    # Requirements
    if abs(lbd) >= 1:
        raise ValueError("Lambda must be less than 1.")
    if tof < 0:
        raise ValueError("Non dimensional tof must be a positive number.")

    # ----------------
    # 1. Detect m_max
    m_max = np.floor(tof / np.pi)
    t_00 = np.arccos(lbd) + lbd * np.sqrt(1.0 - lbd**2) # Eq.(19) in Ref[1]
    t_0m = t_00 + m_max * np.pi # Minimum Energy Transfer Time: Eq.(19) in Ref[1]
    t_1 = 2.0 / 3.0 * (1.0 - lbd**3)

    if (m_multi_revs > 0) and (m_max > 0) and (tof < t_0m):
        x_tmin, t_min = _find_tof_min_by_halley_method(0.0, t_0m, lbd, m_max)

        if (tof < t_min) or np.isnan(t_min):
            m_max -= 1

    # Crop m_max to m_multi_revs
    m_max = int(min(m_multi_revs, m_max))

    # ----------------
    # 2. Calculate All Solutions in x,y
    x_all = np.zeros(2 * m_max + 1)
    iter_all = np.zeros(2 * m_max + 1, dtype=int)

    # 2.1. Single Revolution Solution
    # Initial guess
    if tof >= t_00:
        x_all[0] = -(tof - t_00) / (tof - t_00 + 4.0) #(t_00/tof)**(2/3) - 1.0
    elif tof <= t_1:
        x_all[0] = (5.0 * t_1 * (t_1 - tof)) / (2.0 * tof * (1 - lbd**5)) + 1.0
    else:
        x_all[0] = (tof / t_00)**(np.log(2.0) / np.log(t_1 / t_00)) - 1.0  #(t_00/tof)**(log2(t_1/t_00)) - 1.0


    # Householder iterations
    iter_all[0], x_all[0] = _find_x_by_householder(tof, x_all[0], lbd, 0, 1.0e-5)

    # 2.2. Multi Revolution Solution
    for i in range(m_max):
        # Left Householder iterations
        temp = (((i + 1) * np.pi + np.pi) / (8.0 * tof))**(2.0 / 3.0)
        x_all[2*i+1] = (temp - 1.0) / (temp + 1.0)
        iter_all[2*i+1], x_all[2*i+1] = _find_x_by_householder(tof, x_all[2*i+1], lbd, i+1)

        # Right Householder iterations
        temp = ((8.0 * tof) / ((i + 1) * np.pi))**(2.0 / 3.0)
        x_all[2*(i+1)] = (temp - 1.0) / (temp + 1.0)
        iter_all[2*(i+1)], x_all[2*(i+1)] = _find_x_by_householder(tof, x_all[2*(i+1)], lbd, i+1)

    return iter_all, x_all


def _find_x_by_householder(tof, xn, lbd, m, dx_tol=1.0e-8, max_iter=15):
    # Finds x that satisfies f(x)=tn(x)-tof=0 by Householder's method
    iter = 0
    while True:
        tn = _x2tof(xn, lbd, m)

        # Cannot be calculated
        if np.isnan(tn):
            return iter, np.nan

        # Eqs.(22) in Ref[1]
        def f(x, t):
            return t - tof
        def df_dx(x, t):
            return (3.0 * t * x - 2.0 + 2.0 * lbd ** 3 * x / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2))) / (1.0 - x ** 2)
        def d2f_dx2(x, t):
            return (3.0 * t + 5.0 * x * df_dx(x, t) + 2.0 * (1.0 - lbd ** 2) * lbd ** 3 / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2)) ** 3) / (1.0 - x ** 2)
        def d3f_dx3(x, t):
            return (7.0 * x * d2f_dx2(x, t) + 8.0 * df_dx(x, t) - 6.0 * (1.0 - lbd ** 2) * lbd ** 5 * x / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2)) ** 5) / (1.0 - x ** 2)

        # Householder's Method
        xn_new = xn - f(xn, tn) * (
            (df_dx(xn, tn)**2 - 0.5 * f(xn, tn) * d2f_dx2(xn, tn))
            /
            (df_dx(xn, tn) * ((df_dx(xn, tn)**2) - f(xn, tn) * d2f_dx2(xn, tn)) + d3f_dx3(xn, tn) * (f(xn, tn)**2) / 6.0)
        )

        # Break condition
        if abs(xn_new - xn) < dx_tol:
            tn = _x2tof(xn_new, lbd, m)
            return iter, xn_new
        elif iter > max_iter:
            warnings.warn("Householder iteration reaches maximum iteration!")
            tn = _x2tof(xn_new, lbd, m)
            return iter, xn_new

        # Update the value
        xn = xn_new
        iter += 1


def _find_tof_min_by_halley_method(xn, tn, lbd, m_max, dx_tol=1.0e-13, max_iter=12):
    # Find minimum value of transfer time by Halley's method
    iter = 0
    while True:
        # Eqs.(22) in Ref[1]
        def dt_dx(x, t):
            return (3.0 * t * x - 2.0 + 2.0 * lbd ** 3 * x / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2))) / (1.0 - x ** 2)
        def d2t_dx2(x, t):
            return (3.0 * t + 5.0 * x * dt_dx(x, t) + 2.0 * (1.0 - lbd ** 2) * lbd ** 3 / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2)) ** 3) / (1.0 - x ** 2)
        def d3t_dx3(x, t):
            return (7.0 * x * d2t_dx2(x, t) + 8.0 * dt_dx(x, t) - 6.0 * (1.0 - lbd ** 2) * lbd ** 5 * x / np.sqrt(1.0 - lbd ** 2 * (1.0 - x ** 2)) ** 5) / (1.0 - x ** 2)

        # Halley's Method
        xn_new = xn - (2.0 * dt_dx(xn, tn) * d2t_dx2(xn, tn)) / (2.0 * (d2t_dx2(xn, tn)**2) - dt_dx(xn, tn) * d3t_dx3(xn, tn))

        # Break condition
        if abs(xn_new - xn) < dx_tol:
            tn = _x2tof(xn_new, lbd, m_max)
            return xn_new, tn
        elif iter > max_iter:
            warnings.warn("Halley iteration reaches maximum iteration!")
            tn = _x2tof(xn_new, lbd, m_max)
            return xn_new, tn

        # Update the value
        tn = _x2tof(xn_new, lbd, m_max)
        xn = xn_new
        iter += 1

        # Cannot be calculated
        if np.isnan(tn):
            return xn_new, tn


if __name__ == "__main__":

        # Example Earth (2031/03/01 00:00:00 UTC) to Mars (2032/01/01 00:00:00 UTC)
        mu_1 = 1.327124400419393e11
        r1_1 = np.array([-1.3905317294572383e8, 5.125349129587864e7, -2753.8250392638147])
        r2_1 = np.array([2.0803122655020934e8, 1.8008145720647745e7, -4.721309430616008e6])
        tof_1 = 2.643839999853146e7

        # Solve Lambert's Problem
        v1_1, v2_1, num_sol_1 = lambert_solver(r1_1, r2_1, tof_1, mu_1, 2)

        assert num_sol_1 == 1
        assert np.allclose(v1_1[0, :], [-12.146901903784691, -30.249196666164899, 1.732823081867305], rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_1[0, :], [-5.062557143526494, 22.773747568054521, -1.043526507750940],  rtol=1e-14, atol=1e-14)
        print("Example 1: OK")

        # Example used in pykep Tutorial
        mu_2 = 1.32712440018e+11
        r1_2 = np.array([-25216645.728283768, 144924279.08132498, -38.276915766745136])
        r2_2 = np.array([177909722.13822687, -105168473.55967535, -6575244.5882079229])
        tof_2 = 55296000

        # Solve Lambert's Problem
        v1_2, v2_2, num_sol_2 = lambert_solver(r1_2, r2_2, tof_2, mu_2, 2)

        assert num_sol_2 == 3
        assert np.allclose(v1_2[0, :], [-35.499818966294406, 0.40742940300237478, 1.4595203390221081], rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_2[0, :], [3.0376984971968541, 27.064521805431243, -0.31914600319656404], rtol=1e-14, atol=1e-14)
        assert np.allclose(v1_2[1, :], [-27.81451513546858, -15.91479886802767, 1.2599207890195353],   rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_2[1, :], [16.828356734556462, 14.965476740707376, -0.80053299320499104], rtol=1e-14, atol=1e-14)
        assert np.allclose(v1_2[2, :], [-26.997843155205635, -17.749606456280773, 1.2394297881268258], rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_2[2, :], [18.383111848123743, 13.641214500991762, -0.85508959337278111], rtol=1e-14, atol=1e-14)
        print("Example 2: OK")

        # Example used in pykep Tutorial
        mu_3 = 1.32712440018e+11
        r1_3 = np.array([-2.4813057806272294e7, 1.449945066043632e8, -8.215121737733483e3])
        r2_3 = np.array([-1.603115587336787e8, 1.846445550220809e7, -4.784110675475702e6])
        tof_3 = 5.184e7

        # Solve Lambert's Problem
        v1_3, v2_3, num_sol_3 = lambert_solver(r1_3, r2_3, tof_3, mu_3, 2)

        assert num_sol_3 == 3
        assert np.allclose(v1_3[0, :], [-21.240058935496876, 27.686763544075136, -0.50382285400886622],  rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_3[0, :], [27.049503001979407, -18.040826055804158, 0.73033361456994976],   rtol=1e-14, atol=1e-14)
        assert np.allclose(v1_3[1, :], [-22.652716083100953, 19.414672723686053, -0.58943525930414978],  rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_3[1, :], [18.571456009600071, -19.622376845961542, 0.46414809512526563],   rtol=1e-14, atol=1e-14)
        assert np.allclose(v1_3[2, :], [-32.563106582947683, -7.2041105230019612, -1.0282043543619368],  rtol=1e-14, atol=1e-14)
        assert np.allclose(v2_3[2, :], [-8.9841267620155668, -29.532122399180287, -0.42558653591556856], rtol=1e-14, atol=1e-14)
        print("Example 3: OK")

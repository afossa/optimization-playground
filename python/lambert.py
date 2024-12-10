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

"""
Solves the Lambert's problem, i.e. computes the initial and final velocity
vectors for given initial and final position vectors and time of flight.
See [Battin1999, Izzo2015](@citet).

For `N` maximum number of multiple revolutions, the total number of solutions
is `2N+1`. The output velocity vectors are organized as follows:

| `v1[1,:]` |  `v1[2,:]`   |   `v1[3,:]`   |  `v1[4,:]`   | --> |
|:---------:|:------------:|:-------------:|:------------:|:---:|
|   0 rev   | 1 rev (left) | 1 rev (right) | 1 rev (left) | --> |

## Arguments ##

- `r1`: Initial position vector.
- `r2`: Final position vector.
- `tof`: Time of flight.
- `μ`: Gravitational parameter.
- `multi_revs`: Maximum number of multiple revolutions to compute.

## Keyword Arguments ##

- `is_retrograde`: Flag to indicate if the motion is retrograde (default to `false`).

## Returns ##

- `v1`: Initial velocity vector.
- `v2`: Final velocity vector.
- `num_sol`: Number of solutions.

"""
def lambert_solver(r1,r2,tof,μ,multi_revs=0,is_retrograde=False):

    # 0. Sanity Check
    if tof <= 0:
        raise ValueError("Time of flight must be positive!")
    if μ <= 0:
        raise ValueError("Gravity parameter must be positive!")

    # 1. Preparation
    # Variables used in Lambert's Problem
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    c = np.linalg.norm(r2 - r1)
    s = 0.5 * (r1_norm + r2_norm + c)
    λ2 = 1.0 - c / s
    tof_nomdim = np.sqrt(2.0 * μ / (s**3)) * tof

    # Basis vectors
    ivec_r1 = r1 / r1_norm
    ivec_r2 = r2 / r2_norm
    ivec_h = np.linalg.cross(ivec_r1, ivec_r2)
    ivec_h = ivec_h / np.linalg.norm(ivec_h) # Angular momentum vector
    # NOTE: ivec_h cannot be defined if r1 // r2

    if ivec_h[3] == 0.0:
        raise ValueError("The angular momentum vector has no z component, impossible to define automatically clock or counterclockwise!")
    elif ivec_h[3] < 0.0:
        λ = -np.sqrt(λ2)
        ivec_t1 = -np.linalg.cross(ivec_h, ivec_r1)
        ivec_t2 = -np.linalg.cross(ivec_h, ivec_r2)
    else:
        λ = np.sqrt(λ2)
        ivec_t1 = np.linalg.cross(ivec_h, ivec_r1)
        ivec_t2 = np.linalg.cross(ivec_h, ivec_r2)

    if is_retrograde: # Retrograde Motion
        λ = -λ
        ivec_t1 = -ivec_t1
        ivec_t2 = -ivec_t2

    # 2. Calculate x
    _, x_all = _find_xy(λ, tof_nomdim, multi_revs)

    # 3. Calculate Terminal Velocities
    vvec_1_all = np.zeros(len(x_all), 3)
    vvec_2_all = np.zeros(len(x_all), 3)

    γ = np.sqrt(0.5 * μ * s)
    ρ = (r1_norm - r2_norm) / c
    σ = np.sqrt(1.0 - ρ**2)

    for (i, x) in enumerate(x_all):
        # Calculate Velocity Norm
        y = np.sqrt(1.0 - λ**2 * (1.0 - x**2))
        v_r1 = γ * ((λ * y - x) - ρ * (λ * y + x)) / r1_norm
        v_r2 = -γ * ((λ * y - x) + ρ * (λ * y + x)) / r2_norm
        v_t1 = γ * σ * (y + λ * x) / r1_norm
        v_t2 = γ * σ * (y + λ * x) / r2_norm

        # Calculate Velocity Vectors
        vvec_1_all[i, :] = (v_r1 * ivec_r1 + v_t1 * ivec_t1)
        vvec_2_all[i, :] = (v_r2 * ivec_r2 + v_t2 * ivec_t2)

    # Output
    return vvec_1_all, vvec_2_all, len(x_all)


def _x2tof(x, λ, m_max, Δx_battin=0.01, Δx_lagrange=0.2):
    Δx = abs(x - 1.0)

    if (Δx > Δx_battin) and (Δx < Δx_lagrange):
        # Use Lagrange TOF Expression
        a = 1.0 / (1.0 - x**2) # Non dimensional semi-major axis
        if a > 0: # Ellipse
            α = 2.0 * np.acos(x)
            β = 2.0 * np.asin(λ / np.sqrt(a))
            tof = 0.5 * a**(3 / 2) * ((α - np.sin(α)) - (β - np.sin(β)) + 2.0 * m_max * np.pi) # Non dimensinal TOF
        else: # Hyperbola
            α = 2.0 * np.acosh(x)
            β = 2.0 * np.asinh(λ / np.sqrt(-a))
            tof = -0.5 * (-a)**(3 / 2) * ((α - np.sinh(α)) - (β - np.sinh(β))) # Non dimensinal TOF

    elif (Δx < Δx_battin):
        # Use Battin Series TOF Expression
        ρ = abs(x**2 - 1.0)
        z = np.sqrt(1.0 + λ**2 * (x**2 - 1.0))
        η = z - λ * x
        s1 = 0.5 * (1.0 - λ - x * η)
        q = 4.0 / 3.0 * _hypergeometric_2F1(s1)
        tof = (η**3 * q + 4.0 * λ * η) / 2.0 + m_max * np.pi / (ρ**1.5)

    else:
        # Use Lancaster TOF Expression
        e = x**2 - 1.0
        z = np.sqrt(1.0 + λ**2 * e)
        y = np.sqrt(abs(e))
        if (e < 0.0):
            d = m_max * np.pi + np.acos(x * z - λ * e)
        else:
            d_temp = y * (z - λ * x) + (x * z - λ * e)
            if d_temp > 0.0:
                d = np.log(d_temp)
            else:
                warnings.warn("Fail to Calculate TOF using Lancaster TOF Expression.")
                return np.nan

        tof = (x - λ * z - d / y) / e

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


def _find_xy(λ, tof, m_multi_revs):
    # Requirements
    if abs(λ) >= 1:
        raise ValueError("Lambda must be less than 1.")
    if tof < 0:
        raise ValueError("Non dimensional tof must be a positive number.")

    # ----------------
    # 1. Detect m_max
    m_max = np.floor(tof / np.pi)
    t_00 = np.acos(λ) + λ * np.sqrt(1.0 - λ**2) # Eq.(19) in Ref[1]
    t_0m = t_00 + m_max * np.pi # Minimum Energy Transfer Time: Eq.(19) in Ref[1]
    t_1 = 2.0 / 3.0 * (1.0 - λ**3)

    if (m_multi_revs > 0) and (m_max > 0) and (tof < t_0m):
        x_tmin, t_min = _find_tof_min_by_halley_method(0.0, t_0m, λ, m_max)

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
        x_all[1] = -(tof - t_00) / (tof - t_00 + 4.0) #(t_00/tof)**(2/3) - 1.0
    elif tof <= t_1:
        x_all[1] = (5.0 * t_1 * (t_1 - tof)) / (2.0 * tof * (1 - λ**5)) + 1.0
    else:
        x_all[1] = (tof / t_00)**(np.log(2.0) / np.log(t_1 / t_00)) - 1.0  #(t_00/tof)**(log2(t_1/t_00)) - 1.0


    # Householder iterations
    iter_all[1], x_all[1] = _find_x_by_householder(tof, x_all[1], λ, 0, 1.0e-5)

    # 2.2. Multi Revolution Solution
    for i in range(1, m_max + 1):
        # Left Householder iterations
        temp = ((i * np.pi + np.pi) / (8.0 * tof))**(2.0 / 3.0)
        x_all[2*i] = (temp - 1.0) / (temp + 1.0)
        iter_all[2*i], x_all[2*i] = _find_x_by_householder(tof, x_all[2*i], λ, i)

        # Right Householder iterations
        temp = ((8.0 * tof) / (i * np.pi))**(2.0 / 3.0)
        x_all[2*i+1] = (temp - 1.0) / (temp + 1.0)
        iter_all[2*i+1], x_all[2*i+1] = _find_x_by_householder(tof, x_all[2*i+1], λ, i)

    return iter_all, x_all


def _find_x_by_householder(tof, xn, λ, m, tol_Δx=1.0e-8, max_iter=15):
    # Finds x that satisfies f(x)=tn(x)-tof=0 by Householder's method
    iter = 0
    while True:
        tn = _x2tof(xn, λ, m)

        # Cannot be calculated
        if np.isnan(tn):
            return iter, np.nan

        # Eqs.(22) in Ref[1]
        def f(x, t):
            return t - tof
        def df_dx(x, t):
            return (3.0 * t * x - 2.0 + 2.0 * λ ** 3 * x / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2))) / (1.0 - x ** 2)
        def d2f_dx2(x, t):
            return (3.0 * t + 5.0 * x * df_dx(x, t) + 2.0 * (1.0 - λ ** 2) * λ ** 3 / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2)) ** 3) / (1.0 - x ** 2)
        def d3f_dx3(x, t):
            return (7.0 * x * d2f_dx2(x, t) + 8.0 * df_dx(x, t) - 6.0 * (1.0 - λ ** 2) * λ ** 5 * x / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2)) ** 5) / (1.0 - x ** 2)

        # Householder's Method
        xn_new = xn - f(xn, tn) * (
            (df_dx(xn, tn)**2 - 0.5 * f(xn, tn) * d2f_dx2(xn, tn))
            /
            (df_dx(xn, tn) * ((df_dx(xn, tn)**2) - f(xn, tn) * d2f_dx2(xn, tn)) + d3f_dx3(xn, tn) * (f(xn, tn)**2) / 6.0)
        )

        # Break condition
        if abs(xn_new - xn) < tol_Δx:
            tn = _x2tof(xn_new, λ, m)
            return iter, xn_new
        elif iter > max_iter:
            warnings.warn("Householder iteration reaches maximum iteration!")
            tn = _x2tof(xn_new, λ, m)
            return iter, xn_new

        # Update the value
        xn = xn_new
        iter += 1


def _find_tof_min_by_halley_method(xn, tn, λ, m_max, tol_Δx=1.0e-13, max_iter=12):
    # Find minimum value of transfer time by Halley's method
    iter = 0
    while True:
        # Eqs.(22) in Ref[1]
        def dt_dx(x, t):
            return (3.0 * t * x - 2.0 + 2.0 * λ ** 3 * x / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2))) / (1.0 - x ** 2)
        def d2t_dx2(x, t):
            return (3.0 * t + 5.0 * x * dt_dx(x, t) + 2.0 * (1.0 - λ ** 2) * λ ** 3 / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2)) ** 3) / (1.0 - x ** 2)
        def d3t_dx3(x, t):
            return (7.0 * x * d2t_dx2(x, t) + 8.0 * dt_dx(x, t) - 6.0 * (1.0 - λ ** 2) * λ ** 5 * x / np.sqrt(1.0 - λ ** 2 * (1.0 - x ** 2)) ** 5) / (1.0 - x ** 2)

        # Halley's Method
        xn_new = xn - (2.0 * dt_dx(xn, tn) * d2t_dx2(xn, tn)) / (2.0 * (d2t_dx2(xn, tn)**2) - dt_dx(xn, tn) * d3t_dx3(xn, tn))

        # Break condition
        if abs(xn_new - xn) < tol_Δx:
            tn = _x2tof(xn_new, λ, m_max)
            return xn_new, tn
        elif iter > max_iter:
            warnings.warn("Halley iteration reaches maximum iteration!")
            tn = _x2tof(xn_new, λ, m_max)
            return xn_new, tn

        # Update the value
        tn = _x2tof(xn_new, λ, m_max)
        xn = xn_new
        iter += 1

        # Cannot be calculated
        if np.isnan(tn):
            return xn_new, tn

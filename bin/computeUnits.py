__author__ = 'pslii'
import numpy as np

# constants in cgs units
c_MSUN = 1.99e33  # grams
c_RSUN = 6.96e10  # cm
c_G = 6.67e-8  # dyn cm^2 / g^2
c_AU = 1.496e13  # cm
c_kBoltzmann = 1.380e-16  # erg/K
c_MH = 1.6733e-24  # g mass of hydrogen
c_RGAS = c_kBoltzmann / c_MH  # erg/ K g
M_stars = np.array([1, 1, 1, .53, .49, 2.11]) * c_MSUN
R_gaps = np.array([5 * c_AU, c_AU, 0.1 * c_AU, 8.0 * c_RSUN, 6.4 * c_RSUN, 3.2 * c_RSUN])

# parameters
print "$M_*$ [$M_\sun$] & $r_{gap}$ & $v_0$ [km/s] &" \
      " $\Sigma_0$ [g/$cm^2$] & $T_0$ [K] & $P_0$ [days] & $\Omega_0$ [$s^{-1}$] \\\\"
for M_star, R_gap in zip(M_stars, R_gaps):
    # parameters
    GM = 1.0 * c_G * M_star
    GM_p = 4.5e-5 * GM
    AM_p = 4.5e-5

    M0 = (GM_p / GM) / AM_p * M_star
    r0 = R_gap
    v0 = np.sqrt(GM / r0)
    t0 = r0 / v0
    P0 = 2 * np.pi * t0
    rho0 = M0 / r0 ** 3
    p0 = rho0 * v0 ** 2
    n0 = rho0 / c_MH

    Pi0 = rho0 * v0 ** 2 * r0
    Sigma0 = M0 / r0 ** 2
    T0 = p0 / (c_RGAS * rho0)
    Omega0 = v0 / r0

    print "{:.2e} & {:.2e} & {:.3e} & {:.2e} & {:.2e} & {:.2e} & {:.2e} \\\\".format(M0 / c_MSUN,
                                                                                     r0 / c_RSUN,
                                                                                     v0 / 1e5,
                                                                                     Sigma0,
                                                                                     T0,
                                                                                     P0 / (3600 * 24),
                                                                                     Omega0)
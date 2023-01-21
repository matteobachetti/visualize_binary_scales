import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from astropy import units as u
from astropy import constants as c
from astropy.table import Table, vstack
from astropy.utils import isiterable
from scipy.interpolate import interp1d
import copy


def dyn_timescale(M, R):
    return 30 * u.min * ((R / c.R_sun) ** (1.5) * (M / c.M_sun) ** (-0.5)).to("")


def thermal_timescale(M):
    return 30e6 * u.year * ((M / c.M_sun) ** (-2)).to("")


def nuclear_timescale(M):
    return 10e9 * u.year * ((M / c.M_sun) ** (-2.5)).to("")


def surface_angle(radius, separation):
    return np.arcsin((radius / separation).to(""))


def eclipse_inclination(radius, distance, use_cos=False):
    if use_cos:
        return np.arccos((radius / distance).to(""))

    return np.pi / 2 * u.rad - surface_angle(radius, distance)


def star_rotation_from_superorb(porb, psuperorb):
    """See Van den Heuvel 1994 in Interacting Binaries."""

    # Psuperorb = 2 pi / (Omega_orb - Omega_rot)
    # (Omega_orb - Omega_rot) = 2 pi / Psuperorb
    # Or the same, with Psuperorb = 2 pi / (Omega_rot - Omega_orb)
    # Omega_rot_low = Omega_orb - Omega_superorb
    # Omega_rot_sup = Omega_orb + Omega_superorb

    omegaorb = 2 * np.pi / porb
    omegasuperorb = 2 * np.pi / psuperorb
    return 2 * np.pi / (omegaorb + omegasuperorb), 2 * np.pi / (omegaorb - omegasuperorb)


def mass_function(porb, a1sini):
    omegaorb = 2 * np.pi / porb
    K = omegaorb * a1sini
    return K**3 / omegaorb / c.G


def enforce_distance_unit(val):
    """Convert seconds/light seconds to meters."""
    try:
        # If it's in light seconds, multiply by c
        val.to(u.s)
        val = val * c.c
    except:
        pass
    return val.to(u.Rsun)


def a_from_a1(a1, m1, m2):
    a = a1 * (1 + m1 / m2)
    return enforce_distance_unit(a)


def a1_from_a(a, m1, m2):
    a1 = a * m2 / (m1 + m2)
    return enforce_distance_unit(a1)


def a_from_a2(a2, m1, m2):
    a = a2 * (1 + m2 / m1)
    return enforce_distance_unit(a)


def a2_from_a(a, m1, m2):
    a2 = a * m1 / (m1 + m2)
    return enforce_distance_unit(a2)


def a2_from_a1(a1, m1, m2):
    return enforce_distance_unit(a1 * m1 / m2)


def a1_from_a2(a2, m1, m2):
    return enforce_distance_unit(a2 * m2 / m1)


def roche_lobe(m1=1.4 * c.M_sun, m2=5 * c.M_sun, a1=22.5 * u.s):
    m = m1 + m2
    r = m2 / m1
    a = a_from_a1(a1, m1, m2)
    return (a * 0.49 * r ** (2 / 3) / (0.6 * r ** (2 / 3) + np.log(1 + r ** (1 / 3)))).to(u.km)


def stellar_radius(mass):
    if isiterable(mass):
        return [stellar_radius(m).to(u.R_sun).value for m in mass] * u.R_sun

    if mass > 1 * u.M_sun:
        csi = 0.57
    else:
        csi = 0.8

    r = c.R_sun * ((mass / c.M_sun) ** csi).to("")
    return r


def inclination_from_mass_f_and_m(mf, m1, m2):
    f_trial_noinc = m2**3 / (m1 + m2) ** 2

    # f = f_trial * sin^3 i
    inclination = np.arcsin(((mf / f_trial_noinc) ** (1 / 3)).to("").value) * u.rad
    return inclination.to(u.rad)


def get_inclination(period, a1sini, m1, m2):
    f = mass_function(period, a1sini)
    return inclination_from_mass_f_and_m(f, m1, m2)


def get_m2_from_inclination(inclination, period, a1sini, m1, bracket=(0.1, 100000)):
    from scipy.optimize import minimize_scalar

    if isiterable(inclination):
        return [
            get_m2_from_inclination(incl, period, a1sini, m1, bracket=bracket).value
            for incl in inclination
        ] * u.Msun

    inclination = inclination.to(u.rad)
    f = mass_function(period, a1sini)

    def func_to_minimize(m2):
        incl = inclination_from_mass_f_and_m(f, m1, m2 * u.Msun).to(u.rad)
        if np.isnan(incl.value):
            return np.inf

        return np.abs(incl - inclination)

    res = minimize_scalar(func_to_minimize, bracket=bracket)

    return res.x * u.Msun


def get_valid_inclination(
    m1=1.4 * u.M_sun,
    period=2.54 * u.d,
    a1sini=22.215 * u.s * c.c,
    eclipsing=False,
    m2max=100 * u.Msun,
):
    m2_eclipse = get_eclipse_mass(period, a1sini, m1)
    eclipse_incl = get_inclination(period, a1sini, m1, m2_eclipse)

    if eclipsing:
        incl_min = eclipse_incl.to(u.rad)
        incl_max = (90 * u.deg).to(u.rad)
    else:
        incl_min = get_inclination(period, a1sini, m1, m2max).to(u.rad)
        incl_max = eclipse_incl.to(u.rad)

    return incl_min, incl_max


def get_m2_from_inclination_func(
    m1=1.4 * u.M_sun, period=2.54 * u.d, a1sini=22.215 * u.s * c.c, plot=True, m2max=100 * u.Msun
):

    incl_min = get_inclination(period, a1sini, m1, m2max).to(u.rad)
    incl_max = (90 * u.deg).to(u.rad)

    m2min = get_m2_from_inclination(incl_max, period, a1sini, m1)
    m2max = get_m2_from_inclination(incl_min, period, a1sini, m1)

    m2_grid = np.geomspace(m2min, m2max, 1001)
    incl_grid = get_inclination(period, a1sini, m1, m2_grid)
    m2_func = interp1d(incl_grid.to(u.rad).value, m2_grid.to(u.Msun).value)

    def m2_func_units(incl):
        if incl > incl_grid.max():
            incl = incl_grid.max()
        if incl < incl_grid.min():
            incl = incl_grid.min()

        return m2_func(incl.to(u.rad).value) * u.Msun

    return (incl_min, incl_max), m2_func_units


def generate_mass_values(N, **kwargs):
    (incl_min, incl_max), m2_func = get_m2_from_inclination_func(**kwargs)
    incl_vals = np.random.uniform(incl_min, incl_max, N)

    return m2_func(incl_vals)


def get_eclipse_mass(period, a1sini, m1, bracket=(0.1, 10), r2=None):
    from scipy.optimize import minimize_scalar

    f = mass_function(period, a1sini)

    def func_to_minimize(m2):
        m2 = m2 * u.Msun
        incl = inclination_from_mass_f_and_m(f, m1, m2)
        if np.isnan(incl.value):
            return np.inf

        a1 = a1sini / np.sin(incl)

        # print((a1 * 2 * np.pi / period / c.c).to(""))
        rstar = r2
        if r2 is None:
            rstar = roche_lobe(m1=m1, m2=m2, a1=a1)
        a = a_from_a1(a1, m1, m2)
        return np.abs(np.cos(incl) - (rstar / a).to(""))

    res = minimize_scalar(func_to_minimize, bracket=bracket)

    return res.x * u.Msun


def get_eclipse_inclination(period, a1sini, m1, r2=None):
    m2 = get_eclipse_mass(period, a1sini, m1, r2=r2)
    return get_inclination(period, a1sini, m1, m2)


def get_mass_quantiles(
    eclipse=False,
    m1=1.4 * u.M_sun,
    period=2.54 * u.d,
    a1sini=22.215 * u.s * c.c,
    verbose=False,
    **kwargs,
):
    _, m2_func = get_m2_from_inclination_func(m1=m1, period=period, a1sini=a1sini, **kwargs)
    (incl_min, incl_max) = get_valid_inclination(
        m1=m1, period=period, a1sini=a1sini, eclipsing=eclipse, **kwargs
    )

    eclipse_incl = get_eclipse_inclination(period=period, m1=m1, a1sini=a1sini)
    incl_interval = incl_max - incl_min
    if verbose:
        print("Quantiles:")
    results = {}
    quantiles = [0.05, 0.16, 0.32, 0.5, 0.68, 0.84, 0.95]
    for quantile in quantiles:
        incl = incl_interval * quantile + incl_min
        if verbose:
            print(f"{quantile * 100:.0f}%: {m2_func(incl):.2f}")
        results[f"{quantile * 100:g}%"] = m2_func(incl)
    return results


def test_m2_inclination_roundtrip():
    period = 2.53 * u.d
    a1sini = 22.25 * u.s * c.c
    m1 = 1.4 * u.Msun

    for m2 in np.random.uniform(4, 100, 10) * u.Msun:
        incl = get_inclination(period=period, a1sini=a1sini, m1=m1, m2=m2)
        m2_out = get_m2_from_inclination(incl, period=period, a1sini=a1sini, m1=m1)

        assert np.isclose(m2_out, m2)

    print("Passed")


def specific_angular_momentum(a, Porb):
    Omega = 2 * np.pi / Porb
    return a**2 * Omega


def r_co(p, m1=1.4 * u.M_sun):
    const = c.G * m1
    return ((const * (p / (2.0 * np.pi)) ** 2) ** (1 / 3)).to(u.km)

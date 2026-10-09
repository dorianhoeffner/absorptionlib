"""
Thermophysical property functions for aqueous NaOH-H2O solutions.

Units (uniform across absorptionlib):
    x : salt mass fraction [kg NaOH / kg solution]
    T : temperature [°C]
    p : pressure [Pa]

All property functions accept ``prevent_errors=False``:
    False -> out-of-range inputs emit OutOfRangeWarning /
             CrystallizationWarning; invalid inputs raise ValueError.
    True  -> all warnings are suppressed and ValueError is replaced by a
             ``float('nan')`` return value (safe for optimizers).

Main sources:
    Olsson, Jernqvist, Aly (1997): "Thermophysical Properties of Aqueous
        NaOH-H2O Solutions at High Concentrations", Int. J. Thermophysics 18(3).
    Alexandrov (2004): "The Equations for Thermophysical Properties of
        Aqueous Solutions of Sodium Hydroxide".
    Wang et al. (2008): crystallization curve data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from pyXSteam.XSteam import XSteam

try:
    from .._common import (
        warn_out_of_range,
        warn_crystallization,
        suppress_warnings,
        print_documentation,
        explain_function,
        _msg,
    )
except ImportError:
    # Module was imported standalone (e.g. "import functions" from inside
    # this folder) instead of through the absorptionlib package.
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from _common import (
        warn_out_of_range,
        warn_crystallization,
        suppress_warnings,
        print_documentation,
        explain_function,
        _msg,
    )

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

_MODULE = "NaOH"

_DESCRIPTIONS = {
    "saturation_temperature":   "Boiling point temperature of the solution at given x and p [°C].",
    "saturation_pressure":      "Equilibrium (vapor) pressure of the solution at given x and T [Pa].",
    "saturation_concentration": "Saturation concentration at given p and T [kg/kg].",
    "enthalpy":                 "Specific enthalpy of the solution at given x and T [kJ/kg].",
    "density":                  "Density of the solution at given x and T [kg/m³].",
    "specific_heat_capacity":   "Specific heat capacity of the solution at given x and T [kJ/(kg K)].",
    "dynamic_viscosity":        "Dynamic viscosity of the solution at given x, T and p [Pa s].",
    "thermal_conductivity":     "Thermal conductivity of the solution at given x, T and p [W/(m K)].",
    "solubility_temperature":   "Crystallization temperature at given x [°C].",
    "dhdx":                     "Partial derivative of enthalpy w.r.t. mass fraction [kJ/kg].",
    "dhdT":                     "Partial derivative of enthalpy w.r.t. temperature [kJ/(kg K)].",
    "hxDiagram":                "Plots the enthalpy-concentration diagram.",
    "pTDiagram":                "Plots the pressure-temperature diagram.",
    "crystallization_curve":    "Plots (or returns) the crystallization curve.",
}


def documentation():
    """Print an overview of all public functions of this module."""
    print_documentation(_MODULE, "NaOH", _DESCRIPTIONS)


def explain(function_name):
    """
    Print the documentation for a specific function of this module.

    Parameters:
        function_name (str): Name of the function to explain.

    Returns:
        None
    """
    explain_function(globals(), _MODULE, function_name, _DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Property functions
# ---------------------------------------------------------------------------

def saturation_temperature(x, p, prevent_errors=False):
    """
    Boiling point temperature of an aqueous NaOH-H2O solution.

    Solves saturation_pressure(x, T) = p for T using a bracketing root
    finder (brentq) on the interval 1 °C to 200 °C.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        p (float): Pressure [Pa].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Boiling point temperature [°C] (NaN if no solution exists
        in the search interval and prevent_errors=True).

    Validity (Olsson et al. 1997), water mass fraction xi = 1 - x:
        0 <= T < 20 °C:    xi >= 0.582
        20 <= T < 60 °C:   xi >= 0.500
        60 <= T < 70 °C:   xi >= 0.353
        70 <= T < 150 °C:  xi >= 0.300
        150 <= T <= 200 °C: xi >= 0.200

    Source: Olsson, Jernqvist, Aly (1997), Int. J. Thermophysics 18(3).
    Author: Dorian Höffner
    """
    try:
        with suppress_warnings():
            T = brentq(lambda T: saturation_pressure(x, T, prevent_errors=True) - p,
                       1.0, 200.0)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_temperature",
                              f"no saturation temperature found in 1..200 °C "
                              f"for x = {x}, p = {p} Pa."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "saturation_temperature", T, t_sol, x,
                             prevent_errors)

    # validity check (ranges from Olsson et al. 1997)
    _check_olsson_range("saturation_temperature", x, T, prevent_errors)

    return float(T)


def _check_olsson_range(func, x, T, prevent_errors):
    """Warn if (x, T) is outside the validated range of Olsson et al. (1997)."""
    xi = 1.0 - x  # water mass fraction
    ranges = [(0, 20, 0.582), (20, 60, 0.500), (60, 70, 0.353),
              (70, 150, 0.300), (150, 200.0001, 0.200)]
    if T < 0 or T > 200:
        warn_out_of_range(_MODULE, func,
                          f"T = {T:.2f} °C is outside the validated range "
                          f"0..200 °C (Olsson et al. 1997). x = {x}.",
                          prevent_errors)
        return
    for Tmin, Tmax, xi_min in ranges:
        if Tmin <= T < Tmax and xi < xi_min:
            warn_out_of_range(
                _MODULE, func,
                f"values are outside the validated range (Olsson et al. "
                f"1997) for {Tmin} <= T < {Tmax:.0f} °C: requires "
                f"x <= {1 - xi_min:.3f}, got x = {x}, T = {T:.2f} °C.",
                prevent_errors)
            return


def saturation_pressure(x, T, prevent_errors=False):
    """
    Equilibrium (vapor) pressure of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Pressure [Pa].

    Validity (Olsson et al. 1997), water mass fraction xi = 1 - x:
        0 <= T < 20 °C:    xi >= 0.582
        20 <= T < 60 °C:   xi >= 0.500
        60 <= T < 70 °C:   xi >= 0.353
        70 <= T < 150 °C:  xi >= 0.300
        150 <= T <= 200 °C: xi >= 0.200

    Source: Olsson, Jernqvist, Aly (1997), Int. J. Thermophysics 18(3).
    Authors: Anna Jahnke, Roman Ziegenhardt; Python: Dorian Höffner
    """
    if not 0 < x < 1:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_pressure",
                              f"mass fraction x = {x} must be within (0, 1)."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "saturation_pressure", T, t_sol, x,
                             prevent_errors)

    # validity check (ranges from Olsson et al. 1997)
    _check_olsson_range("saturation_pressure", x, T, prevent_errors)

    xi = 1.0 - x  # water mass fraction

    # Coefficients (Olsson et al. 1997)
    k = np.array([-113.93947, 209.82305, 494.77153, 6860.8330, 2676.6433,
                  -21740.328, -34750.872, -20122.157, -4102.9890])
    l = np.array([16.240074, -11.864008, -223.47305, -1650.3997, -5997.3118,
                  -12318.744, -15303.153, -11707.480, -5364.9554, -1338.5412,
                  -137.96889])
    m = np.array([-226.80157, 293.17155, 5081.8791, 36752.126, 131262.00,
                  259399.54, 301696.22, 208617.90, 81774.024, 15648.526,
                  906.29769])

    log_xi = np.log(xi)
    a1 = np.polyval(k[::-1], log_xi)
    a2 = np.polyval(l[::-1], log_xi)
    a3 = np.polyval(m[::-1], log_xi)

    logP = (a1 + a2 * T) / (T - a3)   # log of pressure in kPa
    p = np.exp(logP) * 1000.0         # -> Pa

    return float(p)


def saturation_concentration(p, T, prevent_errors=False):
    """
    Saturation concentration of NaOH in water at given pressure and temperature.

    Solves saturation_pressure(x, T) = p for x using a bracketing root
    finder (brentq) on the interval 0.001 to 0.785 kg/kg.

    Parameters:
        p (float): Pressure [Pa].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Saturation concentration [kg NaOH / kg solution] (NaN if no
        solution exists in the search interval and prevent_errors=True).

    Author: Dorian Höffner
    """
    try:
        with suppress_warnings():
            x = brentq(lambda x: saturation_pressure(x, T, prevent_errors=True) - p,
                       0.001, 0.785)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_concentration",
                              f"no saturation concentration found in "
                              f"0.001..0.785 kg/kg for p = {p} Pa, "
                              f"T = {T} °C."))

    # validity check (ranges from Olsson et al. 1997)
    _check_olsson_range("saturation_concentration", x, T, prevent_errors)

    return float(x)


def enthalpy(x, T, prevent_errors=False):
    """
    Specific enthalpy of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Specific enthalpy [kJ/kg].

    Validity (T in °C, xi = 1 - x = water mass fraction):
        0 <= T < 4:     xi >= 0.780
        4 <= T < 10:    xi >= 0.680
        10 <= T < 15:   xi >= 0.580
        15 <= T < 26:   xi >= 0.540
        26 <= T < 37:   xi >= 0.440
        37 <= T < 48:   xi >= 0.400
        48 <= T < 60:   xi >= 0.340
        60 <= T < 71:   xi >= 0.300
        71 <= T < 82:   xi >= 0.280
        82 <= T < 93:   xi >= 0.240
        93 <= T <= 204: xi >= 0.220

    Source: Olsson, Jernqvist, Aly (1997), Int. J. Thermophysics 18(3).
    Authors: Roman Ziegenhardt, Elisabeth Thiele; Python: Dorian Höffner
    """
    xi = 1.0 - x  # water mass fraction

    # validity checks
    if T < 0 or T > 204:
        warn_out_of_range(_MODULE, "enthalpy",
                          f"T = {T:.2f} °C is outside the validated range "
                          f"0..204 °C.", prevent_errors)
    else:
        ranges = [(0, 4, 0.780), (4, 10, 0.680), (10, 15, 0.580),
                  (15, 26, 0.540), (26, 37, 0.440), (37, 48, 0.400),
                  (48, 60, 0.340), (60, 71, 0.300), (71, 82, 0.280),
                  (82, 93, 0.240), (93, 204.0001, 0.220)]
        for Tmin, Tmax, xi_min in ranges:
            if Tmin <= T < Tmax and xi < xi_min:
                warn_out_of_range(
                    _MODULE, "enthalpy",
                    f"values are outside the validated range for "
                    f"{Tmin} <= T < {Tmax:.0f} °C: requires "
                    f"x <= {1 - xi_min:.3f}, got x = {x}, T = {T:.2f} °C.",
                    prevent_errors)
                break

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "enthalpy", T, t_sol, x, prevent_errors)

    # Coefficients (Olsson et al. 1997)
    k = np.array([1288.4485, -0.49649131, -4387.8908, -4.0915144, 4938.2298,
                  7.2887292, -1841.1890, -3.0202651])
    l = np.array([2.3087919, -9.0004252, 167.59914, -1051.6368, 3394.3378,
                  -6115.0986, 6220.8249, -3348.8098, 743.87432])
    m = np.array([0.02302860, -0.37866056, 2.4529593, -8.2693542, 15.728833,
                  -16.944427, 9.6254192, -2.2410628])
    n = np.array([-8.5131313e-5, 136.52823e-5, -875.68741e-5, 2920.0398e-5,
                  -5488.2983e-5, 5841.8034e-5, -3278.7483e-5, 754.45993e-5])

    c1 = ((k[0] + k[2] * xi + k[4] * xi**2 + k[6] * xi**3)
          / (1 + k[1] * xi + k[3] * xi**2 + k[5] * xi**3 + k[7] * xi**4))
    c2 = np.polyval(l[::-1], xi)
    c3 = np.polyval(m[::-1], xi)
    c4 = np.polyval(n[::-1], xi)

    h = c1 + c2 * T + c3 * T**2 + c4 * T**3

    return float(h)


def density(x, T, prevent_errors=False):
    """
    Density of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Density [kg/m³].

    Validity:
        0 <= T < 10 °C:    0 < x <= 0.2
        10 <= T < 20 °C:   0 < x <= 0.3
        20 <= T < 60 °C:   0 < x <= 0.5
        60 <= T < 70 °C:   0 < x <= 0.6
        70 <= T < 150 °C:  0 < x <= 0.7
        150 <= T <= 200 °C: 0 < x <= 0.8

    Source: Olsson, Jernqvist, Aly (1997), Int. J. Thermophysics 18(3).
    """
    if not 0 <= x < 1:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "density",
                              f"mass fraction x = {x} must be within [0, 1)."))

    # validity check
    if T < 0 or T > 200:
        warn_out_of_range(_MODULE, "density",
                          f"T = {T:.2f} °C is outside the validated range "
                          f"0..200 °C.", prevent_errors)
    else:
        ranges = [(0, 10, 0.2), (10, 20, 0.3), (20, 60, 0.5),
                  (60, 70, 0.6), (70, 150, 0.7), (150, 200.0001, 0.8)]
        for Tmin, Tmax, x_max in ranges:
            if Tmin <= T < Tmax and x > x_max:
                warn_out_of_range(
                    _MODULE, "density",
                    f"values are outside the validated range for "
                    f"{Tmin} <= T < {Tmax:.0f} °C: requires x <= {x_max}, "
                    f"got x = {x}, T = {T:.2f} °C.", prevent_errors)
                break

    xi = 1.0 - x  # water mass fraction

    # Coefficients (Olsson et al. 1997)
    k = np.array([5007.2279636, -25131.164248, 74107.692582, -104657.48684,
                  69821.773186, -18145.911810])
    l = np.array([-64.786269079, 525.34360564, -1608.4471903, 2350.9753235,
                  -1660.9035108, 457.6437435])
    m = np.array([0.24436776978, -1.9737722344, 6.04601497138, -8.9090614947,
                  6.37146769397, -1.7816083111])

    sq = np.sqrt(xi)
    powers = np.array([1, sq, xi, xi * sq, xi**2, xi**2 * sq])
    b1 = np.dot(k, powers)
    b2 = np.dot(l, powers)
    b3 = np.dot(m, powers)

    rho = b1 + b2 * T + b3 * T**2

    return float(rho)


def specific_heat_capacity(x, T, prevent_errors=False):
    """
    Specific heat capacity of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Specific heat capacity [kJ/(kg K)].

    Validity:
        0 <= T <= 275.85 °C
        0 <= x <= 0.16 kg/kg

    Source: Alexandrov (2004), "The Equations for Thermophysical Properties
        of Aqueous Solutions of Sodium Hydroxide".
    Author: Dorian Höffner
    """
    # hard validity limits
    if T < 0 or T > 275.85 or x < 0 or x > 0.16:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "specific_heat_capacity",
                              f"inputs outside the valid range "
                              f"(0 <= T <= 275.85 °C, 0 <= x <= 0.16 kg/kg): "
                              f"x = {x}, T = {T} °C."))

    # Coefficients (Alexandrov 2004)
    a1 = [9.8555259e1, -3.4501318e2, 4.8180532e2, -3.3440616e2,
          1.1516735e2, -1.5708814e1]
    a2 = [-3.4357815e1, 1.1674552e2, -1.5776854e2, 1.0577045e2,
          -3.5099188e1, 4.5935013]
    a3 = [1.9791083, -5.3828966, 5.5124212, -2.5430046, 4.5943595e-1]
    a4 = [-5.6191575e-2, 8.5936388e-2, -1.6966718e-2, -1.4864492e-2]
    a5 = [7.9944152e-3, -1.5444457e-2, 7.5030322e-3]

    cp_water = steamTable.CpL_t(T)  # [kJ/(kg K)]

    M = 39.9971e-3          # molar mass of NaOH [kg/mol]
    m = x / M               # molality [mol/kg]
    t = (T + 273.15) / 273.15

    cp_diff = (m * np.polyval(a1[::-1], t)
               + m**2 * np.polyval(a2[::-1], t)
               + m**3 * np.polyval(a3[::-1], t)
               + m**4 * np.polyval(a4[::-1], t)
               + m**5 * np.polyval(a5[::-1], t))

    cp = cp_water if x <= 0.0001 else cp_water - cp_diff

    return float(cp)


def dynamic_viscosity(x, T, p, prevent_errors=False):
    """
    Dynamic viscosity of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        p (float): Pressure [Pa].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Dynamic viscosity [Pa s].

    Validity:
        0 <= T <= 250.85 °C
        0 <= x <= 0.12 kg/kg
        0 <= p <= 7 MPa

    Source: Alexandrov (2004), "The Equations for Thermophysical Properties
        of Aqueous Solutions of Sodium Hydroxide".
    """
    # hard validity limits
    if T < 0 or T > 250.85 or x < 0 or p < 0:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "dynamic_viscosity",
                              f"inputs outside the valid range "
                              f"(0 <= T <= 250.85 °C, x >= 0, p >= 0): "
                              f"x = {x}, T = {T} °C, p = {p} Pa."))
    # soft validity limits
    if p > 7e6:
        warn_out_of_range(_MODULE, "dynamic_viscosity",
                          f"p = {p:.4g} Pa is above the validated range "
                          f"(p <= 7 MPa).", prevent_errors)
    if x > 0.12:
        warn_out_of_range(_MODULE, "dynamic_viscosity",
                          f"x = {x} is above the validated range "
                          f"(x <= 0.12 kg/kg).", prevent_errors)

    # prepare inputs
    M = 39.9971e-3                      # molar mass of NaOH [kg/mol]
    m = x / M                           # molality [mol/kg]
    T0 = 293.15                         # reference temperature [K]
    t = T0 / (T + 273.15)               # [-]
    t1 = t - 1.0                        # [-]

    p_MPa = p / 1e6                     # [Pa]  -> [MPa]
    p_sw = steamTable.psat_t(T) / 1e1   # [bar] -> [MPa]

    # water viscosity (coefficients from the publication cited in Alexandrov 2004)
    c = np.array([0.0, 5.17341030, 9.81838817, 2.83021985e1, 7.02071954e1,
                  -9.92041252e2, -1.13267055e4, -5.10988292e4, -1.18863488e5,
                  -1.41053273e5, -6.78490604e4])
    d = np.array([-3.18833435e-1, -1.07314454e1, -8.61347656e1, -6.50268842e2,
                  -6.06767730e3, -4.07022741e4, -1.59650983e5, -3.53438962e5,
                  -4.11357235e5, -1.96118714e5])

    expression1 = np.polyval(c[::-1], t1)          # c[0] = 0 (no constant term)
    expression2 = np.polyval(d[::-1], t1)
    my_water = 1001.6 * (t1 + 1)**2 * np.exp(expression1) \
        + (p_MPa - p_sw) * expression2             # [µPa s]

    # NaOH contribution (Alexandrov 2004)
    b11, b21, b31, b41 = 5.7070102e-1, 4.9395013e-1, -2.0417183, 1.1654862
    b12, b22, b32 = -2.9922166e-1, 3.7957782e-1, -7.423751e-2
    b13, b23 = 4.9815412e-2, -4.8332728e-2

    expression3 = (m * (b11 * t + b21 * t**2 + b31 * t**3 + b41 * t**4)
                   + m**2 * (b12 * t + b22 * t**2 + b32 * t**3)
                   + m**3 * (b13 * t + b23 * t**2))

    my = my_water * np.exp(expression3)  # [µPa s]

    return float(my * 1e-6)  # [Pa s]


def thermal_conductivity(x, T, p, prevent_errors=False):
    """
    Thermal conductivity of an aqueous NaOH-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        p (float): Pressure [Pa].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Thermal conductivity [W/(m K)].

    Validity:
        0 <= T <= 132.85 °C
        0 <= x <= 0.2 kg/kg
        0 <= p <= 15 MPa

    Source: Alexandrov (2004), "The Equations for Thermophysical Properties
        of Aqueous Solutions of Sodium Hydroxide".
    """
    # hard validity limits
    if T < 0 or T > 132.85 or x < 0 or p < 0 or p > 15e6:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "thermal_conductivity",
                              f"inputs outside the valid range "
                              f"(0 <= T <= 132.85 °C, x >= 0, "
                              f"0 <= p <= 15 MPa): x = {x}, T = {T} °C, "
                              f"p = {p} Pa."))
    # soft validity limits
    if x > 0.2:
        warn_out_of_range(_MODULE, "thermal_conductivity",
                          f"x = {x} is above the validated range "
                          f"(x <= 0.2 kg/kg).", prevent_errors)

    # water thermal conductivity
    p_sw = steamTable.psat_t(T) / 1e1   # [bar] -> [MPa]
    p_MPa = p / 1e6                     # [Pa]  -> [MPa]
    T02 = 273.15 + 20                   # [K]
    t = T02 / (T + 273.15) - 1          # [-]

    g = np.array([5.99454842e-1, -4.82554378e-1, -4.31229616e-1,
                  -8.62555022e-1, -3.80050418e-1, 4.85828450e1, 3.35400696e2,
                  1.08007806e3, 1.67727081e3, 1.04225629e3])
    q = np.array([5.31492446e-4, 3.46658996e-4, 1.23050434e-2, 1.27873471e-1,
                  -7.40820487e-1, -1.93072528e1, -1.22835056e2,
                  -3.66150909e2, -5.31321978e2, -3.03153185e2])

    expression1 = np.polyval(g[::-1], t)
    expression2 = np.polyval(q[::-1], t)
    lambda_water = expression1 + (p_MPa - p_sw) * expression2  # [W/(m K)]

    # NaOH contribution (Alexandrov 2004)
    T0 = 403.0  # [K]
    t1 = (T + 273.15) / T0
    M = 39.9971e-3          # molar mass of NaOH [kg/mol]
    m = x / M               # molality [mol/kg]

    e01, e11, e21, e31 = 3.2900544e-1, -1.1048583, 1.2503803, -4.4228179e-1
    e02, e12, e22 = -2.1990820e-2, 5.9100989e-2, -4.4407173e-2
    e03, e13, e23 = 1.5069324e-3, -4.3273501e-3, 3.3763248e-3

    expression3 = (m * (e01 + e11 * t1 + e21 * t1**2 + e31 * t1**3)
                   + m**2 * (e02 + e12 * t1 + e22 * t1**2)
                   + m**3 * (e03 + e13 * t1 + e23 * t1**2))

    lambda_NaOH = lambda_water + expression3  # [W/(m K)]

    return float(lambda_NaOH)  # [W/(m K)]


def solubility_temperature(x, prevent_errors=False):
    """
    Crystallization (solubility) temperature of an aqueous NaOH-H2O solution.

    Parameters:
        x (float or array-like): Salt mass fraction [kg NaOH / kg solution].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float or np.ndarray: Crystallization temperature [°C].

    Validity:
        0 <= x <= 0.787 kg/kg

    Source: Wang et al. (2008), data extracted from the published plot.
    Author: Dorian Höffner
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0) or np.any(x_arr > 0.787):
        if prevent_errors:
            return float("nan") if x_arr.ndim == 0 else np.full(x_arr.shape, np.nan)
        raise ValueError(_msg(_MODULE, "solubility_temperature",
                              f"concentration x = {x} outside the valid "
                              f"range 0..0.787 kg/kg."))

    xs = np.array([pt[0] for pt in _CRYST_DATA]) / 100.0
    Ts = np.array([pt[1] for pt in _CRYST_DATA])
    t_solubility = np.interp(x_arr, xs, Ts)

    return float(t_solubility) if x_arr.ndim == 0 else t_solubility


def dhdx(x, T, prevent_errors=False):
    """
    Partial derivative of enthalpy with respect to NaOH mass fraction.

    Uses a central finite difference on enthalpy().

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: dh/dx [kJ/kg].
    """
    delta = 1e-6
    return (enthalpy(x + delta, T, prevent_errors=prevent_errors)
            - enthalpy(x - delta, T, prevent_errors=prevent_errors)) / (2 * delta)


def dhdT(x, T, prevent_errors=False):
    """
    Partial derivative of enthalpy with respect to temperature.

    Uses a central finite difference on enthalpy().

    Parameters:
        x (float): Salt mass fraction [kg NaOH / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: dh/dT [kJ/(kg K)].
    """
    delta = 1e-4
    return (enthalpy(x, T + delta, prevent_errors=prevent_errors)
            - enthalpy(x, T - delta, prevent_errors=prevent_errors)) / (2 * delta)


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def hxDiagram(editablePlot=False):
    """
    Plots the enthalpy-concentration diagram for NaOH-H2O solutions.

    Parameters:
        editablePlot (bool): If True, the figure is left open so it can be
            modified with matplotlib before showing; otherwise plt.show()
            is called.

    Returns:
        None

    Author: Dorian Höffner
    """
    with suppress_warnings():
        plt.figure(dpi=300)
        T_array = np.array(range(0, 101, 2))
        x_array = np.linspace(0, 0.75, 100)

        for T in T_array:
            h_array = np.array([enthalpy(x, T, prevent_errors=True)
                                for x in x_array])
            if T % 20 == 0:
                plt.plot(x_array, h_array, color='black', alpha=0.7, lw=0.8,
                         zorder=0)
                label_posx = x_array[23] + 0.005
                plt.text(label_posx, h_array[23] + 10, f'{T} °C', fontsize=8,
                         color='black')
            else:
                plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2,
                         zorder=0)

        plt.xlabel('Concentration $x=[kg_{NaOH}/kg_{solution}]$')
        plt.ylabel('Enthalpy $h=[kJ/kg]$')
        plt.xlim(0, 0.75)

        # crystallization curve
        cryst_data = crystallization_curve(return_data=True)
        cryst_x = np.array([x for x, T in cryst_data])
        cryst_h = np.array([enthalpy(x, T, prevent_errors=True)
                            if T > 0 else np.nan
                            for x, T in cryst_data])
        plt.plot(cryst_x, cryst_h, color='black', linestyle="--", lw=0.5,
                 label='Crystallization Curve')
        plt.fill_between(cryst_x, cryst_h, 0, color='white', zorder=0)

    if not editablePlot:
        plt.show()


def pTDiagram(log=True, invT=True, editablePlot=False, show_percentages=True):
    """
    Plots the pressure-temperature diagram for NaOH-H2O solutions.

    Parameters:
        log (bool): If True, the y-axis is logarithmic.
        invT (bool): If True, the x-axis is scaled as -1/T.
        editablePlot (bool): If True, the figure is left open so it can be
            modified with matplotlib before showing.
        show_percentages (bool): If True, the concentrations are labeled.

    Returns:
        None

    Author: Dorian Höffner
    """
    with suppress_warnings():
        # crystallization curve data
        cryst_data = crystallization_curve(return_data=True)
        cryst_T = np.array([T for x, T in cryst_data])
        cryst_p = np.array([saturation_pressure(x, T, prevent_errors=True)
                            for x, T in cryst_data])

        # temperature range
        temperaturesC = np.arange(0, 110, 1)
        temperaturesK = temperaturesC + 273.15
        concentrations = np.arange(0.1, 0.751, 0.01)

        plt.figure(dpi=300)
        plotTemperatures = np.arange(0, 101, 10) + 273.15

        waterPressure = [steamTable.psat_t(T - 273.15) * 1e5
                         for T in temperaturesK]  # bar -> Pa

        temp_plot = -1 / temperaturesK if invT else temperaturesK
        temp_plot_cryst = -1 / (cryst_T + 273.15) if invT else cryst_T + 273.15
        label_pos = temp_plot[-1] + 1e-5 if invT else temp_plot[-1] + 2

        for x in concentrations:
            p = [saturation_pressure(x, T, prevent_errors=True)
                 for T in temperaturesC]

            color = "black" if int(np.round(x * 100)) % 10 == 0 else "grey"
            lw = 1.0 if color == "black" else 0.25

            if log:
                plt.semilogy(temp_plot, p, color=color, lw=lw)
            else:
                plt.plot(temp_plot, p, color=color, lw=lw)

            if show_percentages and int(np.round(x * 100)) % 10 == 0:
                plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8,
                         color='black')

        plt.ylabel('Saturation Pressure [Pa]')
        plt.xlabel('Temperature [°C]')
        plt.xticks(-1 / plotTemperatures if invT else plotTemperatures,
                   [f"{t - 273.15:.0f}" for t in plotTemperatures])

        # concentration unit label
        if show_percentages:
            plt.text(label_pos, p[-1] * 0.85,
                     r'$\left[\frac{\mathrm{kg_{NaOH}}}{\mathrm{kg_{Solution}}}\right]$',
                     fontsize=11, color='black')

        # water line
        if log:
            plt.semilogy(temp_plot, waterPressure, color="grey",
                         linestyle='--', label='Pure Water')
        else:
            plt.plot(temp_plot, waterPressure, color="grey", linestyle='--',
                     label='Pure Water')

        # crystallization curve
        if log:
            plt.semilogy(temp_plot_cryst, cryst_p, color="gray",
                         linestyle='-', lw=1.0, label='Crystallization Curve',
                         zorder=101)
        else:
            plt.plot(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
                     lw=1.0, label='Crystallization Curve', zorder=101)
        plt.fill_between(temp_plot_cryst, np.nan_to_num(cryst_p, nan=1), 1,
                         where=(np.nan_to_num(cryst_p, nan=0) > 1),
                         color='white', zorder=100)

        # axis limits
        if invT:
            plt.xlim(-1 / temperaturesK[0], -1 / temperaturesK[-1])
        else:
            plt.xlim(temperaturesK[0], temperaturesK[-1])
        if log:
            plt.ylim(50, 2e5)
        else:
            plt.ylim(0, max(waterPressure) * 1.1)

        plt.legend()

    if not editablePlot:
        plt.show()


# crystallization data extracted from Wang et al. (2008): [x in %, T in °C]
_CRYST_DATA = [
    [0.2707275803722504, 0.0],
    [2.165820642978004, -1.3333333333333357],
    [4.060913705583756, -2.933333333333337],
    [5.685279187817259, -4.533333333333331],
    [7.580372250423012, -6.666666666666664],
    [9.34010152284264, -8.533333333333335],
    [10.96446700507614, -10.933333333333334],
    [12.588832487309643, -13.333333333333332],
    [14.077834179357025, -16.0],
    [15.5668358714044, -18.933333333333334],
    [16.785109983079526, -21.866666666666667],
    [17.868020304568528, -24.8],
    [18.68020304568528, -28.0],
    [20.169204737732656, -26.4],
    [21.658206429780037, -24.8],
    [22.74111675126904, -22.4],
    [23.688663282571916, -20.53333333333333],
    [24.771573604060915, -18.666666666666668],
    [25.583756345177665, -14.133333333333333],
    [26.395939086294412, -9.6],
    [27.34348561759729, -5.333333333333336],
    [28.56175972927242, -1.3333333333333357],
    [29.780033840947542, 1.8666666666666671],
    [31.810490693739425, 5.066666666666663],
    [32.89340101522843, 7.733333333333334],
    [34.247038917089675, 10.133333333333333],
    [35.736040609137056, 12.533333333333331],
    [37.36040609137056, 14.666666666666664],
    [38.984771573604064, 15.733333333333334],
    [40.33840947546531, 15.466666666666669],
    [41.96277495769881, 13.866666666666667],
    [43.45177664974619, 11.733333333333334],
    [44.67005076142132, 8.799999999999997],
    [45.6175972927242, 6.133333333333333],
    [47.10659898477157, 8.0],
    [48.73096446700507, 9.866666666666667],
    [50.35532994923858, 11.733333333333334],
    [51.43824027072758, 12.533333333333331],
    [52.25042301184433, 17.333333333333336],
    [53.06260575296108, 22.133333333333333],
    [53.87478849407783, 26.66666666666667],
    [54.95769881556684, 32.0],
    [55.90524534686971, 36.8],
    [56.98815566835872, 41.06666666666666],
    [58.34179357021996, 45.86666666666666],
    [59.96615905245347, 50.400000000000006],
    [61.86125211505922, 54.93333333333334],
    [63.756345177664976, 58.66666666666667],
    [65.65143824027072, 61.33333333333333],
    [68.08798646362098, 63.46666666666667],
    [70.25380710659898, 64.0],
    [72.14890016920474, 63.73333333333333],
    [73.36717428087987, 62.93333333333334],
    [73.63790186125212, 68.53333333333333],
    [74.04399323181049, 76.26666666666667],
    [74.45008460236886, 83.2],
    [74.72081218274111, 89.06666666666666],
    [75.26226734348562, 96.53333333333332],
    [75.80372250423012, 103.73333333333332],
    [76.34517766497461, 110.93333333333334],
    [77.02199661590524, 118.4],
    [77.834179357022, 126.13333333333333],
    [78.78172588832487, 133.33333333333334],
]


def crystallization_curve(return_data=False):
    """
    Plots (or returns) the crystallization curve of NaOH-H2O solutions.

    Data digitized from Wang et al. (2008): "Cellulose Fiber Dissolution in
    Sodium Hydroxide Solution at Low Temperature: Dissolution Kinetics and
    Solubility Improvement". The digitized data may contain minor
    inaccuracies.

    Parameters:
        return_data (bool): If True, returns the curve as a list of
            [x (kg/kg), T (°C)] pairs instead of plotting.

    Returns:
        None or list of [float, float]: The data if return_data is True.

    Author: Dorian Höffner
    """
    if return_data:
        return [[x / 100.0, T] for x, T in _CRYST_DATA]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot([pt[0] for pt in _CRYST_DATA],
             [pt[1] for pt in _CRYST_DATA],
             label='Crystallization Curve', color="black")
    plt.xlabel('NaOH Concentration [%]')
    plt.ylabel('Temperature [°C]')
    plt.xlim(0, 78.5)
    plt.ylim(-35, 140)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    plt.legend(loc='upper left')

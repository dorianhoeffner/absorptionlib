"""
Thermophysical property functions for aqueous LiBr-H2O solutions.

Units (uniform across absorptionlib):
    x : salt mass fraction [kg LiBr / kg solution]
    T : temperature [°C]
    p : pressure [Pa]

All property functions accept ``prevent_errors=False``:
    False -> out-of-range inputs emit OutOfRangeWarning /
             CrystallizationWarning; invalid inputs raise ValueError.
    True  -> all warnings are suppressed and ValueError is replaced by a
             ``float('nan')`` return value (safe for optimizers).

Main sources:
    Patek, Klomfar (2006): "A computationally effective formulation of the
        thermodynamic properties of LiBr-H2O solutions from 273 to 500 K
        over full composition range", Int. J. Refrigeration.
    Feuerecker (1994): "Entropieanalyse für Wärmepumpensysteme: Methoden
        und Stoffdaten", TU München.
    Boryta (1970): "Solubility of Lithium Bromide in Water between -50 °C
        and +100 °C", J. Chem. Eng. Data 15(1).
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

_MODULE = "LiBr"

_DESCRIPTIONS = {
    "saturation_temperature":   "Boiling point temperature of the solution at given x and p [°C].",
    "saturation_pressure":      "Equilibrium (vapor) pressure of the solution at given x and T [Pa].",
    "saturation_concentration": "Saturation concentration at given p and T [kg/kg].",
    "enthalpy":                 "Specific enthalpy of the solution at given x and T [kJ/kg].",
    "enthalpy_PK":              "Saturated liquid water enthalpy h'(T) after Patek/Klomfar [J/mol].",
    "density":                  "Density of the solution at given x and T [kg/m³].",
    "solubility_temperature":   "Crystallization temperature at given x [°C].",
    "hxDiagram":                "Plots the enthalpy-concentration diagram.",
    "pTDiagram":                "Plots the pressure-temperature diagram.",
    "crystallization_curve":    "Plots (or returns) the crystallization curve.",
}


def documentation():
    """Print an overview of all public functions of this module."""
    print_documentation(_MODULE, "LiBr", _DESCRIPTIONS)


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
# Parameters for LiBr-H2O correlations from Patek/Klomfar (2006)
# ---------------------------------------------------------------------------

class Params_PK:
    """Regression coefficients and constants from Patek/Klomfar (2006)."""

    def __init__(self):
        # Critical point of pure water
        self.TCritW = 647.096         # K
        self.pCritW = 22.064e6        # Pa
        self.densCritW = 322          # kg/m³
        self.densCritWmol = 17873.727  # mol/m³
        self.cpCritWmol = 76.0226     # J/(mol K)
        self.enthalpyCritWmol = 37548.5  # J/mol
        self.entropyCritWmol = 79.3933   # J/(mol K)

        # Triple point of pure water
        self.TTripW = 273.16          # K
        self.pTripW = 611.657         # Pa
        self.densTripW = 999.789      # kg/m³

        # Molar masses
        self.M_W = 0.018015268        # kg/mol
        self.M_LiBr = 0.08685         # kg/mol

        # Table 4: Pressure
        self.mTab4 = [3, 4, 4, 8, 1, 1, 4, 6]
        self.nTab4 = [0, 5, 6, 3, 0, 2, 6, 0]
        self.tTab4 = [0, 0, 0, 0, 1, 1, 1, 1]
        self.aTab4 = [-241.303, 19175000, -175521000, 32543000, 392.571,
                      -2126.26, 185127000, 1912.16]

        # Table 5: Density
        self.mTab5 = [1, 1]
        self.tTab5 = [0, 6]
        self.aTab5 = [1.746, 4.709]

        # Table 6: Specific heat capacity (cp)
        self.mTab6 = [2, 3, 3, 3, 3, 2, 1, 1]
        self.nTab6 = [0, 0, 1, 2, 3, 0, 3, 2]
        self.tTab6 = [0, 0, 0, 0, 0, 2, 3, 4]
        self.aTab6 = [-14.2094, 40.4943, 111.135, 229.98, 1345.26,
                      -0.014101, 0.0124977, -0.000683209]

        # Table 7: Enthalpy
        self.mTab7 = [1, 1, 2, 3, 6, 1, 3, 5, 4, 5, 5, 6, 6, 1, 2, 2, 2, 5,
                      6, 7, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1]
        self.nTab7 = [0, 1, 6, 6, 2, 0, 0, 4, 0, 4, 5, 5, 6, 0, 3, 5, 7, 0,
                      3, 1, 0, 4, 2, 6, 7, 0, 0, 1, 2, 3]
        self.tTab7 = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                      3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5]
        self.aTab7 = [
            2.27431, -7.99511, 385.239, -16394, -422.562, 0.113314,
            -8.33474, -17383.3, 6.49763, 3245.52, -13464.3, 39932.2,
            -258877, -0.00193046, 2.80616, -40.4479, 145.342, -2.74873,
            -449.743, -12.1794, -0.00583739, 0.23391, 0.341888, 8.85259,
            -17.8731, 0.0735179, -0.00017943, 0.00184261, -0.00624282,
            0.00684765,
        ]

        # Table 8: Entropy
        self.mTab8 = [1, 1, 2, 3, 6, 1, 3, 5, 1, 2, 2, 4, 5, 5, 6, 6, 1, 3,
                      5, 7, 1, 1, 1, 2, 3, 1, 1, 1, 1]
        self.nTab8 = [0, 1, 6, 6, 2, 0, 0, 4, 0, 0, 4, 0, 4, 5, 2, 5, 0, 4,
                      0, 1, 0, 2, 4, 7, 1, 0, 1, 2, 3]
        self.tTab8 = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                      3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5]
        self.aTab8 = [
            1.53091, -4.52564, 698.302, -21666.4, -1475.33, 0.0847012,
            -6.59523, -29533.1, 0.00956314, -0.188679, 9.31752, 5.78104,
            13893.1, -17176.2, 415.108, -55564.7, -0.00423409, 30.5242,
            -1.6762, 14.8283, 0.00303055, -0.040181, 0.149252, 2.5924,
            -0.177421, -0.000069965, 0.000605007, -0.00165228, 0.00122966,
        ]


_PARAMS_PK = Params_PK()  # instantiated once at import time


# ---------------------------------------------------------------------------
# Property functions
# ---------------------------------------------------------------------------

def saturation_temperature(x, p, prevent_errors=False):
    """
    Boiling point temperature of an aqueous LiBr-H2O solution.

    Solves saturation_pressure(x, T) = p for T using a bracketing root
    finder (brentq) on the interval 0 °C to 226.85 °C.

    Parameters:
        x (float): Salt mass fraction [kg LiBr / kg solution], 0.0 to 0.75.
        p (float): Pressure [Pa].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Boiling point temperature [°C] (NaN if no solution exists
        in the search interval and prevent_errors=True).

    Source: based on saturation_pressure (Patek/Klomfar 2006).
    """
    if x < 0.0 or x > 0.75:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_temperature",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0.00..0.75."))

    try:
        with suppress_warnings():
            T = brentq(lambda T: saturation_pressure(x, T, prevent_errors=True) - p,
                       0.0, 226.85)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_temperature",
                              f"no saturation temperature found in "
                              f"0..226.85 °C for x = {x}, p = {p} Pa."))

    # crystallization check (solubility correlation valid for x >= 0.5681)
    if x >= 0.5681:
        t_sol = solubility_temperature(x, prevent_errors=True)
        if T < t_sol:
            warn_crystallization(_MODULE, "saturation_temperature", T, t_sol,
                                 x, prevent_errors)

    return float(T)


def enthalpy_PK(T):
    """
    Molar enthalpy h'(T) of saturated liquid water after Patek/Klomfar.

    Parameters:
        T (float): Temperature [K].

    Returns:
        float: h'(T) [J/mol].

    Source: Patek/Klomfar (2006).
    """
    beta = [1 / 3, 2 / 3, 5 / 6, 21 / 6]
    alpha = [-4.37196e-1, 3.03440e-1, -1.29582, -1.76410e-1]
    Tc = 647.096    # K
    hc = 548.5      # J/mol

    summation = sum(alpha[i] * (1 - T / Tc) ** beta[i] for i in range(4))
    return hc * (1 + summation)  # J/mol


def enthalpy(x, T, prevent_errors=False):
    """
    Specific enthalpy of an aqueous LiBr-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg LiBr / kg solution].
            Validated for 0.4 <= x <= 0.75; for x < 0.4 the result is
            interpolated between pure water and the 40 % solution (not
            validated by experiments; an OutOfRangeWarning is emitted).
        T (float): Temperature [°C], 0 to 190 °C.
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Specific enthalpy [kJ/kg].

    Source: Feuerecker (1994): "Entropieanalyse für Wärmepumpensysteme:
        Methoden und Stoffdaten", TU München, p. 131 Eq. (9.13).
    History:
        - Original MATLAB implementation by Jan Albers, TU Berlin (2003).
        - Standardized API, ValueErrors, crystallization check:
          Dorian Höffner (2024/2026).
    """
    if T < 0 or T > 190:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "enthalpy",
                              f"temperature T = {T} °C outside the valid "
                              f"range 0..190 °C."))
    if x < 0 or x > 0.75 + 1e-9:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "enthalpy",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0..0.75."))
    if x < 0.4 - 1e-9:
        warn_out_of_range(
            _MODULE, "enthalpy",
            f"x = {x} is below 0.40 kg/kg. The result is interpolated "
            f"between pure water and the 40 % solution and is not "
            f"validated by experiments.", prevent_errors)

    # crystallization check (solubility correlation valid for x >= 0.5681)
    if x >= 0.5681:
        t_sol = solubility_temperature(x, prevent_errors=True)
        if T < t_sol:
            warn_crystallization(_MODULE, "enthalpy", T, t_sol, x,
                                 prevent_errors)

    temp = T + 273.15   # Kelvin, as used in Feuerecker's correlation
    konz = x * 100.0    # Feuerecker uses percent, not kg/kg

    def h_feuerecker(konz, temp):
        coefhli = [
            [-954.8, 47.7739, -1.59235, 2.09422e-2, -7.689e-5],
            [-0.3293, 4.076e-2, -1.36e-5, -7.1366e-6],
            [7.4285e-3, -1.5144e-4, 1.3555e-6],
            [-2.269e-6],
        ]
        a = sum(konz ** k * coefhli[0][k] for k in range(len(coefhli[0])))
        b = sum(konz ** k * coefhli[1][k] for k in range(len(coefhli[1])))
        c = sum(konz ** k * coefhli[2][k] for k in range(len(coefhli[2])))
        return a + temp * b + temp**2 * c + coefhli[3][0] * temp**3

    if konz >= 40:
        h_sol = h_feuerecker(konz, temp)
    else:
        # linear interpolation between pure water and the 40 % solution
        h40 = h_feuerecker(40.0, temp)
        h_water = steamTable.hL_t(T + 0.01)
        h_sol = h_water + konz / 40.0 * (h40 - h_water)

    return float(h_sol)


def saturation_pressure(x, T, prevent_errors=False):
    """
    Equilibrium (vapor) pressure of an aqueous LiBr-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg LiBr / kg solution], 0.0 to 0.75.
        T (float): Temperature [°C], 0 to 226.85 °C.
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Equilibrium pressure [Pa].

    Source: Patek, Klomfar (2006): "A computationally effective formulation
        of the thermodynamic properties of LiBr-H2O solutions from 273 to
        500 K over full composition range", Int. J. Refrigeration.
    History:
        - Programmed by Jan Albers (2010); validity limits added 2012.
        - Standardized API and ValueError: Dorian Höffner (2024/2026).
    """
    T_K = T + 273.15  # convert to Kelvin

    # validity checks
    if T_K < 273.15 or T_K > 500:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_pressure",
                              f"temperature T = {T} °C outside the valid "
                              f"range 0..226.85 °C (273.15..500 K)."))
    if x < 0.0 or x > 0.75:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_pressure",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0.00..0.75."))

    # crystallization check (solubility correlation valid for x >= 0.5681)
    if x >= 0.5681:
        t_sol = solubility_temperature(x, prevent_errors=True)
        if T < t_sol:
            warn_crystallization(_MODULE, "saturation_pressure", T, t_sol, x,
                                 prevent_errors)

    def psatW_local(temp):
        """Water vapor pressure [bar] (IAPWS-IF97 region 4)."""
        nreg4_1 = 1167.0521452767
        nreg4_2 = -724213.16703206
        nreg4_3 = -17.073846940092
        nreg4_4 = 12020.82470247
        nreg4_5 = -3232555.0322333
        nreg4_6 = 14.91510861353
        nreg4_7 = -4823.2657361591
        nreg4_8 = 405113.40542057
        nreg4_9 = -0.23855557567849
        nreg4_10 = 650.17534844798

        dl = temp + nreg4_9 / (temp - nreg4_10)
        Aco = dl**2 + nreg4_1 * dl + nreg4_2
        Bco = nreg4_3 * dl**2 + nreg4_4 * dl + nreg4_5
        cco = nreg4_6 * dl**2 + nreg4_7 * dl + nreg4_8
        return (2 * cco / (-Bco + (Bco**2 - 4 * Aco * cco) ** 0.5)) ** 4 * 10

    params = _PARAMS_PK
    x_mol = (x / params.M_LiBr) / (x / params.M_LiBr + (1 - x) / params.M_W)

    tsum = 0.0
    for k in range(len(params.aTab4)):
        tsum += (params.aTab4[k] * x_mol ** params.mTab4[k]
                 * (0.40 - x_mol) ** params.nTab4[k]
                 * (T_K / params.TCritW) ** params.tTab4[k])

    psat = psatW_local(T_K - tsum)  # bar

    return float(psat * 1e5)  # bar -> Pa


def saturation_concentration(p, T, prevent_errors=False):
    """
    Saturation concentration of LiBr in water at given pressure and temperature.

    Solves saturation_pressure(x, T) = p for x using a bracketing root
    finder (brentq) on the interval 0.0 to 0.75 kg/kg.

    Parameters:
        p (float): Pressure [Pa].
        T (float): Temperature [°C], 0 to 226.85 °C.
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Saturation concentration [kg LiBr / kg solution] (NaN if no
        solution exists in the search interval and prevent_errors=True).

    Author: Dorian Höffner
    """
    if T < 0 or T > 226.85:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_concentration",
                              f"temperature T = {T} °C outside the valid "
                              f"range 0..226.85 °C."))

    try:
        with suppress_warnings():
            x = brentq(lambda x: saturation_pressure(x, T, prevent_errors=True) - p,
                       1e-6, 0.75)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_concentration",
                              f"no saturation concentration found in "
                              f"0..0.75 kg/kg for p = {p} Pa, T = {T} °C. "
                              f"The requested pressure may be outside the "
                              f"correlation's range."))

    return float(x)


def density(x, T, prevent_errors=False):
    """
    Density of an aqueous LiBr-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg LiBr / kg solution], 0.0 to 0.75.
            Validated for 0.4 <= x <= 0.75; for x < 0.4 the results are
            extrapolated (an OutOfRangeWarning is emitted).
        T (float): Temperature [°C], 0 to 190 °C.
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Density [kg/m³].

    Notes:
        - The density of pure water is calculated using coefficients from
          Landolt-Börnstein (as used by Peter Müller).
        - Includes the correction from private communication between
          F. Ziegler and G. Feuerecker (1995): the power of 2 was missing
          at the last konz term of the published equation.

    References:
        - Feuerecker (1994): "Entropieanalyse für Wärmepumpensysteme:
          Methoden und Stoffdaten", TU München.
        - Landolt-Börnstein: II. Band, 1. Teil, 6. Auflage, Berlin 1971.
    History:
        - J.A. Nov. 2003 programmed; Feb. 2004, Apr. 2012 corrections.
        - Standardized API, ValueError, extrapolation warning:
          Dorian Höffner (2024/2026).
    """
    if T < 0 or T > 190:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "density",
                              f"temperature T = {T} °C outside the valid "
                              f"range 0..190 °C."))
    if x < 0.0 or x > 0.75 + 1e-9:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "density",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0.00..0.75."))
    if x < 0.4 - 1e-9:
        warn_out_of_range(_MODULE, "density",
                          f"x = {x} is below 0.40 kg/kg; the result is "
                          f"extrapolated.", prevent_errors)

    # pure water density, Landolt-Börnstein coefficients
    a_1 = 3.9863      # K
    a_2 = 508929.2    # K²
    a_3 = 288.9414    # K
    a_4 = 68.12963    # K
    a_5 = 0.999973    # g/cm³

    dens_H2O = (1 - (T - a_1) ** 2 / a_2 * (T + a_3) / (T + a_4)) * a_5

    # Feuerecker correlation incl. Ziegler/Feuerecker (1995) correction
    rho = (dens_H2O / 2.0
           * (np.exp(1.2 * x) + np.exp((0.842 + 1.6414e-3 * T) * x**2))
           ) * 1000.0

    return float(rho)


def solubility_temperature(x, prevent_errors=False):
    """
    Crystallization (solubility) temperature of an aqueous LiBr-H2O solution.

    Parameters:
        x (float or array-like): Salt mass fraction [kg LiBr / kg solution].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float or np.ndarray: Crystallization temperature [°C].

    Validity:
        0.5681 <= x <= 0.75 kg/kg

    Sources:
        Boryta (1970): "Solubility of Lithium Bromide in Water between
            -50 °C and +100 °C (45 to 70 % Lithium Bromide)",
            J. Chem. Eng. Data 15(1), pp. 142-144.
        Feuerecker (1994): "Entropieanalyse für Wärmepumpensysteme:
            Methoden und Stoffdaten", TU München.
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0.5681) or np.any(x_arr > 0.75):
        if prevent_errors:
            return float("nan") if x_arr.ndim == 0 else np.full(x_arr.shape, np.nan)
        raise ValueError(_msg(_MODULE, "solubility_temperature",
                              f"concentration x = {x} outside the valid "
                              f"range 0.5681..0.75 kg/kg."))

    # Parameters
    mu_1 = 0.660036363636364
    mu_2 = 0.0521377438043144

    x_hat = (x_arr - mu_1) / mu_2

    part1 = (-1.25354043437046 * x_hat**7 + 1.60535142980859 * x_hat**6
             + 9.50132460833796 * x_hat**5 - 10.9718095175445 * x_hat**4)
    part2 = (-23.0924483393181 * x_hat**3 + 23.9376211870673 * x_hat**2
             + 57.4166682907763 * x_hat + 55.0110013350386)

    t_solubility = part1 + part2  # already in °C

    return float(t_solubility) if x_arr.ndim == 0 else t_solubility


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def hxDiagram(editablePlot=False):
    """
    Plots the enthalpy-concentration diagram for LiBr-H2O solutions.

    Parameters:
        editablePlot (bool): If True, the figure is left open so it can be
            modified with matplotlib before showing.

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
                label_posx = x_array[55] + 0.005
                plt.text(label_posx, h_array[55] + 2, f'{T} °C', fontsize=8,
                         color='black')
            else:
                plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2,
                         zorder=0)

        plt.xlabel('Concentration $x=[kg_{LiBr}/kg_{solution}]$')
        plt.ylabel('Enthalpy $h=[kJ/kg]$')
        plt.xlim(0, 0.75)

        # mark the interpolated region below 40 % concentration
        plt.axvline(x=0.4, color='grey', linestyle='--', lw=0.5)
        plt.text(0.28, 330, 'Interpolated', fontsize=8, color='grey')
        plt.annotate('', xy=(0.4, 320), xytext=(0.25, 320),
                     arrowprops=dict(arrowstyle='<-', color='grey'))

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
    Plots the pressure-temperature diagram for LiBr-H2O solutions.

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
        concentrations = np.arange(0.1, 0.75, 0.01)

        plt.figure(dpi=300)
        plotTemperatures = np.arange(0, 101, 10) + 273.15

        waterPressure = [steamTable.psat_t(T - 273.15) * 1e5 if T > 273.15
                         else np.nan for T in temperaturesK]  # bar -> Pa

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

            if (show_percentages and int(np.round(x * 100)) % 10 == 0
                    and x > 0.29):
                plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8,
                         color='black')

        plt.ylabel('Saturation Pressure [Pa]')
        plt.xlabel('Temperature [°C]')
        plt.xticks(-1 / plotTemperatures if invT else plotTemperatures,
                   [f"{t - 273.15:.0f}" for t in plotTemperatures])

        # concentration unit label
        if show_percentages:
            plt.text(label_pos, p[-1] * 0.85,
                     r'$\left[\frac{\mathrm{kg_{LiBr}}}{\mathrm{kg_{Solution}}}\right]$',
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
            plt.ylim(100, 1.4e5)
        else:
            plt.ylim(0, np.nanmax(waterPressure) * 1.1)

        plt.legend()

    if not editablePlot:
        plt.show()


def crystallization_curve(return_data=False):
    """
    Plots (or returns) the crystallization curve of LiBr-H2O solutions.

    The curve is generated from solubility_temperature() (Boryta 1970 /
    Feuerecker 1994) on the valid concentration range 0.5681..0.75 kg/kg.

    Parameters:
        return_data (bool): If True, returns the curve as a list of
            [x (kg/kg), T (°C)] pairs instead of plotting.

    Returns:
        None or list of [float, float]: The data if return_data is True.

    Author: Dorian Höffner
    """
    xs_array = np.linspace(0.5681, 0.75, 100)
    ts_array = solubility_temperature(xs_array, prevent_errors=True)

    if return_data:
        return [[float(xs), float(ts)] for xs, ts in zip(xs_array, ts_array)]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(xs_array * 100, ts_array, label='Crystallization Curve',
             color="black")
    plt.xlabel(r'$\mathrm{LiBr}$ Concentration [%]')
    plt.ylabel('Temperature [°C]')
    plt.xlim(0, 75)
    plt.ylim(-60, 200)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    plt.legend(loc='upper left')

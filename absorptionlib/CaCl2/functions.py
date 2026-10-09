"""
Thermophysical property functions for aqueous CaCl2-H2O solutions.

Units (uniform across absorptionlib):
    x : salt mass fraction [kg CaCl2 / kg solution]
    T : temperature [°C]
    p : pressure [Pa]

All property functions accept ``prevent_errors=False``:
    False -> out-of-range inputs emit OutOfRangeWarning /
             CrystallizationWarning; invalid inputs raise ValueError.
    True  -> all warnings are suppressed and ValueError is replaced by a
             ``float('nan')`` return value (safe for optimizers).

Main source:
    Conde (2009): "Aqueous solutions of lithium and calcium chlorides:
        property formulations for use in air conditioning equipment design".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

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

_MODULE = "CaCl2"

_DESCRIPTIONS = {
    "saturation_temperature":    "Boiling point temperature of the solution at given x and p [°C].",
    "saturation_pressure":       "Equilibrium (vapor) pressure of the solution at given x and T [Pa].",
    "saturation_concentration":  "Saturation concentration at given p and T [kg/kg].",
    "enthalpy":                  "Specific enthalpy of the solution at given x and T [kJ/kg].",
    "differential_enthalpy_AD":  "Differential enthalpy of dilution at given x and T [kJ/kg H2O].",
    "density":                   "Density of the solution at given x and T [kg/m³].",
    "specific_heat_capacity":    "Specific heat capacity of the solution at given x and T [kJ/(kg K)].",
    "dynamic_viscosity":         "Dynamic viscosity of the solution at given x and T [Pa s].",
    "diffusion_coefficient":     "Self diffusion coefficient of the solution at given x and T [m²/s].",
    "solubility_temperature":    "Crystallization temperature at given x [°C].",
    "hxDiagram":                 "Plots the enthalpy-concentration diagram.",
    "pTDiagram":                 "Plots the pressure-temperature diagram.",
    "crystallization_curve":     "Plots (or returns) the crystallization curve.",
}


def documentation():
    """Print an overview of all public functions of this module."""
    print_documentation(_MODULE, "CaCl2", _DESCRIPTIONS)


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
    Boiling point temperature of an aqueous CaCl2-H2O solution.

    Solves saturation_pressure(x, T) = p for T using a bracketing root
    finder (brentq) on the interval -40 °C to 220 °C.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 0.6.
        p (float): Pressure [Pa].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Boiling point temperature [°C] (NaN if no solution exists
        in the search interval and prevent_errors=True).

    Source: based on saturation_pressure (Conde 2009).
    """
    try:
        with suppress_warnings():
            T = brentq(lambda T: saturation_pressure(x, T, prevent_errors=True) - p,
                       -40.0, 220.0)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_temperature",
                              f"no saturation temperature found in "
                              f"-40..220 °C for x = {x}, p = {p} Pa."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "saturation_temperature", T, t_sol, x,
                             prevent_errors)

    return float(T)


def enthalpy(x, T, prevent_errors=False):
    """
    Specific enthalpy of an aqueous CaCl2-H2O solution.

    The enthalpy is calculated as the sum of an ideal part (specific heat
    capacities of water and of the 25 % solution) and an excess part
    obtained by numerical integration of the differential enthalpy of
    dilution (differential_enthalpy_AD).

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 0.6.
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Specific enthalpy [kJ/kg].

    Source: excess enthalpy based on Conde (2009).
    Author: Dorian Höffner
    """
    if not 0 <= x <= 0.6:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "enthalpy",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0..0.6."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "enthalpy", T, t_sol, x, prevent_errors)

    def excess_enthalpy(x, T):
        def integrand(x_):
            return differential_enthalpy_AD(x_, T, prevent_errors=True) / x_**2
        x_start = 1e-5
        integral, _ = quad(integrand, x_start, x)
        constant, _ = quad(integrand, x_start, 0.25)
        return x * integral - constant * x

    def ideal_enthalpy(x, T):
        cp_water = steamTable.CpL_t(T + 1e-6)
        cp_CaCl2_25 = specific_heat_capacity(x=0.25, T=T, prevent_errors=True)
        return (1 - x) * cp_water * T + x * cp_CaCl2_25 * T

    return float(ideal_enthalpy(x, T) + excess_enthalpy(x, T))


def differential_enthalpy_AD(x, T, prevent_errors=False):
    """
    Differential enthalpy of dilution of an aqueous CaCl2-H2O solution.

    How much additional energy (compared to the pure vaporization enthalpy)
    is needed to vaporize a certain amount of water from the solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Differential enthalpy [kJ/kg H2O].

    Source: Conde (2009).
    Author: Dorian Höffner
    """
    H1 = 0.855
    H2 = -1.965
    H3 = -2.265
    H4 = 0.8
    H5 = -955.690
    H6 = 3011.974

    if not 0 <= x < H4:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "differential_enthalpy_AD",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0..{H4}."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "differential_enthalpy_AD", T, t_sol,
                             x, prevent_errors)

    Theta = (T + 273.15) / 647.1  # reduced temperature (critical T of water)
    zeta = x / (H4 - x)
    dh_dil0 = H5 + H6 * Theta
    dh_dil = dh_dil0 * (1 + (zeta / H1) ** H2) ** H3

    return float(dh_dil)


def saturation_pressure(x, T, prevent_errors=False):
    """
    Equilibrium (vapor) pressure of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 0.6.
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Vapor pressure over the solution [Pa].

    Source: Conde (2009).
    Authors: O. Buchin (2011); standardized: Dorian Höffner (2024/2026).
    """
    if not 0 <= x <= 0.6:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_pressure",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0..0.6."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "saturation_pressure", T, t_sol, x,
                             prevent_errors)

    zeta = x  # mass fraction of salt in solution

    # Coefficients (Conde 2009, CaCl2)
    pi0, pi1, pi2 = 0.31, 3.698, 0.6
    pi3, pi4, pi5 = 0.231, 4.584, 0.49
    pi6, pi7, pi8, pi9 = 0.478, -5.2, -0.4, 0.018

    A = 2 - (1 + (zeta / pi0) ** pi1) ** pi2
    B = (1 + (zeta / pi3) ** pi4) ** pi5 - 1
    pi25 = (1 - (1 + (zeta / pi6) ** pi7) ** pi8
            - pi9 * np.exp(-((zeta - 0.1) ** 2) / 0.005))

    # water properties
    TcH2O = 647.26  # K
    pcH2O = 22.064  # MPa

    Theta = (T + 273.15) / TcH2O
    fsol = A + B * Theta
    pi_rel = pi25 * fsol

    # vapor pressure of pure water
    tau = 1 - Theta
    A0, A1, A2 = -7.858230, 1.839910, -11.781100
    A3, A4, A5 = 22.670500, -15.939300, 1.775160

    lnpi = (A0 * tau + A1 * tau**1.5 + A2 * tau**3 + A3 * tau**3.5
            + A4 * tau**4 + A5 * tau**7.5) / (1 - tau)
    pdH2O = np.exp(lnpi) * pcH2O * 1e6  # MPa -> Pa

    return float(pi_rel * pdH2O)


def saturation_concentration(p, T, prevent_errors=False):
    """
    Saturation concentration of CaCl2 in water at given pressure and temperature.

    Solves saturation_pressure(x, T) = p for x using a bracketing root
    finder (brentq) on the interval 0.0 to 0.6 kg/kg.

    Parameters:
        p (float): Pressure [Pa].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Saturation concentration [kg CaCl2 / kg solution] (NaN if no
        solution exists in the search interval and prevent_errors=True).

    Author: Dorian Höffner
    """
    try:
        with suppress_warnings():
            x = brentq(lambda x: saturation_pressure(x, T, prevent_errors=True) - p,
                       1e-6, 0.6)
    except ValueError:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "saturation_concentration",
                              f"no saturation concentration found in "
                              f"0..0.6 kg/kg for p = {p} Pa, T = {T} °C."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "saturation_concentration", T, t_sol,
                             x, prevent_errors)

    return float(x)


def density(x, T, prevent_errors=False):
    """
    Density of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 0.6.
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Density [kg/m³].

    Source: Conde (2009).
    Author: Dorian Höffner
    """
    if not 0 <= x <= 0.6:
        if prevent_errors:
            return float("nan")
        raise ValueError(_msg(_MODULE, "density",
                              f"mass fraction x = {x} outside the valid "
                              f"range 0..0.6."))

    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "density", T, t_sol, x, prevent_errors)

    def rho_h2o(T):
        B = [1.9937718430, 1.0985211604, -0.5094492996, -1.7619124270,
             -44.9005480267, -723692.2618632]
        rho_h2o_c = 322.0  # critical density of water [kg/m³]
        Tc = 647.1         # K
        t = 1 - (T + 273.15) / Tc
        return rho_h2o_c * (1 + B[0] * t**(1 / 3) + B[1] * t**(2 / 3)
                            + B[2] * t**(5 / 3) + B[3] * t**(16 / 3)
                            + B[4] * t**(43 / 3) + B[5] * t**(110 / 3))

    c = [1.0, 0.836014, -0.436300, 0.105642]
    rho = rho_h2o(T) * sum(c[i] * (x / (1 - x)) ** i for i in range(4))

    return float(rho)


def specific_heat_capacity(x, T, prevent_errors=False):
    """
    Specific heat capacity of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution].
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Specific heat capacity [kJ/(kg K)].

    Source: Conde (2009).
    Author: Dorian Höffner
    """
    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "specific_heat_capacity", T, t_sol, x,
                             prevent_errors)

    # water cp coefficients (below/above 0 °C)
    if T <= 0:
        a, b, c = 830.54602, -1247.52013, -68.60350
        d, e, f = 491.27650, -1.80692, -137.51511
    else:
        a, b, c = 88.7891, -120.1958, -16.9264
        d, e, f = 52.4654, 0.10826, 0.46988

    A, B, C = 1.63799, -1.69002, 1.05124
    F, G, H = 58.5225, -105.6343, 47.7948

    Theta = (T + 273.15) / 228 - 1

    cpH2O = (a + b * Theta**0.02 + c * Theta**0.04 + d * Theta**0.06
             + e * Theta**1.8 + f * Theta**8)

    f1 = A * x + B * x**2 + C * x**3
    f2 = F * Theta**0.02 + G * Theta**0.04 + H * Theta**0.06

    cp = cpH2O * (1 - f1 * f2)

    return float(cp)


def dynamic_viscosity(x, T, prevent_errors=False):
    """
    Dynamic viscosity of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 1.
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Dynamic viscosity [Pa s].

    Notes:
        The water viscosity is obtained from pyXSteam (IAPWS-IF97) for
        T > 0 °C and from the Conde (2009) sub-cooled water correlation
        for T <= 0 °C.

    Source: Conde (2009).
    Authors: O. Buchin (2010); standardized: Dorian Höffner (2024/2026).
    """
    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "dynamic_viscosity", T, t_sol, x,
                             prevent_errors)

    p_bar = 1.01325  # bar

    # Coefficients (Conde 2009, CaCl2)
    eta1 = -0.169310
    eta2 = 0.817350
    eta3 = 0.574230
    eta4 = 0.398750

    zeta = x / ((1 - x) ** (1 / 0.6))

    # water viscosity [Pa s]
    if T <= 0:
        # Conde (2009) sub-cooled water correlation, anchored at 0 °C
        A = 1.0261862
        B = 12481.702
        C = -19510.923
        D = 7065.286
        E = -395.561
        F = 143922.996
        Theta = (T + 273.15) / 228 - 1
        etaH2O_0 = steamTable.my_pt(p_bar, 1e-7)
        eta_H2O = etaH2O_0 * (A + B * Theta**0.02 + C * Theta**0.04
                              + D * Theta**0.08 + E * Theta**2.85
                              + F * Theta**8)
    else:
        eta_H2O = steamTable.my_pt(p_bar, T)

    # solution viscosity
    TcH2O = 647.26  # K
    Theta_c = (T + 273.15) / TcH2O
    eta = eta_H2O * np.exp(eta1 * zeta**3.6 + eta2 * zeta
                           + eta3 * zeta / Theta_c + eta4 * zeta**2)

    return float(eta)  # [Pa s]


def diffusion_coefficient(x, T, prevent_errors=False):
    """
    Self diffusion coefficient of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float): Salt mass fraction [kg CaCl2 / kg solution], 0 to 1.
        T (float): Temperature [°C].
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float: Self diffusion coefficient [m²/s].

    Sources:
        Conde (2009); water self-diffusion after Holz, Heil, Sacco:
        "Temperature-dependent self-diffusion coefficients of water and six
        selected molecular liquids for calibration in accurate 1H NMR PFG
        measurements".
    Authors: O. Buchin (2011, MATLAB); standardized: Dorian Höffner
        (2024/2026).
    """
    # crystallization check
    t_sol = solubility_temperature(x, prevent_errors=True)
    if T < t_sol:
        warn_crystallization(_MODULE, "diffusion_coefficient", T, t_sol, x,
                             prevent_errors)

    # self diffusion coefficient of water
    D0 = 1.635e-8  # m²/s
    TS = 215.05    # K
    gamma = 2.063
    Dw = D0 * (((T + 273.15) / TS) - 1) ** gamma

    # solution coefficients (Conde 2009, CaCl2)
    d1 = 0.55
    d2 = -5.52
    d3 = -0.56

    D = Dw * (1 - (1 + (np.sqrt(x) / d1) ** d2) ** d3)

    return float(D)


def solubility_temperature(x, prevent_errors=False):
    """
    Crystallization (solubility) temperature of an aqueous CaCl2-H2O solution.

    Parameters:
        x (float or array-like): Salt mass fraction [kg CaCl2 / kg solution],
            0 to 0.78.
        prevent_errors (bool): If True, suppress warnings and return NaN
            instead of raising errors.

    Returns:
        float or np.ndarray: Crystallization temperature [°C].

    Source: Conde (2009).
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    scalar = np.isscalar(x) or np.ndim(x) == 0

    if np.any(x_arr < 0) or np.any(x_arr > 0.78):
        if prevent_errors:
            return float("nan") if scalar else np.full(x_arr.shape, np.nan)
        raise ValueError(_msg(_MODULE, "solubility_temperature",
                              f"concentration x = {x} outside the valid "
                              f"range 0..0.78 kg/kg."))

    # Coefficients (Conde 2009, CaCl2)
    A0 = np.array([0.422088, -0.378950, -0.519970, -1.149044, -2.385836,
                   -2.807560])
    A1 = np.array([-0.066933, 3.456900, 3.400970, 5.509111, 8.084829,
                   4.678250])
    A2 = np.array([-0.282395, -3.531310, -2.851290, -4.642544, -5.303476,
                   0.000000])
    A3_ice = -355.514247

    TcH2O = 647.26  # K

    t = np.zeros((x_arr.size, 6))

    # ice line (k = 0)
    theta = A0[0] + A1[0] * x_arr + A2[0] * x_arr**2.0 + A3_ice * x_arr**7.5
    t[:, 0] = theta * TcH2O - 273.15

    # hydrates (k = 1..5)
    for k in range(1, 6):
        theta = A0[k] + A1[k] * x_arr + A2[k] * x_arr**2.0
        t[:, k] = theta * TcH2O - 273.15

    ts = np.max(t, axis=1)

    return float(ts[0]) if scalar else ts


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def hxDiagram(editablePlot=False):
    """
    Plots the enthalpy-concentration diagram for CaCl2-H2O solutions.

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
        x_array = np.linspace(0.001, 0.6, 50)

        for T in T_array:
            h_array = np.array([enthalpy(x, T, prevent_errors=True)
                                for x in x_array])
            if T % 20 == 0:
                plt.plot(x_array, h_array, color='black', alpha=0.7, lw=0.8,
                         zorder=0)
                label_posx = x_array[15] + 0.01
                plt.text(label_posx, h_array[15] + 10, f'{T} °C', fontsize=8,
                         color='black')
            else:
                plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2,
                         zorder=0)

        plt.xlabel('Concentration $x=[kg_{CaCl_{2}}/kg_{solution}]$')
        plt.ylabel('Enthalpy $h=[kJ/kg]$')
        plt.xlim(0, 0.6)
        plt.ylim(-50, 800)

        # crystallization curve
        cryst_data = crystallization_curve(return_data=True)
        cryst_x = np.array([x for x, T in cryst_data])
        cryst_h = np.array([enthalpy(x, T, prevent_errors=True)
                            if T > 0 else np.nan
                            for x, T in cryst_data])
        plt.plot(cryst_x, cryst_h, color='black', linestyle="--", lw=0.5,
                 label='Crystallization Curve', zorder=101)
        plt.fill_between(cryst_x, np.nan_to_num(cryst_h, nan=-50), -50,
                         where=(np.nan_to_num(cryst_h, nan=-50) > -50),
                         color='white', zorder=100)

    if not editablePlot:
        plt.show()


def pTDiagram(log=True, invT=True, editablePlot=False, show_percentages=True):
    """
    Plots the pressure-temperature diagram for CaCl2-H2O solutions.

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
        concentrations = np.arange(0.1, 0.6, 0.01)

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
                     r'$\left[\frac{\mathrm{kg_{CaCl_2}}}{\mathrm{kg_{Solution}}}\right]$',
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
            plt.ylim(220, 1.1e5)
        else:
            plt.ylim(0, max(waterPressure) * 1.1)

        plt.legend()

    if not editablePlot:
        plt.show()


def crystallization_curve(return_data=False):
    """
    Plots (or returns) the crystallization curve of CaCl2-H2O solutions.

    The curve is generated from solubility_temperature() (Conde 2009) on
    the concentration range 0.01..0.78 kg/kg.

    Parameters:
        return_data (bool): If True, returns the curve as a list of
            [x (kg/kg), T (°C)] pairs instead of plotting.

    Returns:
        None or list of [float, float]: The data if return_data is True.

    Author: Dorian Höffner
    """
    xs_array = np.linspace(0.01, 0.78, 100)
    ts_array = solubility_temperature(xs_array, prevent_errors=True)

    if return_data:
        return [[float(xs), float(ts)] for xs, ts in zip(xs_array, ts_array)]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(xs_array * 100, ts_array, label='Crystallization Curve',
             color="black")
    plt.xlabel(r'$\mathrm{CaCl_2}$ Concentration [%]')
    plt.ylabel('Temperature [°C]')
    plt.xlim(0, 78)
    plt.ylim(-60, 200)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    plt.legend(loc='upper left')

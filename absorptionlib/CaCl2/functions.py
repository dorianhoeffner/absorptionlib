import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt

from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

import sys
import os


def documentation():
    print("""
This module contains functions for calculating properties of CaCl2-H2O solutions:

| Function Name              | Description                                                                                   |
|---------------------------|------------------------------------------------------------------------------------------------|
| saturation_temperature    | Calculate the boiling point temperature of an aqueous Lithium Bromide solution.                |
| enthalpy                  | Calculate the enthalpy of an H2O-CaCl2 solution at a given temperature and concentration.       |
| differential_enthalpy_AD  | Calculates the differential enthalpy of a CaCl2 solution.                                      |
| saturation_pressure       | Calculate the equilibrium pressure of an H2O-CaCl2 solution.                                    |
| saturation_concentration  | Calculates the saturation concentration of CaCl2 in water based on the temperature and pressure.|
| density                   | Calculate the density of a water-CaCl2 solution.                                                |
| specific_heat_capacity    | Calculate the specific heat capacity of a CaCl2 solution.                                      |
| dynamic_viscosity         | Calculate the dynamic viscosity of a CaCl2 solution.                                            |
| diffusion_coefficient     | Computes the self diffusion coefficient of a CaCl2 solution.                                   |
| hxDiagram                 | Plots the pressure-temperature diagram for CaCl2-H2O solutions.                                 |
| pTDiagram                 | Plots the pressure-temperature diagram for CaCl2-H2O solutions.                                 |
| solubility_temperature    | Calculate the crystallization temperature of CaCl2 solution in water.                           |

For more information use the following function: CaCl2.explain("function_name")

For example: CaCl2.explain("enthalpy")
    """)

def explain(function_name):
    """
    Prints the documentation for a specific function in the module.
    
    Parameters:
        function_name (str): The name of the function to explain.
        
    Returns:
        None
    """
    
    # Get the function object from the module
    func = globals().get(function_name)
    
    if func is None:
        print(f"Function '{function_name}' not found.")
        return
    
    # Print the docstring of the function
    print(func.__doc__)


# CaCl2 property functions
# DONE
def saturation_temperature(x, p):
    """
    Calculates the saturation temperature for a given concentration and pressure of CaCl2 solution.
    Parameters:
        x (float): Mass fraction [kg CaCl2 / kg solution] or concentration of CaCl2 in the solution.
        p (float): Pressure [Pa] at which to calculate the saturation temperature (same units as used in saturation_pressure).
    Returns:
        float: Saturation temperature (in the same units as used in saturation_pressure and solubility_temperature).
    Warnings:
        Prints a warning if the calculated saturation temperature is below the solubility temperature for the given concentration.
    Notes:
        - Uses a numerical solver to find the temperature at which the solution's saturation pressure equals the specified pressure.
        - Assumes the existence of `saturation_pressure(x, T)` and `solubility_temperature(x)` functions.
    """
    T_guess = 20
    T = fsolve(lambda T: saturation_pressure(x, T) - p, T_guess)[0]

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)[0]
    if T < t_sol:
        print(f"Warning from CaCl2.saturation_temperature: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")
    
    return T

# DONE
def enthalpy(x, T, prevent_errors=False):
    """
    Calculate the specific enthalpy of a CaCl2 solution at a given concentration and temperature.
    Parameters:
        x (float): Mass fraction of CaCl2 in the solution, defined as x = m_LiCl / (m_H2O + m_LiCl).
        T (float): Temperature in degrees Celsius [°C].
        prevent_errors (bool, optional): If True, suppresses warnings and returns NaN when the temperature is below the solubility temperature. Default is False.
    Returns:
        float: Specific enthalpy of the solution in [kJ/kg]. Returns np.nan if the temperature is below the solubility temperature and prevent_errors is False.
    Notes:
        - The enthalpy is calculated as the sum of the ideal enthalpy and the excess enthalpy.
        - The ideal enthalpy is based on the specific heat capacities of water and CaCl2 at 25% mass fraction.
        - The excess enthalpy is computed via numerical integration of the differential enthalpy of solution.
        - Issues a warning and returns np.nan if the solution is not in the liquid state (i.e., T < solubility temperature) unless prevent_errors is True.
    References:
        - Differential enthalpy and solubility temperature functions must be defined elsewhere in the codebase.
        - Uses steamTable.CpL_t and specific_heat_capacity for heat capacity calculations.
    
    Date:   2025-03-20
    Author: Dorian Höffner

    Parameter:
     T in [°C]
     x = m_LiCl/(m_H2O + m_LiCl)
     h in [kJ/kg]
    """
    
    def excess_enthalpy(x, T):
        def dh_sol(x, T):
            return differential_enthalpy_AD(x, T)
        def integrand(x):
            return dh_sol(x,T) / x**2
        x_start = 0.00001
        integral, error = quad(integrand, x_start, x)
        CONSTANT, _ = quad(integrand, x_start, 0.25)
        excess_enthalpy = x * integral - CONSTANT * x
        return excess_enthalpy    

    def ideal_enthalpy(x,T):
        cp_water    = steamTable.CpL_t(T+0.000001)
        cp_LiCl_25  = specific_heat_capacity(x=0.25, T=T)
        return (1-x) * cp_water * (T)  + x * cp_LiCl_25 * (T)
    
    # check if the solution is in liquid state
    if not prevent_errors:
        t_sol = solubility_temperature(x)
        if T < t_sol:
            print(f"Warning from CaCl2.enthalpy: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")
            return np.nan

    return ideal_enthalpy(x, T) + excess_enthalpy(x, T)

def differential_enthalpy_AD(x, T):
    """
    Calculates the differential enthalpy [kJ/kgH2O] of a CaCl2 solution based on concentration and temperature.
    How much additional energy (in comparison to pure vaporization enthalpy) is needed to vaporize a certain amount of water in a CaCl2 solution.
    
    Parameters:
        x (float): Concentration of CaCl2 in the solution [kg/kg].
        T (float): Temperature of the solution [°C].

    Returns:
        float: Differential enthalpy [kJ/kgH2O].

    Author: Dorian Höffner 11/2024
    Source: CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    H1 = 0.855
    H2 = -1.965
    H3 = -2.265
    H4 = 0.8
    H5 = -955.690
    H6 = 3011.974

    # Reduced Temperature (with critical T of water)
    Theta = (T + 273.15) / 647.1 # FLAG

    # zeta
    zeta = x / (H4-x)

    # Reference differental enthalpy
    dh_dil0 =  H5 + H6 * Theta

    # differental enthalpy
    dh_dil = dh_dil0 * (1 + (zeta/H1)**H2)**H3

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)[0]
    if T < t_sol:
        print(f"Warning from CaCl2.differential_enthalpy_AD: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return dh_dil

# DONE
def saturation_pressure(x, T):
    """
    Calculates the pressure of aqueous NaOH-H2O solutions at high concentrations.
    
    Last Change: Dorian Höffner 10/2024
    Author: O. Buchin, 03/2011
    Source: Correlation according to CONDE2009 "Aqueous solutions of lithium and calcium chlorides:
    property formulations for use in air conditioning equipment design" (2009)
    
    Parameters:
    T (array-like): Temperature in [°C].
    x (array-like): Mass fraction, defined as m_NaOH / (m_H2O + m_NaOH).
    
    Returns:
    pd (float or array-like): Pressure in [Pa]
    """
    zeta = x  # Mass fraction of salt in solution = Msalt / Msolution

    # Coefficients
    pi0 = 0.31
    pi1 = 3.698
    pi2 = 0.6
    pi3 = 0.231
    pi4 = 4.584
    pi5 = 0.49
    pi6 = 0.478
    pi7 = -5.2
    pi8 = -0.4
    pi9 = 0.018

    A = 2 - (1 + (zeta / pi0) ** pi1) ** pi2
    B = (1 + (zeta / pi3) ** pi4) ** pi5 - 1
    pi25 = (
        1
        - (1 + (zeta / pi6) ** pi7) ** pi8
        - pi9 * np.exp(-((zeta - 0.1) ** 2) / 0.005)
    )

    # Theta - reduced temperature
    TcH2O = 647.26  # K
    pcH2O = 22.064  # MPa
    Theta = (T + 273.15) / TcH2O  # Corrected variable name from 't' to 'T'

    fsol = A + B * Theta
    pi = pi25 * fsol

    # Vapor pressure of water
    tau = 1 - Theta

    A0 = -7.858230
    A1 = 1.839910
    A2 = -11.781100
    A3 = 22.670500
    A4 = -15.939300
    A5 = 1.775160

    lnpi = (
        A0 * tau
        + A1 * tau ** 1.5
        + A2 * tau ** 3
        + A3 * tau ** 3.5
        + A4 * tau ** 4
        + A5 * tau ** 7.5
    ) / (1 - tau)
    pdH2O = np.exp(lnpi) * pcH2O * 1e6  # Convert from MPa to Pa

    pd = pi * pdH2O

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)[0]
    if T < t_sol:
        print(f"Warning from CaCl2.saturation_pressure: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return pd

# DONE
def saturation_concentration(p, T):
    """
    Calculates the saturation concentration of NaOH in water based on the temperature and pressure.

    Author: Dorian Höffner
    Date: 2024-09-17

    Parameters:
        p (float): Pressure in Pa.
        T (float): Temperature in °C.

    Returns:
        float: Saturation concentration in kg NaOH / kg solution.
    """

    # Calculate the saturation concentration
    x = 0.001
    while saturation_pressure(x, T) - p > 1:   # 1 Pa tolerance
        x += 0.001

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)[0]
    if T < t_sol:
        print(f"Warning: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return x

# DONE
def density(x, T):
    """
    Calculates the density of CaCl2 solution based on concentration and temperature.

    Parameters:
    x (float): Mole fraction of NaOH in the solution. [m_CaCl2 / (m_h2o+m_CaCl2)]
    T (float): Temperature in °C.

    Returns:
    float: Density in kg/m^3.
    ---
    Author: Dorian Höffner 11/2024
    Source: CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """
    if x > 0.6:
        raise ValueError("Concentration of CaCl2 cannot be greater than 0.6 kg/kg. Correlation is not valid for this concentration.")
    
    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)[0]
    if T < t_sol:
        print(f"Warning: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    # Water Correlation
    def rho_h2o(T):
        B = [1.9937718430, 1.0985211604, -0.5094492996, -1.7619124270, -44.9005480267, -723692.2618632]
        rho_h2o_c = 322.0  # kg/m^3 (critical density of water)
        Tc = 647.1  # K
        Theta = (T+273.15) / Tc
        t = 1-Theta
        rho_h2o = rho_h2o_c * (1 + B[0] * t**(1/3) + B[1] * t**(2/3) + B[2] * t**(5/3) + B[3] * t**(16/3) + B[4] * t**(43/3) + B[5] * t**(110/3))
        return rho_h2o
    
    # CaCl2-H2O Correlation
    c = [1.0, 0.836014, -0.436300, 0.105642]
    rho = rho_h2o(T) * sum([c[i] * (x/(1-x))**i for i in range(4)])

    return rho # density in kg/m^3

# # DONE
# def density_old(x, T):
#     """
#     Calculates the density of CaCl2 solution based on concentration and temperature.

#     Parameters:
#     x (float): Mole fraction of CaCl2 in the solution. [m_CaCl2 / (m_h2o+m_CaCl2)]
#     T (float): Temperature in °C.

#     Returns:
#     float: Density in kg/m^3.
#     ---
#     Author: Dorian Höffner 11/2024
#     Data extracted from: Hao et al. 2016 "Modeling of CO2 solubility in single and mixed electrolyte solutions using statistical associating fluid theory"
#     """
#     # Convert molality to concentration in kg/kg for CaCl solution
#     molalities = np.array([
#         [0.2380952380952381, 0.7471264367816093, 1.2807881773399015, 1.781609195402299, 2.3070607553366176,
#         2.889983579638752, 3.472906403940887, 4.010673234811166, 4.486863711001642, 4.909688013136289],
#         [0.24630541871921183, 0.7594417077175698, 1.293103448275862, 1.7980295566502464, 2.323481116584565,
#         2.9105090311986865, 3.477011494252874, 4.018883415435139, 4.503284072249589, 4.909688013136289],
#         [0.2545155993431856, 0.7881773399014779, 1.3382594417077176, 1.8267651888341545, 2.3399014778325125,
#         2.935139573070608, 3.493431855500821, 4.0353037766830875, 4.5073891625615765, 4.917898193760263],
#         [0.2504105090311987, 0.8087027914614122, 1.3669950738916257, 1.8596059113300494, 2.376847290640394,
#         2.9761904761904763, 3.501642036124795, 4.059934318555008, 4.527914614121511, 4.946633825944171],
#         [0.513136288998358, 0.8292282430213465, 1.416256157635468, 1.8308702791461413, 2.389162561576355,
#         3.0049261083743843, 3.5180623973727423, 4.072249589490969, 4.499178981937603, 4.9384236453201975]
#     ])

#     M_CaCl2 = 110.98  # g/mol for CaCl2
#     x_values = molalities * M_CaCl2 / 1000  # Convert molality to kg/kg

#     temperatures = [
#         [298.15] * 10,
#         [323.15] * 10,
#         [373.15] * 10,
#         [423.15] * 10,
#         [473.15] * 10
#     ]

#     densities = [
#         [1.026256077795786, 1.0656401944894651, 1.1055105348460292, 1.1434359805510534, 1.1794165316045382,
#         1.220259319286872, 1.2572123176661265, 1.2922204213938413, 1.3218800648298217, 1.347163695299838],
#         [1.013128038897893, 1.0544570502431119, 1.0952998379254457, 1.1322528363047002, 1.1696920583468395,
#         1.2100486223662885, 1.247001620745543, 1.280551053484603, 1.3097244732576985, 1.3340356564019449],
#         [0.9810372771474878, 1.0257698541329012, 1.0675850891410048, 1.1055105348460292, 1.1414910858995138,
#         1.1828200972447327, 1.220745542949757, 1.2552674230145868, 1.2829821717990275, 1.3082658022690439],
#         [0.9397082658022691, 0.986871961102107, 1.033063209076175, 1.0709886547811993, 1.1094003241491086,
#         1.152674230145867, 1.1881685575364669, 1.2246353322528363, 1.253808752025932, 1.2781199351701784],
#         [0.9149108589951378, 0.9440842787682334, 0.993679092382496, 1.0286871961102106, 1.072933549432739,
#         1.1176661264181524, 1.1541329011345218, 1.1915721231766614, 1.2188006482982172, 1.2455429497568882]
#     ]

#     # Flatten data for interpolation
#     flat_concentrations = np.concatenate(x_values)
#     flat_temperatures = np.concatenate(temperatures)
#     flat_densities = np.concatenate(densities)

#     # Create grid data
#     points = np.array(list(zip(flat_concentrations, flat_temperatures)))
#     values = flat_densities

#     # Interpolate density at the given (x, T) point
#     return griddata(points, values, (x, T + 273.15), method='linear')
    

# NOT AVAILABLE YET
# DONE
def specific_heat_capacity(x, T):
    """
    Calculate the specific heat capacity of a CaCl2 solution (CaCl2-H2O) as a function of temperature and concentration.
    
    ---
    x: float
        Concentration of CaCl2 in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Specific heat capacity of the solution [kJ/kgK]
    ---
    Author: Dorian Höffner
    Date: 2024-11-04
    Source: Conde 2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    # Coefficients
    a = np.where(T <= 0, 830.54602, 88.7891)
    b = np.where(T <= 0, -1247.52013, -120.1958)
    c = np.where(T <= 0, -68.60350, -16.9264)
    d = np.where(T <= 0, 491.27650, 52.4654)
    e = np.where(T <= 0, -1.80692, 0.10826)
    f = np.where(T <= 0, -137.51511, 0.46988)

    A = 1.63799
    B = -1.69002
    C = 1.05124
    D = 0.0
    E = 0.0
    F = 58.5225
    G = -105.6343
    H = 47.7948

    # Theta - reduced temperature
    Theta = (T + 273.15) / 228 - 1

    # cpH2O calculation
    cpH2O = (a + b * Theta**0.02 + c * Theta**0.04 + 
             d * Theta**0.06 + e * Theta**1.8 + f * Theta**8)

    # f1 calculation
    f1 = A * x + B * x**2 + C * x**3

    # f2 calculation
    f2 = F * Theta**0.02 + G * Theta**0.04 + H * Theta**0.06

    # cp calculation
    cp = cpH2O * (1 - f1 * f2)

    return cp

    #return cp # specific heat capacity in kJ/kgK

# DONE
def dynamic_viscosity(x, T):
    """
    Calculate the dynamic viscosity of a NaOH solution as a function of temperature and concentration.
    The function is based on the following publication: "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    ---
    x: float
        Concentration of NaOH in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Dynamic viscosity of the solution [Pa s]
    ---
    Last Change: Dorian Höffner 10/2024
    Author: O. Buchin 09/2010
    """

    p = 1.01325  # bar

    # Coefficients
    eta1 = -0.169310
    eta2 = 0.817350
    eta3 = 0.574230
    eta4 = 0.398750
    
    # Intermediate values
    xi = x
    zeta = xi / ((1 - xi) ** (1 / 0.6))
    
    # Coefficients for eta_H2O calculation
    A = 1.0261862
    B = 12481.702
    C = -19510.923
    D = 7065.286
    E = -395.561
    F = 143922.996
    
    # Reduced temperature
    Theta = (T + 273.15) / 228 - 1

    # Calculate eta_H2O for T <= 0 and T > 0
    etaH2O_0 = steamTable.my_pt(p, 0.0000001)
    eta_H2O = np.where(T <= 0, 
                       etaH2O_0 * (A + B * Theta**0.02 + C * Theta**0.04 + D * Theta**0.08 + E * Theta**2.85 + F * Theta**8),
                       steamTable.my_pt(p, T))

    
    # Final viscosity calculation
    TcH2O = 647.26  # K
    Theta = (T + 273.15) / TcH2O
    eta = eta_H2O * np.exp(eta1 * zeta**3.6 + eta2 * zeta + eta3 * zeta / Theta + eta4 * zeta**2)
    
    # Output in mPas
    return float(eta * 1000)

# DONE
def diffusion_coefficient(x, T):
    """
    Computes the self diffusion coefficient of a CaCl2 solution.

    Parameters:
        T (np.ndarray or float): Temperature of the solution (in Celsius) (column vector).
        x (np.ndarray or float): Concentration of solution (0 to 1) kgSalt/kgSolution (column vector).
    
    Returns:
        float or np.ndarray: Self diffusion coefficient (m^2/s).

    ---
    Author: O. Buchin 03/2011 (original matlab)
    Last change: Dorian Höffner 11/2024
    Source: Holz, Manfred; Heil, Stefan R.; Sacco, Antonio "Temperature-dependent self-diffusion coefficients of water and six selected molecular liquids for calibration in accurate 1H NMR PFG measurements"
    """
    
    # Self Diffusion coefficient for water
    D0 = 1.635e-8  # m^2/s
    TS = 215.05    # K
    gamma = 2.063
    
    Dw = D0 * (((T + 273.15) / TS) - 1) ** gamma
    
    # Self Diffusion coefficient for solution
    d1 = 0.55
    d2 = -5.52
    d3 = -0.56
    
    D = Dw * (1 - (1 + (np.sqrt(x) / d1) ** d2) ** d3)
    
    # Return as float if input is scalar
    return float(D) if np.isscalar(D) else D

# NOT AVAILABLE YET
def thermal_conductivity(x, T, p):
    """
    Calculate the thermal conductivity of a NaOH solution as a function of temperature and concentration.
    The function is based on the following publication:
    ---
    x: float
        Concentration of NaOH in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Thermal conductivity of the solution [W/mK]
    ---
    """
    print("Thermal conductivity function not implemented yet.")

    #return lambda_NaOH # [W/mK]

def hxDiagram(editablePlot=False):
    """
    Plots the enthalpy-concentration (h-x) diagram for CaCl₂-H₂O solutions over a range of temperatures.
    
    Parameters:
        editablePlot (bool, optional): If True, the plot remains open for further editing (e.g., in interactive
            environments). If False (default), the plot is displayed immediately.
    Notes:
        - The crystallization curve indicates the onset of salt crystallization in the solution.

    Author: Dorian Höffner 2024-11-22    

    Returns:
        None
    """

    # suppress all print statements
    devnull = open(os.devnull, 'w')
    stout_old = sys.stdout
    sys.stdout = devnull

    plt.figure(dpi=300)
    T_array = np.array(range(0, 101, 2))
    x_array = np.linspace(0.001, 0.6, 50)

    for T in T_array:
        h_array = np.array([enthalpy(x, T, prevent_errors=True) for x in x_array])
        if T%20 == 0:
            plt.plot(x_array, h_array, color='black', alpha=0.7, lw=0.8, zorder=0)
            label_posx = x_array[15] + 0.01
            plt.text(label_posx, h_array[15]+10, f'{T} °C', fontsize=8, color='black')
        else:
            plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2, zorder=0)

    plt.xlabel('Concentration $x=[kg_{CaCl_{2}}/kg_{solution}]$')
    plt.ylabel('Enthalpy $h=[kJ/kg]$')
    plt.xlim(0, 0.6)
    plt.ylim(-50, 800)

    # plot crystallization curve
    cryst_data = crystallization_curve(return_data=True)
    cryst_x = np.array([x for x, T in cryst_data], dtype=float).flatten()
    cryst_T = np.array([T[0] for x, T in cryst_data], dtype=float).flatten()
    cryst_h = np.array([enthalpy(x, T) if T > 0 else np.nan for x, T in cryst_data], dtype=float).flatten()
    plt.plot([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)],
             color='black', linestyle="--", lw=0.5,
             label='Crystallization Curve', zorder=101)
    # fill between crystallization curve and xaxis
    plt.fill_between(cryst_x, cryst_h, -50, where=(cryst_h > -50), color='white', zorder=100)

    if not editablePlot:
        plt.show()

# DONE
def pTDiagram(log=True, invT=True, editablePlot=False, show_percentages=True):
    """
    Plots the pressure-temperature diagram for NaOH-H2O solutions.
    Author: Dorian Höffner 2024-04-26
    Last Change: 2024-09-17
    
    Parameters:
        log (bool): If True, the y-axis will be logarithmic.
        invT (bool): If True, the x-axis will be scaled as -1/T.
        editablePlot (bool): If True, the plot will be editable.
        show_percentages (bool): If True, the concentrations will be labeled.

    Returns:
        None
    """

    # turn off print statements
    devnull = open(os.devnull, 'w')
    stdout_old = sys.stdout
    sys.stdout = devnull

    
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

    # get data to plot crystallization curve
    cryst_data = crystallization_curve(return_data=True)
    cryst_T = np.array([T for x, T in cryst_data]).flatten()
    cryst_p = np.array([saturation_pressure(x, T) for x, T in cryst_data]).flatten()
    
    # Temperature range
    temperaturesC = np.arange(0, 110, 1)
    temperaturesK = temperaturesC + 273.15
    concentrations = np.arange(0.1, 0.69, 0.01)
    cr_temperaturesC = cryst_T
    cr_temperaturesK = cryst_T + 273.15
    
    # Prepare the plot
    plt.figure(dpi=300)
    
    # these temperatures are used for the x-axis
    plotTemperatures = np.arange(0, 101, 10) + 273.15
    
    # Calculate water vapor pressure using XSteam
    waterPressure = [steamTable.psat_t(T - 273.15) * 1e5 for T in temperaturesK]  # convert bar to Pa
    
    # Plot the data
    for x in concentrations:

        # Calculate saturation pressure for each temperature at the given concentration
        p = []
        for T in temperaturesC:
            p.append(saturation_pressure(x, T))  # Assuming PressureNaOH is defined elsewhere

    
        # Set color and line width
        color = "black" if int(np.round(x * 100)) % 10 == 0 else "grey"
        lw = 1.0 if color == "black" else 0.25
        
        # Plotting based on conditions
        temp_plot       = -1/temperaturesK     if invT else temperaturesK
        temp_plot_cryst = -1/(cryst_T+273.15)  if invT else cryst_T+273.15
        ylabel = 'Saturation Pressure [Pa]' if log else 'Saturation Pressure [Pa]'
        xlabel = 'Temperature [°C]' if invT else 'Temperatur [°C]'
        
        # Plot isosteres
        if log:
            plt.semilogy(temp_plot, p, color=color, lw=lw)
        else:
            plt.plot(temp_plot, p, color=color, lw=lw)
        
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        #plt.xticks(-1/plotTemperatures if invT else plotTemperatures, plotTemperatures - 273.15)
        plt.xticks(-1/plotTemperatures if invT else plotTemperatures, [f"{t-273.15:.0f}" for t in plotTemperatures])
        
        # Label concentrations
        if show_percentages and int(np.round(x * 100)) % 10 == 0 and x>0.29:
            label_pos = temp_plot[-1] + 1e-5 if invT else temp_plot[-1]+2
            plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8, color='black')#

    # add description for concentration: % = "kg NaOH / kg solution"
    plt.text(label_pos, p[-1] * 0.85, r'$\left[\frac{\mathrm{kg_{CaCl_2}}}{\mathrm{kg_{Solution}}}\right]$', fontsize=11, color='black')

    # Plotting water line
    if log:
        plt.semilogy(temp_plot, waterPressure, color="grey", linestyle='--',
                     label='Pure Water')
    else:
        plt.plot(temp_plot, waterPressure, color="grey", linestyle='--', label='Pure Water')

    # Plotting crystallization curve
    if log:
        plt.semilogy(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
                    lw=1.0, label='Crystallization Curve', zorder=101)
        # fill area between crystallization curve and the minimum positive value (e.g., 1)
        plt.fill_between(temp_plot_cryst, cryst_p, 1, where=(cryst_p > 1), color='white', zorder=100)
    else:
        plt.plot(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
                 lw=1.0, label='Crystallization Curve', zorder=101)
        # fill area between crystallization curve and a fixed value (e.g., 50)
        plt.fill_between(temp_plot_cryst, cryst_p, 1, where=(cryst_p > 1),  color='white', zorder=100)

    

    # Setting axis limits
    if invT:
        plt.xlim(-1/temperaturesK[0], -1/temperaturesK[-1])
    else:
        plt.xlim(temperaturesK[0], temperaturesK[-1])
    
    if log:
        plt.ylim(220, 1.1e5)
    else:
        plt.ylim(0, max(waterPressure) * 1.1)  # Adjust as needed to make sure all data is visible


    plt.legend()
    
    if not editablePlot:
        plt.show()

    # turn on print statements again
    sys.stdout = stdout_old

# DONE
def solubility_temperature(xs):
    """
    Computes the solubility boundary temperature with salt concentration xs.

    Parameters:
        xs (array-like): Concentration of solution (0...1) kgSalt/kgSolution (column vector).

    Returns:
        ts (numpy array): Temperature of crystallization (column vector). If xs is a scalar, ts is a scalar.

    Restrictions:
        - The concentration of salt must be between 0.0 and 0.78.
    """

    if np.any(xs < 0) or np.any(xs > 0.78):
        raise ValueError("The concentration of salt must be between 0.0 and 0.78.")

    # Coefficients
    A0 = np.array([0.422088, -0.378950, -0.519970, -1.149044, -2.385836, -2.807560])
    A1 = np.array([-0.066933, 3.456900, 3.400970, 5.509111, 8.084829, 4.678250])
    A2 = np.array([-0.282395, -3.531310, -2.851290, -4.642544, -5.303476, 0.000000])
    A3 = np.array([-355.514247])

    # Reduced temperature
    TcH2O = 647.26  # K

    # Initialize temperature array
    if isinstance(xs, (list, np.ndarray)):
        t = np.ones((6, len(xs)))
    else:
        t = np.ones((6, 1))

    # Iceline
    k = 0
    theta = A0[k] + A1[k] * xs + A2[k] * xs ** 2.0 + A3[k] * xs ** 7.5
    t[k] = theta * TcH2O - 273.15

    # Hydrates
    for k in range(1, 6):
        theta = A0[k] + A1[k] * xs + A2[k] * xs ** 2.0
        t[k] = theta * TcH2O - 273.15

    ts = np.max(t, axis=0)

    if len(ts.shape) == 1:
        ts = ts[0]
    return ts

# DONE
def crystallization_curve(return_data=False):
    """    
    Author: Dorian Höffner
    Date: 2024-11-20
    Publication: Conde 2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design"

    Parameters:
        return_data (bool): If True, the data will be returned as a list of lists.

    Returns:
        None or list of lists: If return_data is True, the data will be returned as a list of lists (x, T).
    """

    # cryst_data = x, T // create cryst_data based on "solubility_temperate"
    xs_array = np.linspace(0.01, 0.78, 100)
    cryst_data = [[xs, solubility_temperature(xs)] for xs in xs_array]
    temperatures = [item[1] for item in cryst_data]
    
        
    if return_data:
        return cryst_data
    
    else:
        cryst_data = [[xs_array[i]*100, temperatures[i]] for i,_ in enumerate(xs_array)]
        # prepare plot
        plt.figure(figsize=(6,4), dpi=300)
        plt.plot([x[0] for x in cryst_data],
                    [x[1] for x in cryst_data],
                    label='Crystallization Curve',
                    color="black")
        plt.xlabel(r'$\mathrm{CaCl_2}$ Concentration [%]')
        plt.ylabel('Temperature [°C]')
        plt.xlim(0, 75)
        plt.ylim(-60, 200)
        # make beautiful grid
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
        # add legend (top left)
        plt.legend(loc='upper left')
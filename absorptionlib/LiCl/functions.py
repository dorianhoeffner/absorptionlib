import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt

from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

import sys
import os
import contextlib

# LiCl property functions

# DONE
import sys
import contextlib
from scipy.optimize import fsolve


def documentation():
    print("""
This module contains functions for calculating properties of LiCl-H2O solutions:

| Function Name              | Description                                                                                   |
|---------------------------|------------------------------------------------------------------------------------------------|
| saturation_temperature    | Calculate the boiling point temperature of an aqueous Lithium Bromide solution.                |
| enthalpy                  | Calculate the enthalpy of an H2O-LiCl solution at a given temperature and concentration.       |
| differential_enthalpy_AD  | Calculates the differential enthalpy of a CaCl2 solution.                                      |
| saturation_pressure       | Calculate the equilibrium pressure of an H2O-LiCl solution.                                    |
| saturation_concentration  | Calculates the saturation concentration of LiCl in water based on the temperature and pressure.|
| density                   | Calculate the density of a water-LiCl solution.                                                |
| specific_heat_capacity    | Calculate the specific heat capacity of a CaCl2 solution.                                      |
| dynamic_viscosity         | Calculate the dynamic viscosity of a LiCl solution.                                            |
| diffusion_coefficient     | Computes the self diffusion coefficient of a CaCl2 solution.                                   |
| hxDiagram                 | Plots the pressure-temperature diagram for LiCl-H2O solutions.                                 |
| pTDiagram                 | Plots the pressure-temperature diagram for LiCl-H2O solutions.                                 |
| solubility_temperature    | Calculate the crystallization temperature of LiCl solution in water.                           |

For more information use the following function: LiCl.explain("function_name")

For example: LiCl.explain("enthalpy")
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



def saturation_temperature(x, p):
    """
    Calculates the saturation temperature of a LiCl solution based on the pressure and concentration.
    ---
    Parameters:
        x (float): Concentration of LiCl in the solution [kg/kg].
        p (float): Pressure in [Pa].
    ---
    author: Dorian Höffner 11/2024
    """

    # Silence print statements during fsolve
    T_guess = 20
    with contextlib.redirect_stdout(sys.stdout):  # Redirect stdout temporarily
        T = fsolve(lambda T: saturation_pressure(x, T) - p, T_guess)[0]

    # Check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.saturation_temperature: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")
    
    return T


# DONE
def enthalpy(x, T, prevent_errors=False):
    """
    Calculate the specific enthalpy of a LiCl-H2O solution at a given concentration and temperature.
    ---
    Parameters:
        x (float): Mass fraction of LiCl in the solution, defined as x = m_LiCl / (m_H2O + m_LiCl). Must be between 0 and 1.
        T (float): Temperature in degrees Celsius [°C].
        prevent_errors (bool, optional): If True, suppresses warnings and error checks related to solubility limits. Default is False.
    ---
    Returns:
        float: Specific enthalpy of the solution in kJ/kg. Returns np.nan if the temperature is below the solubility temperature and prevent_errors is False.
    ---
    Notes:
        - The enthalpy is calculated as the sum of the ideal enthalpy and the excess enthalpy of the solution.
        - The ideal enthalpy is based on the specific heat capacities of water and LiCl at 25% mass fraction.
        - The excess enthalpy is computed via numerical integration of the differential enthalpy of solution.
        - If the temperature is below the solubility temperature for the given concentration and prevent_errors is False, a warning is printed and np.nan is returned.
    ---
    References:
        - Differential enthalpy calculations are based on the function `differential_enthalpy_AD`.
        - Specific heat capacities are obtained from `steamTable.CpL_t` and `specific_heat_capacity`.
    
    Date:   2025-03-20
    Author: Dorian Höffner
    """
    
    def excess_enthalpy(x, T):
        def dh_sol(x, T):
            return differential_enthalpy_AD(x, T, prevent_errors=True)
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
            print(f"Warning from LiCl.enthalpy: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")
            return np.nan

    return ideal_enthalpy(x, T) + excess_enthalpy(x, T)
    

# DONE
def differential_enthalpy_AD(x, T, prevent_errors=False):
    """
    Calculates the differential enthalpy [kJ/kgH2O] of a LiCl solution based on concentration and temperature.
    How much additional energy (in comparison to pure vaporization enthalpy) is needed to vaporize a certain amount of water in a LiCl solution.
    
    Parameters:
        x (float): Concentration of LiCl in the solution [kg/kg].
        T (float): Temperature of the solution [°C].
        prevent_errors (bool): If True, suppresses warnings and error checks related to solubility limits. Default is False.

    Returns:
        float: Differential enthalpy [kJ/kgH2O].

    Author: Dorian Höffner 11/2024
    Source: CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    H1 = 0.845
    H2 = -1.965
    H3 = -2.265
    H4 = 0.6
    H5 = 169.105
    H6 = 457.850

    # Reduced Temperature (with critical T of water)
    Theta = (T + 273.15) / 647.1 # FLAG

    # zeta
    zeta = x / (H4-x)

    # Reference differental enthalpy
    dh_dil0 =  H5 + H6 * Theta

    # differental enthalpy
    dh_dil = dh_dil0 * (1 + (zeta/H1)**H2)**H3

    if not prevent_errors:
        # check if the solution is in liquid state
        t_sol = solubility_temperature(x)
        if T < t_sol:
            print(f"Warning from LiCl.differential_enthalpy_AD: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return dh_dil


# DONE
def saturation_pressure(x, T):
    """
    Computes the vapor pressure over a solution of LiCl with temperature T and concentration x.

    Parameters:
        T (numpy array): Temperature of the solution in [°C]
        x (numpy array): Concentration of the solution as m_LiCl / (m_H2O + m_LiCl)
    ---
    Returns:
        pd (float or array-like): Vapor pressure over the solution in [Pa]
    ---
    Last Change: Dorian Höffner 11/2024
    Author: O. Buchin 03/2011
    Source: CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """
    # Mass fraction of salt in solution
    zeta = x

    # Coefficients
    pi0 = 0.28
    pi1 = 4.3
    pi2 = 0.6
    pi3 = 0.21
    pi4 = 5.1
    pi5 = 0.49
    pi6 = 0.362
    pi7 = -4.75
    pi8 = -0.4
    pi9 = 0.03

    # Calculate A and B
    A = 2 - (1 + (zeta / pi0)**pi1)**pi2
    B = ((1 + (zeta / pi3)**pi4)**pi5) - 1
    pi25 = 1 - (1 + (zeta / pi6)**pi7)**pi8 - pi9 * np.exp(-((zeta - 0.1)**2) / 0.005)

    # Constants for water properties
    TcH2O = 647.26  # K
    pcH2O = 22.064  # MPa

    # Reduced temperature
    Theta = (T + 273.15) / TcH2O

    # Calculate fsol and pi
    fsol = A + B * Theta
    pi = pi25 * fsol

    # Calculate vapor pressure of water
    tau = 1 - Theta

    A0 = -7.858230
    A1 = 1.839910
    A2 = -11.781100
    A3 = 22.670500
    A4 = -15.939300
    A5 = 1.775160

    lnpi = (A0 * tau + A1 * tau**1.5 + A2 * tau**3 + A3 * tau**3.5 + 
            A4 * tau**4 + A5 * tau**7.5) / (1 - tau)
    pdH2O = np.exp(lnpi) * pcH2O * 1e6  # Convert to Pa

    # Final vapor pressure over the solution
    pd = pi * pdH2O

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.saturation_pressure: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return pd

    
# DONE
def saturation_concentration(p, T):
    """
    Calculates the saturation concentration of LiCl in water based on the temperature and pressure.

    Author: Dorian Höffner
    Date: 2024-09-17

    Parameters:
        p (float): Pressure in Pa.
        T (float): Temperature in °C.

    Returns:
        float: Saturation concentration in kg LiCl / kg solution.
    """

    # turn off print statemtnes for fsolve
    stout_old = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull

    # Calculate the saturation concentration
    x_guess = 0.001
    x = fsolve(lambda x: saturation_pressure(x, T) - p, x_guess)[0]

    # turn on print statements again
    sys.stdout = stout_old

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.saturation_concentration: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return x

# DONE
def density(x, T):
    """
    Calculates the density of LiCl solution based on concentration and temperature.

    Parameters:
    x (float): Mole fraction of NaOH in the solution. [m_LiCl / (m_h2o+m_LiCl)]
    T (float): Temperature in °C.

    Returns:
    float: Density in kg/m^3.
    ---
    Author: Dorian Höffner 11/2024
    Source: CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """
    if x > 0.56:
        raise ValueError("Concentration of LiCl cannot be greater than 0.56 kg/kg. Correlation is not valid for this concentration.")

    # Water Correlation
    def rho_h2o(T):
        B = [1.9937718430, 1.0985211604, -0.5094492996, -1.7619124270, -44.9005480267, -723692.2618632]
        rho_h2o_c = 322.0  # kg/m^3 (critical density of water)
        Tc = 647.1  # K
        Theta = (T+273.15) / Tc
        t = 1-Theta
        rho_h2o = rho_h2o_c * (1 + B[0] * t**(1/3) + B[1] * t**(2/3) + B[2] * t**(5/3) + B[3] * t**(16/3) + B[4] * t**(43/3) + B[5] * t**(110/3))
        return rho_h2o
    
    # LiCl-H2O Correlation
    c = [1.0, 0.540966, -0.303792, 0.100791]
    rho = rho_h2o(T) * sum([c[i] * (x/(1-x))**i for i in range(4)])

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.density: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return rho # kg/m^3


# DONE
def specific_heat_capacity(x, T):
    """
    Calculate the specific heat capacity of a LiCl solution (LiCl-H2O) as a function of temperature and concentration.
    
    ---
    x: float
        Concentration of LiCl in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Specific heat capacity of the solution [kJ/kgK]
    ---
    Last Change: Dorian Höffner
    Author: O. Buchin 03/2011
    Date: 2024-11-04
    Source: Correlation according to CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    # Coefficients
    a = np.where(T <= 0, 830.54602, 88.7891)
    b = np.where(T <= 0, -1247.52013, -120.1958)
    c = np.where(T <= 0, -68.60350, -16.9264)
    d = np.where(T <= 0, 491.27650, 52.4654)
    e = np.where(T <= 0, -1.80692, 0.10826)
    f = np.where(T <= 0, -137.51511, 0.46988)

    A = 1.43980
    B = -1.24317
    C = -0.12070
    D = 0.12825
    E = 0.62934
    F = 58.5225
    G = -105.6343
    H = 47.7948

    # Theta - reduced temperature
    Theta = (T + 273.15) / 228 - 1

    # cpH2O calculation
    cpH2O = (a + b * Theta**0.02 + c * Theta**0.04 + 
             d * Theta**0.06 + e * Theta**1.8 + f * Theta**8)

    # f1 calculation
    f1 = np.where(x <= 0.31, 
                  A * x + B * x**2 + C * x**3, 
                  D + E * x)

    # f2 calculation
    f2 = F * Theta**0.02 + G * Theta**0.04 + H * Theta**0.06

    # cp calculation
    cp = cpH2O * (1 - f1 * f2)

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.specific_heat_capacity: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return cp

# DONE
def dynamic_viscosity(x, T):
    """
    Computes the dynamic viscosity of LiCl with temperature T and concentration x.
    ---
    Parameters:
        T (numpy array): Temperature of the solution (in Celsius) (column vector)
        x (numpy array): Concentration of the solution (0...1) kgSalt/kgSolution (column vector)
    ---
    Returns:
        numpy array: Dynamic viscosity (mPas) (column vector)
    ---
    Last Change: Dorian Höffner 10/2024
    Author: O. Buchin 09/2010
    Source: correlation according to CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    # Pressure in bar
    p = 1.01325  # bar

    # Coefficients
    eta1 = 0.090481
    eta2 = 1.390262
    eta3 = 0.675875
    eta4 = -0.583517

    # Mass fraction
    xi = x
    zeta = xi / ((1 - xi)**(1 / 0.6))

    # Constants for eta_H2O
    A = 1.0261862
    B = 12481.702
    C = -19510.923
    D = 7065.286
    E = -395.561
    F = 143922.996

    # Theta - reduced temperature
    Theta = (T + 273.15) / 228 - 1
    TcH2O = 647.26  # K
    
    # Placeholder for eta_H2O computation
    etaH2O_0 = 0.001  # This should be replaced with an accurate function for water viscosity

    # Calculate eta_H2O
    eta_H2O = np.where(
        T <= 0,
        etaH2O_0 * (A + B * Theta**0.02 + C * Theta**0.04 + D * Theta**0.08 +
                    E * Theta**2.85 + F * Theta**8),
        etaH2O_0 * np.ones_like(T)  # Replace with actual function for T > 0 if available
    )

    # Adjusted Theta for eta calculation
    Theta_adjusted = (T + 273.15) / TcH2O

    # Calculate dynamic viscosity
    eta = eta_H2O * np.exp(
        eta1 * zeta**3.6 +
        eta2 * zeta +
        eta3 * zeta / Theta_adjusted +
        eta4 * zeta**2
    )

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.dynamic_viscosity: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return eta * 1000  # Return result in mPas

# DONE
def diffusion_coefficient(x, T):
    """
    Computes the self diffusion coefficient of a XXXXXX solution.

    Parameters:
        T (np.ndarray or float): Temperature of the solution (in Celsius) (column vector).
        x (np.ndarray or float): Concentration of solution (0 to 1) kgSalt/kgSolution (column vector).
    
    Returns:
        float or np.ndarray: Self diffusion coefficient (m^2/s).

    ---
    Author: O. Buchin 03/2011 (original matlab)
    Last change: Dorian Höffner 11/2024
    Source: correlation according to CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    # Self Diffusion coefficient for water
    D0 = 1.635e-8  # m^2/s
    TS = 215.05  # K
    gamma = 2.063

    Dw = D0 * (((T + 273.15) / TS) - 1)**gamma

    # Coefficients for the solution
    d1 = 0.52
    d2 = -4.92
    d3 = -0.56

    # Self Diffusion coefficient for solution
    D = Dw * (1 - (1 + (np.sqrt(x) / d1)**d2)**d3)

    # check if the solution is in liquid state
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning from LiCl.diffusion_coefficient: The temperature {T} is below the solubility temperature {t_sol} at x={x}.")

    return D

# # TODO
# def thermal_conductivity(x, T, p):
#     """
#     Calculate the thermal conductivity of a NaOH solution as a function of temperature and concentration.
#     The function is based on the following publication:
#     ---
#     x: float
#         Concentration of NaOH in the solution [kg/kg]
#     T: float
#         Temperature of the solution [°C]
#     ---
#     returns: float
#         Thermal conductivity of the solution [W/mK]
#     ---
#     """
#     print("Thermal conductivity function not available.")

#     #return lambda_NaOH # [W/mK]

def hxDiagram(editablePlot=False):
    """
    Plots the pressure-temperature diagram for NaOH-H2O solutions.
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

    plt.xlabel('Concentration $x=[kg_{LiCl}/kg_{solution}]$')
    plt.ylabel('Enthalpy $h=[kJ/kg]$')
    plt.xlim(0, 0.6)
    plt.ylim(-50, 800)

    # plot crystallization curve
    cryst_data = crystallization_curve(return_data=True)
    cryst_T = np.array([T for x, T in cryst_data]).flatten()
    cryst_h = np.array([enthalpy(x, T) if T > 0 else np.nan for x, T in cryst_data]).flatten()
    plt.plot([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], color='black', linestyle="--", lw=0.5, label='Crystallization Curve')
    plt.fill_between([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], 0, color='white', zorder=0)    

    if not editablePlot:
        plt.show()

# DONE
def pTDiagram(log=True, invT=True, editablePlot=False, show_percentages=True):
    """
    Plots the pressure-temperature diagram for LiCl-H2O solutions.
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

    # suppress print statements (warnings
    stout_old = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

    # get data to plot crystallization curve
    cryst_data = crystallization_curve(return_data=True)
    cryst_T = np.array([T for x, T in cryst_data]).flatten()
    cryst_p = np.array([saturation_pressure(x, T) for x, T in cryst_data]).flatten()
    
    # Temperature range
    temperaturesC = np.arange(0, 110, 1)
    temperaturesK = temperaturesC + 273.15
    concentrations = np.arange(0.1, 0.59, 0.01)
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
        if show_percentages and int(np.round(x * 100)) % 10 == 0 and x>0.19:
            label_pos = temp_plot[-1] + 1e-5 if invT else temp_plot[-1]+2
            plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8, color='black')

    # add description for concentration: % = "kg NaOH / kg solution"
    plt.text(label_pos, p[-1] * 0.85, r'$\left[\frac{\mathrm{kg_{LiCl}}}{\mathrm{kg_{Solution}}}\right]$', fontsize=11, color='black')

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
    sys.stdout = stout_old

# DONE
def solubility_temperature(xs):
    """
    Computes the solubility boundary temperature with salt concentration xs.
    ---
    Parameters:
        xs (float or array-like): Concentration of solution (0...1) kgSalt/kgSolution.
    ---
    Returns:
        ts (float or array-like): Temperature of crystallization.
    ---
    Author: O. Buchin 03/2011
    Last change: Dorian Höffner 11/2024
    Source: correlation according to CONDE2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
    """

    if np.isscalar(xs):
        xs = np.array([xs])
    

    # Coefficients
    A0 = np.array([0.422088, -0.005340, -0.560360, -0.315220, -1.312310, -1.356800])
    A1 = np.array([-0.090410, 2.015890, 4.723080, 2.882480, 6.177670, 3.448540])
    A2 = np.array([-2.936350, -3.114590, -5.811050, -2.624330, -5.034790, 0.000000])

    # Constants
    TcH2O = 647.26  # K

    # Initialize temperature array
    t = np.zeros((xs.shape[0], 6))

    # Compute temperature for the ice line (k=1)
    k = 0
    theta = A0[k] + A1[k] * xs + A2[k] * xs**2.5
    t[:, k] = theta * TcH2O - 273.15

    # Compute temperatures for hydrates (k=2 to 6)
    for k in range(1, 6):
        theta = A0[k] + A1[k] * xs + A2[k] * xs**2.0
        t[:, k] = theta * TcH2O - 273.15

    # Find the maximum temperature across all hydrate cases
    ts = np.max(t, axis=1)

    if len(ts) == 1:
        ts = ts[0]

    return ts

# DONE
def crystallization_curve(return_data=False):
    """    
    Author: Dorian Höffner
    Date: 2024-11-04

    Parameters:
        return_data (bool): If True, the data will be returned as a list of lists.

    Returns:
        None or list of lists: If return_data is True, the data will be returned as a list of lists (x, T).
    """

    # cryst_data = x, T // create cryst_data based on "solubility_temperate"
    xs_array = np.linspace(0.01, 0.6, 100)
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
        plt.xlim(0, 60)
        plt.ylim(-100, 200)
        # make beautiful grid
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
        # add legend (top left)
        plt.legend(loc='upper left')
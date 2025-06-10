import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

import sys
import os
import contextlib
import io

###### INEFFICIENCIES ######
# "saturation_concentration: Uses a while loop with small increments, which can be slow."
# "enthalpy: Repeated polynomial calculations could be optimized."
# "saturation_pressure: Contains many warnings and checks that could be streamlined."
# "saturation_temperature: Redirects stdout to suppress warnings, which is inefficient."

def documentation():
    print("""
This module contains functions for calculating properties of NaOH-H2O solutions:

| Function Name              | Description                                                                                   |
|---------------------------|------------------------------------------------------------------------------------------------|
| saturation_temperature    | Calculate the boiling point temperature of an aqueous Lithium Bromide solution.                |
| enthalpy                  | Calculate the enthalpy of an H2O-LiBr solution at a given temperature and concentration.       |
| differential_enthalpy_AD  | Calculates the differential enthalpy of a CaCl2 solution.                                      |
| saturation_pressure       | Calculate the equilibrium pressure of an H2O-LiBr solution.                                    |
| saturation_concentration  | Calculates the saturation concentration of LiBr in water based on the temperature and pressure.|
| density                   | Calculate the density of a water-LiBr solution.                                                |
| specific_heat_capacity    | Calculate the specific heat capacity of a CaCl2 solution.                                      |
| dynamic_viscosity         | Calculate the dynamic viscosity of a LiBr solution.                                            |
| thermal_conductivity      | Calculate the thermal conductivity of a LiBr solution.                                         |
| hxDiagram                 | Plots the pressure-temperature diagram for LiBr-H2O solutions.                                 |
| pTDiagram                 | Plots the pressure-temperature diagram for LiBr-H2O solutions.                                 |
| solubility_temperature    | Calculate the crystallization temperature of LiBr solution in water.                           |

For more information use the following function: NaOH.explain("function_name")

For example: NaOH.explain("enthalpy")
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


# NaOH property functions
def saturation_temperature(x, P, prevent_errors=False):
    """
    Calculates the boiling point temperature of aqueous NaOH-H2O solutions at high concentrations.
    Uses saturation_pressure() to calculate the pressure and then calculates the temperature.
    ---
    Parameters:
        x (float): Mass fraction, defined as m_NaOH / (m_H2O + m_NaOH).
        P (float): Pressure in [Pa].
        prevent_errors (bool): If True, suppresses warnings and returns None for invalid inputs.
    ---
    Returns:
        float: Boiling point temperature in [°C].
    ---
    Author: Dorian Höffner
    Berechnung nach J. Olsson et. al "Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations",
    International Journal of Thermophysics, Vol. 18, No. 3, 1997
    Boiling point temperature of aqueous NaOH-H2O Solution in °C
    """

    T_guess = 50 # initial guess for fsolve
    # suppress print statements
    with contextlib.redirect_stdout(io.StringIO()):
        #T = fsolve(lambda T: saturation_pressure(x, T, prevent_errors=True) - P, T_guess)[0]
        T = brentq(lambda T: saturation_pressure(x, T, prevent_errors=True) - P, 1, 200)
    
    # check crystallization line
    t_sol = solubility_temperature(x)
    if T < t_sol:
        print(f"Warning: T = {T} °C is below the crystallization line at x={x}. Saturation temperature function at x={x} not valid for T < {t_sol}°C")
        if not prevent_errors:
            return None

    # implement checks for input values (checks are from saturation_pressure())
    if T < 0:
        print(f"Warning: values are outside the valid range, because T < 0 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
    elif T < 20:
        if 1-x < 0.582:
            print(f"Warning: values are outside the valid range for T < 20 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
    elif T < 60:
        if 1-x < 0.500:
            print(f"Warning: values are outside the valid range for 20 <= T < 60 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
    elif T < 70:
        if 1-x < 0.353:
            print(f"Warning: values are outside the valid range for 60 <= T < 70 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
    elif T < 150:
        if 1-x < 0.300:
            print(f"Warning: values are outside the valid range for 70 <= T < 150 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
    elif T <= 200:
        if 1-x < 0.200:
            print(f"Warning: values are outside the valid range for 150 <= T <= 200 °C\nx = {x}\np = {P} Pa\nT = {T} °C")
                
    return T

def enthalpy(x, T, prevent_errors=False):
    """
    Calculates the enthalpy of an aqueous NaOH-H2O solution at high concentrations.

    Parameter:
        T in [°C]
        x = m_NaOH/(m_H2O + m_NaOH)
        h in [kJ/kg]
        prevent_errors: If True, suppresses errors and returns NaN/None for invalid inputs.

    Returns:
        h (float): Enthalpy in [kJ/kg].

    Wertebereich: t in Grad Celsius!
    0<=t<4			0.780 <=xi<=1
    4=<t<10	        0.680<=xi<=1
    10=<t<15		0.580<=xi<=1
    15=<t<26		0.540<=xi<=1
    26 =<t<37		0.440<=xi<=1
    37=<t<48		0.400<=xi<=1
    48=<t<60		0.340<=xi<=1
    60=<t<71		0.300<=xi<=1
    71=<t<82		0.280<=xi<=1
    82=<t<93		0.240<=xi<=1
    93=< t=<204	0.220<=xi<=1
    
    Last Change: brought to python by Dorian Höffner
    Autor: 	Roman Ziegenhardt
    Quelle: 	Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations, J. OIsson, A. Jernqvist, G. Aly, 
    		International Journal of Thermophysics Vol. 18. No. 3. 1997
    zuletzt geändert: Elisabeth Thiele: Temperatur in °C und salt mass
    fraction statt water mass fraction
    
    """

    # Convert NaOH mass fraction to water mass fraction
    xi = 1 - x

    # Define temperature ranges and corresponding xi thresholds
    ranges = [
        (0, 4, 0.780),
        (4, 10, 0.680),
        (10, 15, 0.580),
        (15, 26, 0.540),
        (26, 37, 0.440),
        (37, 48, 0.400),
        (48, 60, 0.340),
        (60, 71, 0.300),
        (71, 82, 0.280),
        (82, 93, 0.240),
        (93, 204, 0.220)
    ]

    # Check temperature and xi thresholds
    if T < 0:
        print(f"Warning: T = {T} °C is below 0°C. Enthalpy function not valid for T < 0°C")
    elif T > 204:
        print(f"Warning: T = {T} °C is above 204°C. Enthalpy function not valid for T > 204°C")
    else:
        for Tmin, Tmax, xi_min in ranges:
            if Tmin <= T < Tmax and xi < xi_min:
                print(f"Warning: values are outside the valid range for {Tmin} <= T < {Tmax} °C\nx = {x}\nT = {T} °C")
                break

    # # implement checks for input values (checks are from saturation_pressure())
    # if T < 0:
    #     #raise ValueError('Temperature must be greater than 0 °C')
    #     print(f"Warning: T = {T} °C is below 0°C. Enthalpy function not valid for T < 0°C")
    # elif T < 4:
    #     if xi < 0.780:
    #         print(f"Warning: values are outside the valid range for T < 4 °C\nx = {x}\nT = {T} °C")
    # elif T < 10:
    #     if xi < 0.680:
    #         print(f"Warning: values are outside the valid range for 4 <= T < 10 °C\nx = {x}\nT = {T} °C")
    # elif T < 15:
    #     if xi < 0.580:
    #         print(f"Warning: values are outside the valid range for 10 <= T < 15 °C\nx = {x}\nT = {T} °C")
    # elif T < 26:
    #     if xi < 0.540:
    #         print(f"Warning: values are outside the valid range for 15 <= T < 26 °C\nx = {x}\nT = {T} °C")
    # elif T < 37:
    #     if xi < 0.440:
    #         print(f"Warning: values are outside the valid range for 26 <= T < 37 °C\nx = {x}\nT = {T} °C")
    # elif T < 48:
    #     if xi < 0.400:
    #         print(f"Warning: values are outside the valid range for 37 <= T < 48 °C\nx = {x}\nT = {T} °C")
    # elif T < 60:
    #     if xi < 0.340:
    #         print(f"Warning: values are outside the valid range for 48 <= T < 60 °C\nx = {x}\nT = {T} °C")
    # elif T < 71:
    #     if xi < 0.300:
    #         print(f"Warning: values are outside the valid range for 60 <= T < 71 °C\nx = {x}\nT = {T} °C")
    # elif T < 82:
    #     if xi < 0.280:
    #         print(f"Warning: values are outside the valid range for 71 <= T < 82 °C\nx = {x}\nT = {T} °C")
    # elif T < 93:
    #     if xi < 0.240:
    #         print(f"Warning: values are outside the valid range for 82 <= T < 93 °C\nx = {x}\nT = {T} °C")
    # elif T <= 204:
    #     if xi < 0.220:
    #         print(f"Warning: values are outside the valid range for 93 <= T <= 204 °C\nx = {x}\nT = {T} °C")
    # elif T>204:
    #     #raise ValueError('Temperature must be less than 204 °C')
    #     print(f"Warning: T = {T} °C is above 204°C. Enthalpy function not valid for T > 204°C")

    # Coefficients
    k = np.array([1288.4485, -0.49649131, -4387.8908, -4.0915144, 4938.2298, 7.2887292, -1841.1890, -3.0202651])
    l = np.array([2.3087919, -9.0004252, 167.59914, -1051.6368, 3394.3378, -6115.0986, 6220.8249, -3348.8098, 743.87432])
    m = np.array([0.02302860, -0.37866056, 2.4529593, -8.2693542, 15.728833, -16.944427, 9.6254192, -2.2410628])
    n = np.array([-8.5131313e-5, 136.52823e-5, -875.68741e-5, 2920.0398e-5, -5488.2983e-5, 5841.8034e-5, -3278.7483e-5, 754.45993e-5])

    # Calculation of coefficients
    c1 = (k[0] + k[2]*xi + k[4]*(xi**2) + k[6]*(xi**3)) / (1 + k[1]*xi + k[3]*(xi**2) + k[5]*(xi**3) + k[7]*(xi**4))
    c2 = np.polyval(l[::-1], xi)
    c3 = np.polyval(m[::-1], xi)
    c4 = np.polyval(n[::-1], xi)
    # c2 = l[0] + l[1]*xi + l[2]*(xi**2) + l[3]*(xi**3) + l[4]*(xi**4) + l[5]*(xi**5) + l[6]*(xi**6) + l[7]*(xi**7) + l[8]*(xi**8)
    # c3 = m[0] + m[1]*xi + m[2]*(xi**2) + m[3]*(xi**3) + m[4]*(xi**4) + m[5]*(xi**5) + m[6]*(xi**6) + m[7]*(xi**7)
    # c4 = n[0] + n[1]*xi + n[2]*(xi**2) + n[3]*(xi**3) + n[4]*(xi**4) + n[5]*(xi**5) + n[6]*(xi**6) + n[7]*(xi**7)


    # Calculate enthalpy
    h = c1 + c2*T + c3*(T**2) + c4*(T**3)

    if prevent_errors:
        t_sol = solubility_temperature(x)
        if T < t_sol:
            print(f"Warning: T = {T} °C is below the crystallization line at x={x}. Enthalpy function at x={x} not valid for T < {t_sol}°C")
            return np.nan

    return float(h)

def saturation_pressure(xi, T, prevent_errors=False):
    """
    Calculates the pressure of aqueous NaOH-H2O solutions at high concentrations.
    
    Last Change: Dorian Höffner 2024-04-26 (translated to Python, changed input T to [°C], changed return value to [Pa])
    Author: Anna Jahnke, Roman Ziegenhardt
    Source: Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations, J. Olsson, A. Jernqvist, G. Aly,
            International Journal of Thermophysics Vol. 18, No. 3, 1997
    
    Parameters:
    T (array-like): Temperature in [°C].
    x (array-like): Mass fraction, defined as m_NaOH / (m_H2O + m_NaOH).
    prevent_errors (bool): If True, suppresses errors and returns None for invalid inputs.
    
    Returns:
    p (float or array-like): Pressure in [Pa]

    Notes:
    Wertebereich: t in Grad Celsius!
    0<=t<20    		0.582<=1-x<=1
    20<=t<60		0.500<=1-x<=1
    60<=t<70       	0.353<=1-x<=1
    70<=t<150		0.300<=1-x<=1
    150<=t<=200		0.200<=1-x<=1
    """

    # check with crystallization line
    t_sol = solubility_temperature(xi)
    if T < t_sol:
        print(f"Warning: T = {T} °C is below the crystallization line at x={xi}. Saturation pressure function at x={xi} not valid for T < {t_sol}°C")
        if not prevent_errors:
            return None


    # convert NaOH mass fraction to water mass fraction
    x = 1 - xi

    # implement checks for input values (checks are from saturation_pressure())
    if T < 0:
        print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for T < 0 °C\nx = {xi}\nT = {T} °C")
    elif T < 20:
        if x <= 0.582:
            print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for T < 20 °C\nx = {xi}\nT = {T} °C")
    elif T < 60:
        if x <= 0.500:
            print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for 20 <= T < 60 °C\nx = {xi}\nT = {T} °C")
    elif T < 70:
        if x <= 0.353:
            print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for 60 <= T < 70 °C\nx = {xi}\nT = {T} °C")
    elif T < 150:
        if x <= 0.300:
            print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for 70 <= T < 150 °C\nx = {xi}\nT = {T} °C")
    elif T <= 200:
        if x <= 0.200:
            print(f"Warning: values are outside the validated range (from Olsson et al. 1997) for 150 <= T <= 200 °C\nx = {xi}\nT = {T} °C")

   
    if isinstance(x, (int, float)) or isinstance(T, (int, float)):
        pass
    elif len(x) != len(T):
        raise ValueError('x and T must have the same length')

    # Constants
    k = np.array([-113.93947, 209.82305, 494.77153, 6860.8330, 2676.6433,
                  -21740.328, -34750.872, -20122.157, -4102.9890])
    l = np.array([16.240074, -11.864008, -223.47305, -1650.3997, -5997.3118,
                  -12318.744, -15303.153, -11707.480, -5364.9554, -1338.5412,
                  -137.96889])
    m = np.array([-226.80157, 293.17155, 5081.8791, 36752.126, 131262.00,
                  259399.54, 301696.22, 208617.90, 81774.024, 15648.526,
                  906.29769])

    # Calculate pressure
    log_x = np.log(x)
    a1 = np.polyval(k[::-1], log_x)
    a2 = np.polyval(l[::-1], log_x)
    a3 = np.polyval(m[::-1], log_x)

    logP = (a1 + a2 * T) / (T - a3)
    p = np.exp(logP) * 1000

    return float(p) # pressure in Pa

# # Precompute the lookup table on import
# def precalculate_saturation_pressure_lookup_table(xi_range, T_range):
#     """
#     Precomputes a lookup table for saturation pressure values over specified ranges.
    
#     Parameters:
#     xi_range (iterable): Range of xi values (mass fraction).
#     T_range (iterable): Range of T values (temperature in °C).
    
#     Returns:
#     dict: Lookup table with keys as (xi, T) tuples and values as saturation pressure in [Pa].
#     """
#     lookup_table = {}
#     for xi in xi_range:
#         for T in T_range:
#             try:
#                 lookup_table[(round(xi, 3), round(T, 1))] = saturation_pressure(xi, T, prevent_errors=True)
#             except:
#                 lookup_table[(round(xi, 3), round(T, 1))] = None  # Handle invalid cases
#         return lookup_table

# # Generate the lookup table during import
# xi_values = np.linspace(0.2, 0.8, 601)
# T_values = np.linspace(0, 200, 2001)
# saturation_pressure_table = precalculate_saturation_pressure_lookup_table(xi_values, T_values)

# def saturation_pressure_lookup(xi, T, lookup_table=saturation_pressure_table, prevent_errors=False):
#     """
#     Efficiently calculates the pressure of aqueous NaOH-H2O solutions at high concentrations
#     using a lookup table for precomputed values.
    
#     Parameters:
#     xi (float): Mass fraction, defined as m_NaOH / (m_H2O + m_NaOH).
#     T (float): Temperature in [°C].
#     lookup_table (dict, optional): Precomputed lookup table with keys as (xi, T) tuples.
#     prevent_errors (bool): If True, suppresses warnings and returns None for invalid inputs.
    
#     Returns:
#     float: Pressure in [Pa].
#     """
#     if lookup_table is not None:
#         key = (round(xi, 4), round(T, 1))
#         if key in lookup_table:
#             return lookup_table[key]



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
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress stdout
        x = brentq(lambda x: saturation_pressure(x, T, prevent_errors=True) - p, 0.001, 0.785)
        # x = 0.001
        # while saturation_pressure(x, T) < p:
        #     x += 0.001
        


    # devnull = open(os.devnull, 'w')
    # stdold = sys.stdout
    # sys.stdout = devnull

    # x = fsolve(lambda x: saturation_pressure(x, T) - p, 0.01)[0]
    
    # sys.stdout = stdold



    # implement checks for input values (checks are from saturation_pressure())
    if T < 0:
        #raise ValueError('Temperature must be greater than 0 °C')
        print(f"Warning: T = {T} °C is below 0°C. Saturation concentration function not valid for T < 0°C")
    elif T < 20:
        if 1-x < 0.582:
            print(f"Warning: values are outside the valid range for T < 20 °C\nx = {x}\np = {p} Pa\nT = {T} °C")
    elif T < 60:
        if 1-x < 0.500:
            print(f"Warning: values are outside the valid range for 20 <= T < 60 °C\nx = {x}\np = {p} Pa\nT = {T} °C")
    elif T < 70:
        if 1-x < 0.353:
            print(f"Warning: values are outside the valid range for 60 <= T < 70 °C\nx = {x}\np = {p} Pa\nT = {T} °C")
    elif T < 150:
        if 1-x < 0.300:
            print(f"Warning: values are outside the valid range for 70 <= T < 150 °C\nx = {x}\np = {p} Pa\nT = {T} °C")
    elif T <= 200:
        if 1-x < 0.200:
            print(f"Warning: values are outside the valid range for 150 <= T <= 200 °C\nx = {x}\np = {p} Pa\nT = {T} °C")

    return x

# def precalculate_saturation_concentration_lookup_table(p_range, T_range):
#     """
#     Precomputes a lookup table for saturation concentration values over specified ranges.
    
#     Parameters:
#     p_range (iterable): Range of pressure values in Pa.
#     T_range (iterable): Range of temperature values in [°C].
    
#     Returns:
#     dict: Lookup table with keys as (p, T) tuples and values as saturation concentration in kg NaOH / kg solution.
#     """
#     lookup_table = {}
#     for p in p_range:
#         for T in T_range:
#             try:
#                 lookup_table[(round(p, 0), round(T, 1))] = saturation_concentration(p, T)
#             except:
#                 lookup_table[(round(p, 0), round(T, 1))] = None  # Handle invalid cases
#     return lookup_table


# # Generate the lookup table during import
# p_values = np.linspace(0, 50000, 5001)  # Pressure range from 0 to 2 MPa
# T_values = np.linspace(0, 200, 2001)  # Temperature range from 0 to 200 °C
# saturation_concentration_table = precalculate_saturation_concentration_lookup_table(p_values, T_values)

# def saturation_concentration_lookup(p, T, lookup_table=saturation_concentration_table, prevent_errors=False):

#     """
#     Efficiently calculates the saturation concentration of NaOH in water using a lookup table for precomputed values.
    
#     Parameters:
#     p (float): Pressure in Pa.
#     T (float): Temperature in [°C].
#     lookup_table (dict, optional): Precomputed lookup table with keys as (p, T) tuples.
#     prevent_errors (bool): If True, suppresses warnings and returns None for invalid inputs.
    
#     Returns:
#     float: Saturation concentration in kg NaOH / kg solution.
#     """
#     if lookup_table is not None:
#         key = (round(p, 0), round(T, 1))
#         if key in lookup_table:
#             return lookup_table[key]


def density(x, T):
    """
    DensityNaOH calculates the density of NaOH solution based on concentration and temperature.
    ---
    Parameters:
        x (array-like): Mole fraction of NaOH in the solution. [m_NaOH / (m_h2o+m_naoh)]
        T (array-like): Temperature in °C.
    ---
    Returns:
        rho (numpy array): Density in kg/m^3.
    ---
    Source: "Thermophysical Properties of Aqueous NaOH-H2O Solutions at High Concentrations,
    J. OIsson, A. Jernqvist, G. Aly, International Journal of Thermophysics Vol. 18. No. 3. 1997"
    ---
    Restrictions: 
     0 <= t <  10 ------ 0.0 < x <= 0.2
    10 <= t <  20 ------ 0.0 < x <= 0.3
    20 <= t <  60 ------ 0.0 < x <= 0.5
    60 <= t <  70 ------ 0.0 < x <= 0.6
    70 <= t < 150 ------ 0.0 < x <= 0.7
    150 <= t < 200 ------ 0.0 < x <= 0.8
    """

    # check validity
    if T < 0:
        #raise ValueError('Temperature must be greater than 0 °C')
        print(f"Warning: T = {T} °C is below 0°C. Density function not valid for T < 0°C")
    elif T < 10:
        if x > 0.2:
            print(f"Warning: values are outside the valid range for T < 10 °C\nx = {x}\nT = {T} °C")
    elif T < 20:
        if x > 0.3:
            print(f"Warning: values are outside the valid range for 10 <= T < 20 °C\nx = {x}\nT = {T} °C")
    elif T < 60:
        if x > 0.5:
            print(f"Warning: values are outside the valid range for 20 <= T < 60 °C\nx = {x}\nT = {T} °C")
    elif T < 70:
        if x > 0.6:
            print(f"Warning: values are outside the valid range for 60 <= T < 70 °C\nx = {x}\nT = {T} °C")
    elif T < 150:
        if x > 0.7:
            print(f"Warning: values are outside the valid range for 70 <= T < 150 °C\nx = {x}\nT = {T} °C")
    elif T <= 200:
        if x > 0.8:
            print(f"Warning: values are outside the valid range for 150 <= T <= 200 °C\nx = {x}\nT = {T} °C")
    elif T > 200:
        #raise ValueError('Temperature must be less than 200 °C')
        print(f"Warning: T = {T} °C is above 200°C. Density function not valid for T > 200°C")
    

    # transform x=x_NaOH to x=x_h2o
    if isinstance(x, (int, float)) or isinstance(T, (int, float)):
        x = 1 - x
        pass
    elif len(x) != len(T):
        raise ValueError('x and T must have the same length')
    else:
        x = 1 - np.array(x)
        T = np.array(T)

    # Coefficients
    k = np.array([5007.2279636, -25131.164248, 74107.692582, -104657.48684, 69821.773186, -18145.911810])
    l = np.array([-64.786269079, 525.34360564, -1608.4471903, 2350.9753235, -1660.9035108, 457.6437435])
    m = np.array([0.24436776978, -1.9737722344, 6.04601497138, -8.9090614947, 6.37146769397, -1.7816083111])

    x = np.array(x)
    
    b1 = k[0] + k[1] * (x ** (1/2)) + k[2] * x + k[3] * (x ** (3/2)) + k[4] * (x ** 2) + k[5] * (x ** (5/2))
    b2 = l[0] + l[1] * (x ** (1/2)) + l[2] * x + l[3] * (x ** (3/2)) + l[4] * (x ** 2) + l[5] * (x ** (5/2))
    b3 = m[0] + m[1] * (x ** (1/2)) + m[2] * x + m[3] * (x ** (3/2)) + m[4] * (x ** 2) + m[5] * (x ** (5/2))

    rho = b1 + b2 * T + b3 * (T ** 2)

    return float(rho)

def specific_heat_capacity(x, T):
    """
    Calculate the specific heat capacity of a NaOH solution as a function of temperature and concentration.
    The function is based on the following publication:
    Alexandrov 2004 - "The Equations for Thermophysical Properties of Aqueous Solutions of Sodium Hydroxide"
    ---
    x: float
        Concentration of NaOH in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Specific heat capacity of the solution [kJ/kgK]
    ---
    Author: Dorian Höffner
    Date: 2024-08-19
    ---
    Restrictions:
    0 <= T <= 275.85 °C
    0 <= x <= 0.16 kg/kg
    """

    # check validity
    if T < 0:
        raise ValueError('Temperature must be greater than 0 °C')
    elif T > 275.85:
        raise ValueError('Temperature must be less than 275.85 °C')
    elif x < 0:
        raise ValueError('Concentration must be greater than 0 kg/kg')
    elif x > 0.16:
        raise ValueError('Concentration must be less than 0.16 kg/kg. Correlation will provide incorrect results')
        #print(f"Warning: Values are outside the valid range for x > 0.16 kg/kg")

    # Constants
    a01=9.8555259e1
    a11=-3.4501318e2
    a21=4.8180532e2
    a31=-3.3440616e2
    a41=1.1516735e2
    a51=-1.5708814e1

    a02=-3.4357815e1
    a12=1.1674552e2
    a22=-1.5776854e2
    a32=1.0577045e2
    a42=-3.5099188e1
    a52=4.5935013
    
    a03=1.9791083
    a13=-5.3828966
    a23=5.5124212
    a33=-2.5430046
    a43=4.5943595e-1
    
    a04=-5.6191575e-2
    a14=8.5936388e-2
    a24=-1.6966718e-2
    a34=-1.4864492e-2
    
    a05=7.9944152e-3
    a15=-1.5444457e-2
    a25=7.5030322e-3


    cp_water = steamTable.CpL_t(T) # [kJ/kgK]

    # calculate molality (mol/kg) from concentration (kg/kg)
    M = 39.9971e-3          # [kg/mol]
    m = x / M               # [mol/kg]

    # calculate relative temperature
    t = (T + 273.15) / 273.15

    cp_diff = m    * (a01 + a11 * t + a21 * t**2 + a31 * t**3 + a41 * t**4 + a51 * t**5) + \
              m**2 * (a02 + a12 * t + a22 * t**2 + a32 * t**3 + a42 * t**4 + a52 * t**5) + \
              m**3 * (a03 + a13 * t + a23 * t**2 + a33 * t**3 + a43 * t**4) + \
              m**4 * (a04 + a14 * t + a24 * t**2 + a34 * t**3) + \
              m**5 * (a05 + a15 * t + a25 * t**2)

    if x <= 0.0001:
        cp = cp_water
    else:
        cp = cp_water - cp_diff #* 1e-3 # [kJ/kgK]

    return cp

def dynamic_viscosity(x, T, p):
    """
    Calculate the dynamic viscosity of a NaOH solution as a function of temperature and concentration.
    The function is based on the following publication:
    Alexandrov 2004 - "The Equations for Thermophysical Properties of Aqueous Solutions of Sodium Hydroxide"
    ---
    x: float
        Concentration of NaOH in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    p: float
        Pressure of the solution [Pa]
    ---
    returns: float
        Dynamic viscosity of the solution [Pa s]
    ---
    Restrictions:
    0 <= T <= 250.85 °C
    0 <= x <= 0.12 kg/kg
    0 <= p <= 7 MPa
    """

    # check validity
    if T < 0:
        raise ValueError('Temperature must be greater than 0 °C')
    elif T > 250.85:
        raise ValueError('Temperature must be less than 250.85 °C')
    elif x < 0:
        raise ValueError('Concentration must be greater than 0 kg/kg')
    elif p < 0:
        raise ValueError('Pressure must be greater than 0 Pa')
    elif p > 7e6:
        print(f"Warning: Values are outside the valid range for p > 15 MPa")
    elif x > 0.12:
        print(f"Warning: Values are outside the valid range for x > 0.12 kg/kg")


    # prepare inputs
    M = 39.9971e-3                      # [kg/mol]
    m = x / M                           # [mol/kg]
    T0 = 293.15                         # [K] (yes 293.15, not 273.15)
    t = T0 / (T + 273.15)               # [-]
    t1 = t - 1                          # [-]

    p = p / 1e6                         # [Pa]  -> [MPa]
    p_sw = steamTable.psat_t(T) / 1e1   # [bar] -> [MPa]
    
    # coeffs from cited publicatiosn
    c = np.array([np.nan, 5.17341030, 9.81838817, 2.83021985e1, 7.02071954e1, -9.92041252e2, -1.13267055e4, -5.10988292e4, -1.18863488e5, -1.41053273e5, -6.78490604e4])
    d = np.array([-3.18833435e-1, -1.07314454e1, -8.61347656e1, -6.50268842e2, -6.06767730e3, -4.07022741e4, -1.59650983e5, -3.53438962e5, -4.11357235e5, -1.96118714e5, np.nan])

    expression1 =        c[1] * t1 + c[2] * t1**2 + c[3] * t1**3 + c[4] * t1**4 + c[5] * t1**5 + c[6] * t1**6 + c[7] * t1**7 + c[8] * t1**8 + c[9] * t1**9 + c[10] * t1**10
    expression2 = d[0] + d[1] * t1 + d[2] * t1**2 + d[3] * t1**3 + d[4] * t1**4 + d[5] * t1**5 + d[6] * t1**6 + d[7] * t1**7 + d[8] * t1**8 + d[9] * t1**9
    my_water    = 1001.6 * (t1 + 1)**2 * np.exp(expression1) + (p-p_sw) * expression2 # [µPa s]

    # coeffs from actual publication
    b11 =  5.7070102e-1
    b21 =  4.9395013e-1
    b31 = -2.0417183
    b41 =  1.1654862

    b12 = -2.9922166e-1
    b22 =  3.7957782e-1
    b32 = -7.423751e-2

    b13 =  4.9815412e-2
    b23 = -4.8332728e-2

    expression3 = m    * (b11 * t + b21 * t**2 + b31 * t**3 + b41 * t**4) + \
                  m**2 * (b12 * t + b22 * t**2 + b32 * t**3) + \
                  m**3 * (b13 * t + b23 * t**2)
    
    my_rel = np.exp(expression3)

    my = my_water * np.exp(expression3) # [µPa s]

    return float(my * 1e-6) # [Pa s]

def thermal_conductivity(x, T, p):
    """
    Calculate the thermal conductivity of a NaOH solution as a function of temperature and concentration.
    The function is based on the following publication:
    Alexandrov 2004 - "The Equations for Thermophysical Properties of Aqueous Solutions of Sodium Hydroxide"
    ---
    x: float
        Concentration of NaOH in the solution [kg/kg]
    T: float
        Temperature of the solution [°C]
    ---
    returns: float
        Thermal conductivity of the solution [W/mK]
    ---
    Restrictions:
        0 <= T <= 132.85 °C
        0 <= x <= 0.2 kg/kg
        0 <= p <= 15 MPa
    """

    # check validity
    if T < 0:
        raise ValueError('Temperature must be greater than 0 °C')
    elif T > 132.85:
        raise ValueError('Temperature must be less than 132.85 °C')
    elif x < 0:
        raise ValueError('Concentration must be greater than 0 kg/kg')
    elif p < 0:
        raise ValueError('Pressure must be greater than 0 Pa')
    elif p > 15e6:
        raise ValueError('Pressure must be less than 15 MPa')
    elif x > 0.2:
        print(f"Warning: Values are outside the valid range for x > 0.2 kg/kg")

    # prepare inputs for water calculation
    p_sw = steamTable.psat_t(T) / 1e1   # [bar] -> [MPa]
    p = p / 1e6                         # [Pa]  -> [MPa]
    T02 = 273.15 + 20                   # [K]
    t = T02 / (T + 273.15) - 1          # [-]

    # coeffs from cited publication (for water)
    g = np.array([5.99454842e-1, -4.82554378e-1, -4.31229616e-1, -8.62555022e-1, -3.80050418e-1, 4.85828450e1, 3.35400696e2, 1.08007806e3, 1.67727081e3, 1.04225629e3])
    q = np.array([5.31492446e-4, 3.46658996e-4, 1.23050434e-2, 1.27873471e-1, -7.40820487e-1, -1.93072528e1, -1.22835056e2, -3.66150909e2, -5.31321978e2, -3.03153185e2])

    expression1 = g[0] + g[1] * t**1 + g[2] * t**2 + g[3] * t**3 + g[4] * t**4 + g[5] * t**5 + g[6] * t**6 + g[7] * t**7 + g[8] * t**8 + g[9] * t**9
    expression2 = q[0] + q[1] * t**1 + q[2] * t**2 + q[3] * t**3 + q[4] * t**4 + q[5] * t**5 + q[6] * t**6 + q[7] * t**7 + q[8] * t**8 + q[9] * t**9
    lambda_water = expression1 + (p-p_sw) * expression2  # [kW/mK]

    
    # prepare inputs for NaOH calculation
    T0 = 403.0 # [K]
    t1 = (T+273.15) / T0
    M = 39.9971e-3          # [kg/mol]
    m = x / M               # [mol/kg]
    
    # coeffs from actual publication (for NaOH)
    e01 = 3.2900544e-1
    e11 = -1.1048583
    e21 = 1.2503803
    e31 = -4.4228179e-1

    e02 = -2.1990820e-2
    e12 = 5.9100989e-2
    e22 = -4.4407173e-2
    
    e03 = 1.5069324e-3
    e13 = -4.3273501e-3
    e23 = 3.3763248e-3

    expression3 = m    * (e01 + e11 * t1 + e21 * t1**2 + e31 * t1**3) + \
                  m**2 * (e02 + e12 * t1 + e22 * t1**2) + \
                  m**3 * (e03 + e13 * t1 + e23 * t1**2)
    
    lambda_NaOH = lambda_water + expression3 # [kW/mK]

    return float(lambda_NaOH * 1e3) # [W/mK]

def dhdx(x,T):
    """
    Calculate the partial derivative of enthalpy with respect to NaOH mass fraction.
    Parameters
    ----------
    x : float
        NaOH concentration in kg NaOH per kg solution.
    T : float
        Temperature in degrees Celsius.
    Returns
    -------
    float
        Partial derivative of enthalpy with respect to NaOH mass fraction [kJ/kg].
    Notes
    -----
    This function uses a central difference method for numerical differentiation.
    """

    delta = 0.000001
    x1 = x - delta
    x2 = x + delta

    dhdx = (enthalpy(x2,T) - enthalpy(x1,T)) / (2*delta)
    return dhdx

def dhdT(x,T):
    """
    Calculate the partial derivative of enthalpy with respect to temperature for a NaOH solution.
    Parameters
    ----------
    x : float
        NaOH concentration in the solution [kg NaOH/kg solution].
    T : float
        Temperature [°C].
    Returns
    -------
    float
        Partial derivative of enthalpy with respect to temperature [kJ/kgK].
    Notes
    -----
    Uses central finite difference approximation with a small delta for numerical differentiation.
    Requires the `enthalpy(x, T)` function to be defined elsewhere.
    """

    delta = 0.0001
    T1 = T - delta
    T2 = T + delta

    dhdT = (enthalpy(x,T1) - enthalpy(x,T2)) / (2*delta)
    return dhdT

def hxDiagram(editablePlot=False):
    """
    Plots the enthalpy-concentration diagram for NaOH-H2O solutions.
    Author: Dorian Höffner 2024-11-22

    Parameters:
        editablePlot (bool): If True, the plot will be editable with matplotlib.

    Returns:
        None
    """

    # # suppress all print statements
    # devnull = open(os.devnull, 'w')
    # stout_old = sys.stdout
    # sys.stdout = devnull

    with contextlib.redirect_stdout(io.StringIO()):  # Suppress stdout    

        plt.figure(dpi=300)
        T_array = np.array(range(0, 101, 2))
        x_array = np.linspace(0, 0.75, 100)

        for T in T_array:
            h_array = np.array([enthalpy(x, T, prevent_errors=True) for x in x_array])
            if T%20 == 0:
                plt.plot(x_array, h_array, color='black', alpha=0.7, lw=0.8, zorder=0)
                label_posx = x_array[23] + 0.005
                plt.text(label_posx, h_array[23]+10, f'{T} °C', fontsize=8, color='black')
            else:
                plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2, zorder=0)

        plt.xlabel('Concentration $x=[kg_{NaOH}/kg_{solution}]$')
        plt.ylabel('Enthalpy $h=[kJ/kg]$')
        plt.xlim(0, 0.75)

        # plot crystallization curve
        cryst_data = crystallization_curve(return_data=True)
        cryst_T = np.array([T for x, T in cryst_data]).flatten()
        cryst_h = np.array([enthalpy(x, T) if T > 0 else np.nan for x, T in cryst_data]).flatten()
        plt.plot([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], color='black', linestyle="--", lw=0.5, label='Crystallization Curve')
        plt.fill_between([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], 0, color='white', zorder=0)    


        if not editablePlot:
            plt.show()




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
    
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

    # # suppress all print statements
    # devnull = open(os.devnull, 'w')
    # stout_old = sys.stdout
    # sys.stdout = devnull

    with contextlib.redirect_stdout(io.StringIO()):  # Suppress stdout
        
        # get data to plot crystallization curve
        cryst_data = crystallization_curve(return_data=True)
        cryst_T = np.array([T for x, T in cryst_data])
        #cryst_p = np.array([saturation_pressure(x, T) for x, T in cryst_data])


        # Calculate crystallization curve pressure safely
        cryst_p = []
        for x, T in cryst_data:
            try:
                p = saturation_pressure(x, T, prevent_errors=True)
                if p is None:
                    raise ValueError("Invalid pressure value")
            except:
                p = np.nan  # Replace invalid values with NaN
            cryst_p.append(p)

        cryst_p = np.array(cryst_p)
        
        # Temperature range
        temperaturesC = np.arange(0, 110, 1)
        temperaturesK = temperaturesC + 273.15
        concentrations = np.arange(0.1, 0.751, 0.01)
        cr_temperaturesC = cryst_T
        cr_temperaturesK = cryst_T + 273.15
        
        # Prepare the plot
        plt.figure(dpi=300)
        
        # these temperatures are used for the x-axis
        plotTemperatures = np.arange(0, 101, 10) + 273.15
        
        # Calculate water vapor pressure using XSteam
        waterPressure = [steamTable.psat_t(T - 273.15) * 1e5 for T in temperaturesK]  # convert bar to Pa
        
        # SUPPRESS WARNINGS


        # Plot the data
        for x in concentrations:

            # Calculate saturation pressure for each temperature at the given concentration
            p = []
            for T in temperaturesC:
                p.append(saturation_pressure(x, T))           

        
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
            if show_percentages and int(np.round(x * 100)) % 10 == 0:
                label_pos = temp_plot[-1] + 1e-5 if invT else temp_plot[-1]+2
                plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8, color='black')
            
        # add description for concentration: % = "kg NaOH / kg solution"
        plt.text(label_pos, p[-1] * 0.85, r'$\left[\frac{\mathrm{kg_{NaOH}}}{\mathrm{kg_{Solution}}}\right]$', fontsize=11, color='black')

        # Plotting water line
        if log:
            plt.semilogy(temp_plot, waterPressure, color="grey", linestyle='--',
                        label='Pure Water')
        else:
            plt.plot(temp_plot, waterPressure, color="grey", linestyle='--', label='Pure Water')

        # # Plotting crystallization curve
        # if log:
        #     plt.semilogy(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
        #                 lw=1.0, label='Crystallization Curve', zorder=101)
        #     # fill area between crystallization curve and the minimum positive value (e.g., 1)
        #     plt.fill_between(temp_plot_cryst, cryst_p, 1, where=(cryst_p > 1), color='white', zorder=100)
        # else:
        #     plt.plot(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
        #             lw=1.0, label='Crystallization Curve', zorder=101)
        #     # fill area between crystallization curve and a fixed value (e.g., 50)
        #     plt.fill_between(temp_plot_cryst, cryst_p, 1, where=(cryst_p > 1),  color='white', zorder=100)

        if log:
            plt.semilogy(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
                        lw=1.0, label='Crystallization Curve', zorder=101)
            plt.fill_between(temp_plot_cryst, np.nan_to_num(cryst_p, nan=1), 1, 
                            where=(np.nan_to_num(cryst_p, nan=0) > 1), color='white', zorder=100)
        else:
            plt.plot(temp_plot_cryst, cryst_p, color="gray", linestyle='-',
                    lw=1.0, label='Crystallization Curve', zorder=101)
            plt.fill_between(temp_plot_cryst, np.nan_to_num(cryst_p, nan=1), 1, 
                            where=(np.nan_to_num(cryst_p, nan=0) > 1), color='white', zorder=100)
            

        

        # Setting axis limits
        if invT:
            plt.xlim(-1/temperaturesK[0], -1/temperaturesK[-1])
        else:
            plt.xlim(temperaturesK[0], temperaturesK[-1])
        
        if log:
            plt.ylim(50, 2e5)
        else:
            plt.ylim(0, max(waterPressure) * 1.1)  # Adjust as needed to make sure all data is visible


        plt.legend()
        
        if not editablePlot:
            plt.show()

        # # reset print statements
        # sys.stdout = stout_old
        

def solubility_temperature(x):
    """
    Return the crystallization temperature of NaOH solution in water.
    Source: Wang et al. 2008 (data was extracted from the plot)
    ---
    Parameters:
        x (float or np.array): Concentration of NaOH in kg/kg
    ---
    Returns:
        float or np.array: Crystallization temperature in °C
    ---
    Restrictions:
        0 <= x <= 0.787 kg/kg
    """

    # check validity
    if x < 0:
        raise ValueError('Concentration must be greater than 0 kg/kg')
    elif x > 0.787:
        raise ValueError('Concentration x = {x} must be less than 0.787 kg/kg')

    # cryst_data = x, T
    cryst_data   = [[0.2707275803722504,0.0],
                    [2.165820642978004,-1.3333333333333357],
                    [4.060913705583756,-2.933333333333337],
                    [5.685279187817259,-4.533333333333331],
                    [7.580372250423012,-6.666666666666664],
                    [9.34010152284264,-8.533333333333335],
                    [10.96446700507614,-10.933333333333334],
                    [12.588832487309643,-13.333333333333332],
                    [14.077834179357025,-16.0],
                    [15.5668358714044,-18.933333333333334],
                    [16.785109983079526,-21.866666666666667],
                    [17.868020304568528,-24.8],
                    [18.68020304568528,-28.0],
                    [20.169204737732656,-26.4],
                    [21.658206429780037,-24.8],
                    [22.74111675126904,-22.4],
                    [23.688663282571916,-20.53333333333333],
                    [24.771573604060915,-18.666666666666668],
                    [25.583756345177665,-14.133333333333333],
                    [26.395939086294412,-9.6],
                    [27.34348561759729,-5.333333333333336],
                    [28.56175972927242,-1.3333333333333357],
                    [29.780033840947542,1.8666666666666671],
                    [31.810490693739425,5.066666666666663],
                    [32.89340101522843,7.733333333333334],
                    [34.247038917089675,10.133333333333333],
                    [35.736040609137056,12.533333333333331],
                    [37.36040609137056,14.666666666666664],
                    [38.984771573604064,15.733333333333334],
                    [40.33840947546531,15.466666666666669],
                    [41.96277495769881,13.866666666666667],
                    [43.45177664974619,11.733333333333334],
                    [44.67005076142132,8.799999999999997],
                    [45.6175972927242,6.133333333333333],
                    [47.10659898477157,8.0],
                    [48.73096446700507,9.866666666666667],
                    [50.35532994923858,11.733333333333334],
                    [51.43824027072758,12.533333333333331],
                    [52.25042301184433,17.333333333333336],
                    [53.06260575296108,22.133333333333333],
                    [53.87478849407783,26.66666666666667],
                    [54.95769881556684,32.0],
                    [55.90524534686971,36.8],
                    [56.98815566835872,41.06666666666666],
                    [58.34179357021996,45.86666666666666],
                    [59.96615905245347,50.400000000000006],
                    [61.86125211505922,54.93333333333334],
                    [63.756345177664976,58.66666666666667],
                    [65.65143824027072,61.33333333333333],
                    [68.08798646362098,63.46666666666667],
                    [70.25380710659898,64.0],
                    [72.14890016920474,63.73333333333333],
                    [73.36717428087987,62.93333333333334],
                    [73.63790186125212,68.53333333333333],
                    [74.04399323181049,76.26666666666667],
                    [74.45008460236886,83.2],
                    [74.72081218274111,89.06666666666666],
                    [75.26226734348562,96.53333333333332],
                    [75.80372250423012,103.73333333333332],
                    [76.34517766497461,110.93333333333334],
                    [77.02199661590524,118.4],
                    [77.834179357022,126.13333333333333],
                    [78.78172588832487,133.33333333333334]]

    # create interpolation function
    x = x * 100  # convert to percentage
    t_solubility = np.interp(x, [x for x, T in cryst_data], [T for x, T in cryst_data])

    return t_solubility # [°C]


def crystallization_curve(return_data=False):
    """
    Plots or returns the crystallization curve for NaOH concentration vs. temperature, based on data extracted from Wang (2008).
    This function visualizes or provides the crystallization boundary for sodium hydroxide (NaOH) solutions, as digitized from the publication:
    "Wang 2008 - Cellulose Fiber Dissolution in Sodium Hydroxide Solution at Low Temperature: Dissolution Kinetics and Solubility Improvement".
        
    Parameters:
        return_data (bool, optional): 
            If True, returns the crystallization curve data as a list of [NaOH_concentration, temperature] pairs, 
            where NaOH concentration is given as a fraction (0–0.78). 
            If False (default), displays a plot of the crystallization curve.
        None or list of [float, float]:
            If return_data is True, returns a list of [NaOH_concentration (fraction), temperature (°C)] data points.
            If return_data is False, displays a matplotlib plot and returns None.
    Notes:
        - NaOH concentration is given in percent (%) for plotting, and as a fraction (0–1) when returned as data.
        - The data was digitized from a published plot and may contain minor inaccuracies.
        - The plot includes grid lines, axis labels, and a legend for clarity.
    Author:
        Dorian Höffner
    Date:
        2024-09-17
    """
    
    # cryst_data = x, T
    cryst_data   = [[0.2707275803722504,0.0],
                    [2.165820642978004,-1.3333333333333357],
                    [4.060913705583756,-2.933333333333337],
                    [5.685279187817259,-4.533333333333331],
                    [7.580372250423012,-6.666666666666664],
                    [9.34010152284264,-8.533333333333335],
                    [10.96446700507614,-10.933333333333334],
                    [12.588832487309643,-13.333333333333332],
                    [14.077834179357025,-16.0],
                    [15.5668358714044,-18.933333333333334],
                    [16.785109983079526,-21.866666666666667],
                    [17.868020304568528,-24.8],
                    [18.68020304568528,-28.0],
                    [20.169204737732656,-26.4],
                    [21.658206429780037,-24.8],
                    [22.74111675126904,-22.4],
                    [23.688663282571916,-20.53333333333333],
                    [24.771573604060915,-18.666666666666668],
                    [25.583756345177665,-14.133333333333333],
                    [26.395939086294412,-9.6],
                    [27.34348561759729,-5.333333333333336],
                    [28.56175972927242,-1.3333333333333357],
                    [29.780033840947542,1.8666666666666671],
                    [31.810490693739425,5.066666666666663],
                    [32.89340101522843,7.733333333333334],
                    [34.247038917089675,10.133333333333333],
                    [35.736040609137056,12.533333333333331],
                    [37.36040609137056,14.666666666666664],
                    [38.984771573604064,15.733333333333334],
                    [40.33840947546531,15.466666666666669],
                    [41.96277495769881,13.866666666666667],
                    [43.45177664974619,11.733333333333334],
                    [44.67005076142132,8.799999999999997],
                    [45.6175972927242,6.133333333333333],
                    [47.10659898477157,8.0],
                    [48.73096446700507,9.866666666666667],
                    [50.35532994923858,11.733333333333334],
                    [51.43824027072758,12.533333333333331],
                    [52.25042301184433,17.333333333333336],
                    [53.06260575296108,22.133333333333333],
                    [53.87478849407783,26.66666666666667],
                    [54.95769881556684,32.0],
                    [55.90524534686971,36.8],
                    [56.98815566835872,41.06666666666666],
                    [58.34179357021996,45.86666666666666],
                    [59.96615905245347,50.400000000000006],
                    [61.86125211505922,54.93333333333334],
                    [63.756345177664976,58.66666666666667],
                    [65.65143824027072,61.33333333333333],
                    [68.08798646362098,63.46666666666667],
                    [70.25380710659898,64.0],
                    [72.14890016920474,63.73333333333333],
                    [73.36717428087987,62.93333333333334],
                    [73.63790186125212,68.53333333333333],
                    [74.04399323181049,76.26666666666667],
                    [74.45008460236886,83.2],
                    [74.72081218274111,89.06666666666666],
                    [75.26226734348562,96.53333333333332],
                    [75.80372250423012,103.73333333333332],
                    [76.34517766497461,110.93333333333334],
                    [77.02199661590524,118.4],
                    [77.834179357022,126.13333333333333],
                    [78.78172588832487,133.33333333333334]]
        
    if return_data:
        cryst_data = [[x/100, T] for x, T in cryst_data]
        return cryst_data
    
    else:
        # prepare plot
        plt.figure(figsize=(6,4), dpi=300)
        plt.plot([x[0] for x in cryst_data],
                    [x[1] for x in cryst_data],
                    label='Crystallization Curve',
                    color="black")
        plt.xlabel('NaOH Concentration [%]')
        plt.ylabel('Temperature [°C]')
        plt.xlim(0, 78.5)
        plt.ylim(-35, 140)
        # make beautiful grid
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
        # add legend (top left)
        plt.legend(loc='upper left')
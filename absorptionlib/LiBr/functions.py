import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


from CoolProp.CoolProp import PropsSI
from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

import os
import sys

def documentation():
    print("""
This module contains functions for calculating properties of LiBr-H2O solutions:

| Function Name              | Description                                                                                   |
|---------------------------|------------------------------------------------------------------------------------------------|
| saturation_temperature    | Calculate the boiling point temperature of an aqueous Lithium Bromide solution.                |
| enthalpy_PK               | Calculate h'(T) based on the given equation.                                                   |
| enthalpy                  | Calculate the enthalpy of an H2O-LiBr solution at a given temperature and concentration.       |
| saturation_pressure       | Calculate the equilibrium pressure of an H2O-LiBr solution.                                    |
| saturation_concentration  | Calculates the saturation concentration of LiBr in water based on the temperature and pressure.|
| density                   | Calculate the density of a water-LiBr solution.                                                |
| hxDiagram                 | Plots the pressure-temperature diagram for LiBr-H2O solutions.                                 |
| pTDiagram                 | Plots the pressure-temperature diagram for LiBr-H2O solutions.                                 |
| solubility_temperature    | Calculate the crystallization temperature of LiBr solution in water.                           |

For more information use the following function: LiBr.explain("function_name")

For example: LiBr.explain("enthalpy")
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


# LiBr property functions
# DONE
def saturation_temperature(x, p):
    """
    Calculate the boiling point temperature of an aqueous Lithium Bromide solution.
    ---
    Parameters:
        p (float): Pressure in Pa.
        x (float): Concentration in kg LiBr/kg solution, i.e., 0.0 to 0.75.
    ---
    Returns:
        float: Boiling point temperature in °C.
    """

    # check validity of concentration
    if x > 0.75:
        raise ValueError("Concentration out of valid range (0.0 to 0.75).")

    # use saturation_pressure and fsolve to find the temperature
    def f(T):
        return saturation_pressure(x, T) - p
    
    # initial guess for fsolve
    T_guess = 50 # [°C]

    # find the saturation temperature
    T = fsolve(f, T_guess)[0]

    return T # [°C]



def enthalpy_PK(x, T):
    """
    Calculates h'(T) based on the given equation.

    Parameters:
    T (float): Temperature
    h_c (float): Reference heat transfer coefficient
    T_c (float): Critical temperature
    alpha (list): List of alpha_i coefficients (length 4)
    beta (list): List of beta_i coefficients (length 4)

    Returns:
    float: The value of h'(T)
    """

    beta = [1/3, 2/3, 5/6, 21/6]
    alpha = [-4.37196e-1, 3.03440e-1, -1.29582, -1.76410e-1]
    Tc = 647.096    # K
    hc = 548.5      # J/mol

    if len(alpha) != 4 or len(beta) != 4:
        raise ValueError("alpha and beta must each contain exactly 4 elements.")

    summation = sum(alpha[i] * ((1 - T / Tc) ** beta[i]) for i in range(4))
    h = hc * (1 + summation)
    
    return h # J/mol




# DONE (40%-75%)
def enthalpy(konz, tempC):
    """
    Calculate the enthalpy of an H2O-LiBr solution at a given temperature and concentration.
    ---
    Parameters:
        konz (float): Salt mass fraction in kg LiBr/kg solution (0.4 <= konz <= 0.75).
        tempC (float): Temperature in degrees Celsius (°C).
    ---
    Returns:
        float: Enthalpy in kJ/kg. Returns -1 if the temperature is out of range (0°C to 190°C).
            Returns -2 if the concentration is out of range (0.4 to 0.75).
    ---
    Notes:
        This function calculates the enthalpy based on the method described in:
        Feuerecker, Günther: Entropieanalyse für Wärmepumpensysteme: Methoden und Stoffdaten.
        Institut für Festkörperphysik und Technische Physik der Technischen Universität München,
        München 1994, S. 131 Gleichung (9.13).
    ---
    History:
    - Originally programmed for MATLAB by Jan Albers, TU Berlin, jan.albers@tu-berlin.de, on 2003-12-03.
    - standardized API, added ValueErrors, added crystallization check by Dorian Höffner 11/2024
    """

    temp = tempC + 273.15  # convert to Kelvin

    if temp < 273.15 or temp > 190 + 273.15:
        raise ValueError(f"Temperature {tempC}°C out of valid range (0°C to 190°C).")
    if konz > 0.75 + 10 ** (-9):
        raise ValueError(f"Concentration x = {konz} out of valid range (> 0.75).")
    if konz < 0.4 - 10 ** (-9):
        print(f"Warning: Concentration x = {konz} is below 40% in enthalpy calculation. Results were interpolated between pure water and the 40% solution. Results are not validated by experiments.")


    # G.F. rechnet in # nicht in kg/kg, daher Konvertierung
    konz = konz * 100
    
    def h(konz,temp):

        # Coefficients for the enthalpy calculation
        coefhli = [[-954.8,    47.7739,    -1.59235,    2.09422e-2, -7.689e-5],
                   [-0.3293,    4.076e-2,  -1.36e-5,   -7.1366e-6],
                   [7.4285e-3, -1.5144e-4,  1.3555e-6],
                   [-2.269e-6]]

        a = 0
        for k in range(len(coefhli[0])):
            a += konz ** k * coefhli[0][k]

        b = 0
        for k in range(len(coefhli[1])):
            b += konz ** k * coefhli[1][k]
        c = 0
        for k in range(len(coefhli[2])):
            c += konz ** k * coefhli[2][k]
            
        h = a + temp * b + temp ** 2 * c + coefhli[3][0] * temp ** 3
        return h

    if konz >= 40:
        h_sol = h(konz,temp)

    #### OLD VERSION
    # if konz < 40:
    #     # get dh/dx at 40% concentration
    #     h40  = h(40,temp)
    #     h39  = h(39,temp)
    #     dhdx = (h40-h39)/1
    #     # calculate enthalpy at given concentration (linear extrapolation)
    #     h = h40 + (konz-40) * dhdx

    if konz < 40:
        # get enthalpy 40% and given temperature
        h40 = h(40, temp)
        
        # get water enthalpy given temperature
        h_water = steamTable.hL_t(tempC+0.01)

        # interpolate between h40 and h_water
        h_sol = h_water + konz/-40 * (h_water - h40)

    return h_sol


# TODO
# def differential_enthalpy_AD(x, T):
#     """
#     Calculates the differential enthalpy [kJ/kgH2O] of a CaCl2 solution based on concentration and temperature.
#     How much additional energy (in comparison to pure vaporization enthalpy) is needed to vaporize a certain amount of water in a CaCl2 solution.
    
#     Parameters:
#         x (float): Concentration of CaCl2 in the solution [kg/kg].
#         T (float): Temperature of the solution [°C].

#     Returns:
#         float: Differential enthalpy [kJ/kgH2O].

#     Author: Dorian Höffner 11/2024
#     Source: 
#     """

#     return

# DONE
def saturation_pressure(x, T):
    """
    Calculate the equilibrium pressure of an H2O-LiBr solution.
    Parameters:
        x (float): Mass fraction of LiBr in the solution (kg LiBr/kg solution), should be between 0.00 and 0.75.
        T (float): Temperature in Kelvin, should be between 273.15 K and 500 K.
    Returns:
        float: Equilibrium pressure in bar. Returns -1 if temperature is out of valid range, 
            and -2 if mass fraction is out of valid range.
    References:
        J. Patek and J. Klomfar, "A computationally effective formulation of the thermodynamic properties 
        of LiBr-H2O solutions from 273 to 500 K over full composition range," to be published in IJR.
    History:
        - Programmed by Jan Albers on DD.MM.2010
        - Validity limits added on 02.08.2012 by J.A.
        - standardized API and added ValueError by Dorian Höffner 2024-11-05
    Restrictions:
        - The function is only valid for temperatures between 273.15 K and 500 K.
        - The function is only valid for mass fractions between 0.00 and 0.75.
    """

    def psatW_local(temp):
        """
        Calculates water vapor pressure based on matlab package h2o_IAPWS97.
        Author: Dorian Höffner 2024-11-21
        """
        
        # Coefficients
        nreg4_9  = -0.23855557567849
        nreg4_10 = 650.17534844798
        nreg4_1  = 1167.0521452767
        nreg4_2  = -724213.16703206
        nreg4_3  = -17.073846940092
        nreg4_4  = 12020.82470247
        nreg4_5  = -3232555.0322333
        nreg4_6  = 14.91510861353
        nreg4_7  = -4823.2657361591
        nreg4_8  = 405113.40542057

        dl = temp + nreg4_9 / (temp - nreg4_10)
        Aco = dl ** 2 + nreg4_1 * dl + nreg4_2
        Bco = nreg4_3 * dl ** 2 + nreg4_4 * dl + nreg4_5
        cco = nreg4_6 * dl ** 2 + nreg4_7 * dl + nreg4_8
        pSatW = (2 * cco / (-Bco + (Bco ** 2 - 4 * Aco * cco) ** 0.5)) ** 4 * 10
    
        return pSatW
    
    T = T + 273.15  # convert to Kelvin

    # get necessary parameters from Params class (Patek/Klomfar 2006)
    params  = Params_PK()
    M_LiBr  = params.M_LiBr
    M_W     = params.M_W
    TCritW  = params.TCritW
    aTab4   = params.aTab4
    mTab4   = params.mTab4
    nTab4   = params.nTab4
    tTab4   = params.tTab4

    # Validity Check
    if T < 273.15 or T > 500:
        raise ValueError(f"Temperature (T={T}) out of valid range (273.15 K to 500 K).")
    if x < 0.00 or x > 0.75:
        raise ValueError(f"Mass fraction (x={x}) out of valid range (0.00 to 0.75).")

    # Calculation
    tsum = 0
    
    x_mol = (x/M_LiBr)/(x/M_LiBr + (1-x)/M_W)
    
    for k in range(len(aTab4)):
        tsum += aTab4[k] * x_mol ** mTab4[k] * (0.40 - x_mol) ** nTab4[k] * (T / TCritW) ** tTab4[k]

    # return psat * 1e5  # convert bar to Pa
    psat = psatW_local(T - tsum)  # in bar

    return psat * 1e5  # convert bar to Pa



# DONE
def saturation_concentration(p, T):
    """
    Calculates the saturation concentration of LiBr in water based on the temperature and pressure.
    Source: Feuerecker et al. 1994
    
    Author: Dorian Höffner
    Date: 2024-09-17
    ---
    Parameters:
        p (float): Pressure in Pa.
        T (float): Temperature in °C.
    ---
    Returns:
        float: Saturation concentration in kg LiBr / kg solution.
    ---
    Restrictions:
        - The function is only valid for temperatures between 273.15 K and 500 K.
    """

    if T < 0 or T > 227.85:
        raise ValueError("Temperature out of valid range (0°C to 227.85°C).")

    # Calculate the saturation concentration
    x = 0.4
    while saturation_pressure(x, T) - p > 1:   # 1 Pa tolerance
        x += 0.001

    # retry with lower concentration initil guess
    x = 0.001
    while saturation_pressure(x, T) - p > 1:   # 1 Pa tolerance
        x += 0.001
            

    if x > 0.75:
        print(f"Warning: Saturation concentration is above 75% at {T}°C and {p} Pa. This is outside the publication's validity range.")

    return x

# DONE
def density(konz, tempC):
    """
    Calculate the density of a water-LiBr solution.
    This function calculates the density of a water-LiBr (Lithium Bromide) solution 
    based on the given mass fraction and temperature using the correlations provided 
    by Günther Feuerecker and corrections from private communication between F. Ziegler 
    and G. Feuerecker.
    ---
    Parameters:
        konz (float): Mass fraction of LiBr in the solution (kg_LiBr/kg_Sol). Valid range is 0.0 to 0.75.
        tempC (float): Temperature in degrees Celsius. Valid range is 0°C to 190°C.
    ---
    Returns:
        float: Density of the water-LiBr solution in kg/m³.
    ---
    Raises:
        ValueError: If the temperature is out of the valid range (0°C to 190°C).
        ValueError: If the mass fraction is out of the valid range (0.0 to 0.75).
    ---
    Notes:
    - If the mass fraction is below 0.4, the results are extrapolated.
    - The density of pure water is calculated using coefficients from Landolt-Börnstein.
    - The function includes corrections from private communication between F. Ziegler and G. Feuerecker (1995).
    References:
    - Feuerecker, Günther: Entropieanalyse für Wärmepumpensysteme: Methoden und Stoffdaten. 
      Institut für Festkörperphysik und Technische Physik der Technischen Universität München, 
      München 1994.
    - Landolt-Börnstein: II. Band Eigenschaften der Materie in ihren Aggregatzuständen.
      1. Teil Mechanisch-thermische Zustandsgrößen. 6. Auflage Berlin 1971.
    """
    # Historie
    # J.A. Nov. 2003 programmiert
    # J.A. Feb. 2004 Begrenzung der Wasserdichte (H2O) auf Siededichte eingefügt
    # J.A. Apr. 2012 Correction from private communication between F. Ziegler and G. Feuerecker 1995 included
    #                Implementation of pure water coefficients from Landolt-Börnstein (as used by Peter Müller)
    # Dorian Höffner: standardized API, added ValueError, added extrapolation for mass fraction below 40%
    #
    # Conversion from input temperature in Kelvin to Celsius used in the correlations of G.F.
    temp = tempC + 273.15

    if tempC < 0 or tempC > 190:
        raise ValueError(f"Temperature {tempC}°C out of valid range (0°C to 190°C).")
    if konz < 0.4 - 10 ** (-9):
        print("Warning: Results for density were extrapolated for mass fractions from 0.0 to 0.4.")
    if konz > 0.75 + 10 ** (-9):
        raise ValueError("Mass fraction out of valid range (0.4 to 0.75).")
    if konz < 0.0:
        raise ValueError("Mass fraction out of valid range (0.0 to 0.75).")

    # New from Apr 2012 on: The same water density is used here as was done by Peter Müller
    # to derive the coefficients for the density of H_2O/LiBr-solution.
    #
    # Peter Müller used the coeffients of
    # Landolt-Börnstein: II. Band Eigenschaften der Materie in ihren Aggregatzuständen.
    # 1. Teil Mechanisch-thermische Zustandsgrößen. 6. Auflage Berlin 1971
    a_1 = 3.9863  # K
    a_2 = 508929.2  # K**2
    a_3 = 288.9414  # K
    a_4 = 68.12963  # K
    a_5 = 0.999973  # g/cm**3

    dens_H2O = (1 - (tempC - a_1) ** 2 / a_2 * (tempC + a_3) / (tempC + a_4)) * a_5
    # todo use denseH2O  function instead?

    # published equation of G.F.
    # densH2OLiBr_GF = dens_H2O  / 2  * (exp(1.2  *  konz) + exp((0.842+1.6414e-3 * tempC) * konz))

    # including the correction from private communication between F. Ziegler and G. Feuerecker 1995
    # (the power of 2 was missing at last konz value)
    return (dens_H2O / 2.0 * (np.exp(1.2 * konz) + np.exp((0.842 + 1.6414e-3 * tempC) * konz ** 2))) * 1000

# # DONE
# def density OLD NOT WORKING PROPERLY(x, T):
#     """
#     Calculate the density of H2O-LiBr solution.
#     This function computes the density of a H2O-LiBr solution based on the 
#     temperature and mass fraction of LiBr in the solution.
#     ---
#     Parameters:
#         x (float): Mass fraction of LiBr in the solution (kg_LiBr/kg_Solution), e.g., 0 to 0.75.
#         T (float): Temperature in °C.
#     ---
#     Returns:
#         float: Density of the H2O-LiBr solution in kg/m³.
#     ---
#     References:
#         Chua et al. 1999 "Improved thermodynamic property Æelds of LiBr±H2O solution"
#     ---
#     Author: Dorian Höffner 2024-11-05
#     """

#     T = T + 273.15  # convert to Kelvin
    
#     G = [[9-99100e2,      7.74931e0, 5.36509e-3, 1.34988e-3, -3.08671e-6],
#          [-2.39865e-2,  -1.28346e-2, 2.07232e-4, -9.08213e-6, 9.94788e-8],
#          [-3.90453e-3,  -5.55855e-5, 1.09879e-5, -2.39834e-7, 1.53514e-9]]
#     G = np.array(G)

#     delta_rho = np.sum([x**(i) * (G[0,i] + T * G[1,i] + T**2 * G[2,i]) for i in range(5)])
#     rho = delta_rho + steamTable.rhoL_t(T-273.15)  # kg/m³

#     return rho

    



# # TODO
# def specific_heat_capacity(x, T):
#     """
#     Calculate the specific heat capacity of a CaCl2 solution (CaCl2-H2O) as a function of temperature and concentration.
    
#     ---
#     x: float
#         Concentration of CaCl2 in the solution [kg/kg]
#     T: float
#         Temperature of the solution [°C]
#     ---
#     returns: float
#         Specific heat capacity of the solution [kJ/kgK]
#     ---
#     Author: Dorian Höffner
#     Date: 2024-11-04
#     Source: Conde 2009 "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
#     """


# # TODO
# def dynamic_viscosity(x, T):
#     """
#     Calculate the dynamic viscosity of a LiBr solution as a function of temperature and concentration.
#     The function is based on the following publication: "Aqueous solutions of lithium and calcium chlorides: property formulations for use in air conditioning equipment design" (2009)
#     ---
#     x: float
#         Concentration of LiBr in the solution [kg/kg]
#     T: float
#         Temperature of the solution [°C]
#     ---
#     returns: float
#         Dynamic viscosity of the solution [Pa s]
#     ---
#     Last Change: Dorian Höffner 10/2024
#     Author: O. Buchin 09/2010
#     """


# # TODO
# def diffusion_coefficient(x, T):
#     """
#     Computes the self diffusion coefficient of a CaCl2 solution.

#     Parameters:
#         T (np.ndarray or float): Temperature of the solution (in Celsius) (column vector).
#         x (np.ndarray or float): Concentration of solution (0 to 1) kgSalt/kgSolution (column vector).
    
#     Returns:
#         float or np.ndarray: Self diffusion coefficient (m^2/s).

#     ---
#     Author: 
#     Last change: Dorian Höffner 11/2024
#     Source: 
#     """
    

# # TODO
# def thermal_conductivity(x, T, p):
#     """
#     Calculate the thermal conductivity of a LiBr solution as a function of temperature and concentration.
#     The function is based on the following publication:
#     ---
#     x: float
#         Concentration of LiBr in the solution [kg/kg]
#     T: float
#         Temperature of the solution [°C]
#     ---
#     returns: float
#         Thermal conductivity of the solution [W/mK]
#     ---
#     """

def hxDiagram(editablePlot=False):
    """
    Plots the pressure-temperature diagram for LiBr-H2O solutions.
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
    x_array = np.linspace(0, 0.75, 100)

    for T in T_array:
        h_array = np.array([enthalpy(x, T) for x in x_array])
        if T%20 == 0:
            plt.plot(x_array, h_array, color='black', alpha=0.7, lw=0.8, zorder=0)
            label_posx = x_array[55] + 0.005
            plt.text(label_posx, h_array[55]+2, f'{T} °C', fontsize=8, color='black')
        else:
            plt.plot(x_array, h_array, color='black', alpha=0.2, lw=0.2, zorder=0)

    plt.xlabel('Concentration $x=[kg_{LiBr}/kg_{solution}]$')
    plt.ylabel('Enthalpy $h=[kJ/kg]$')
    plt.xlim(0, 0.75)

    # plot vertical line at 40% concentration and mark left side of it as extrapolated
    plt.axvline(x=0.4, color='grey', linestyle='--', lw=0.5)
    plt.text(0.28, 330, 'Interpolated', fontsize=8, color='grey')
    # add arrow below text to point to the interpolated area
    plt.annotate('', xy=(0.4, 320), xytext=(0.25, 320), arrowprops=dict(arrowstyle='<-', color='grey'))

    # plot crystallization curve
    cryst_data = crystallization_curve(return_data=True)
    cryst_T = np.array([T for x, T in cryst_data]).flatten()
    cryst_h = np.array([enthalpy(x, T) if T > 0 else np.nan for x, T in cryst_data]).flatten()
    plt.plot([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], color='black', linestyle="--", lw=0.5, label='Crystallization Curve')
    plt.fill_between([x for x, T in cryst_data], [h for x, h in zip(cryst_data, cryst_h)], 0, color='white', zorder=0)    


    if not editablePlot:
        plt.show()

    sys.stdout = stout_old


# TODO
def pTDiagram(log=True, invT=True, editablePlot=False, show_percentages=True):
    """
    Plots the pressure-temperature diagram for LiBr-H2O solutions.
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
    concentrations = np.arange(0.1, 0.75, 0.01)
    cr_temperaturesC = cryst_T
    cr_temperaturesK = cryst_T + 273.15
    
    # Prepare the plot
    plt.figure(dpi=300)
    
    # these temperatures are used for the x-axis
    plotTemperatures = np.arange(0, 101, 10) + 273.15
    
    # Calculate water vapor pressure using XSteam
    waterPressure = [steamTable.psat_t(T - 273.15) * 1e5 if T>273.15 else np.nan for T in temperaturesK]  # convert bar to Pa

    
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
        if show_percentages and int(np.round(x * 100)) % 10 == 0 and x>0.29:
            label_pos = temp_plot[-1] + 1e-5 if invT else temp_plot[-1]+2
            plt.text(label_pos, p[-1], f'{x * 100:.0f} %', fontsize=8, color='black')

    # add description for concentration: % = "kg NaOH / kg solution"
    plt.text(label_pos, p[-1] * 0.85, r'$\left[\frac{\mathrm{kg_{LiBr}}}{\mathrm{kg_{Solution}}}\right]$', fontsize=11, color='black')

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
        plt.ylim(100, 1.4e5)
    else:
        plt.ylim(0, max(waterPressure) * 1.1)  # Adjust as needed to make sure all data is visible


    plt.legend()
    
    if not editablePlot:
        plt.show()

    # turn on print statements again
    sys.stdout = stdout_old

# DONE (56%-75%)
def solubility_temperature(x):
    """
    Calculate the crystallization temperature of LiBr solution in water.
    
    Parameters:
    x (float or np.array): Concentration of LiBr in kg/kg
    
    Returns:
    float or np.array: Crystallization temperature in °C, or -1 if outside valid concentration range.
    
    Valid concentration range:
    0.5681 <= x (kg/kg) <= 0.75

    Sources:
        Boryta, D.A. : Solubility of Lithium Bromide in water 
        between -50∞ and +100∞C (45 to 70% Lithium Bromide). 
        In: Journal of chemical and Engineering Data, Vol. 15 (1970) 1, S. 142-144

        and
        Feuerecker, G¸nther: Entropieanalyse f¸r W‰rmepumpensysteme: Methoden und Stoffdaten. 
        Institut f¸r Festkˆrperphysik und Technische Physik der Technischen Universit‰t M¸nchen,
        M¸nchen 1994
    """

    # catch invalid concentration
    if np.any((x < 0.5681) | (x > 0.75)):
        raise ValueError("Concentration out of valid range (0.5681 to 0.75).")

    # Parameters
    mu_1 = 0.660036363636364
    mu_2 = 0.0521377438043144
    
    # Adjusted concentration
    x_dach = (x - mu_1) / mu_2
    
    # Polynomial terms
    part1 = -1.25354043437046 * x_dach**7 + 1.60535142980859 * x_dach**6 + 9.50132460833796 * x_dach**5 - 10.9718095175445 * x_dach**4
    part2 = -23.0924483393181 * x_dach**3 + 23.9376211870673 * x_dach**2 + 57.4166682907763 * x_dach + 55.0110013350386
    
    # Crystallization temperature in Kelvin
    t_cryst_H2OLiBr_DBGF = part1 + part2 + 273.15
    
    # Validity check
    if np.any((x < 0.5681 - 0.0005) | (x > 0.75 + 0.0005)):
        t_cryst_H2OLiBr_DBGF = -1

    t_solubility = t_cryst_H2OLiBr_DBGF - 273.15
    
    return t_solubility


    
# TODO
def crystallization_curve(return_data=False):
    """    
    Author: Dorian Höffner
    Date:
    Data Source: Data was extracted from plot in publication.
    Publication:

    Parameters:
        return_data (bool): If True, the data will be returned as a list of lists.

    Returns:
        None or list of lists: If return_data is True, the data will be returned as a list of lists (x, T).
    """

    # cryst_data = x, T // create cryst_data based on "solubility_temperate"
    xs_array = np.linspace(0.5681, 0.75, 100)
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
        plt.xlabel(r'$\mathrm{LiBr}$ Concentration [%]')
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


# Contains Parameters for LiBr-H2O Correlations from Patek/Klomfar 2006
class Params_PK:
    def __init__(self):
        # Critical point of pure water
        self.TCritW = 647.096         # K
        self.pCritW = 22.064e6        # Pa
        self.densCritW = 322          # kg/m³
        self.densCritWmol = 17873.727 # mol/m³
        self.cpCritWmol = 76.0226     # J/(mol-K)
        self.enthalpyCritWmol = 37548.5  # J/mol
        self.entropyCritWmol = 79.3933   # J/(mol-K)

        # Triple point of pure water
        self.TTripW = 273.16          # K
        self.pTripW = 611.657         # Pa
        self.densTripW = 999.789      # kg/m³

        # Molar masses
        self.M_W = 0.018015268        # kg/mol
        self.M_LiBr = 0.08685         # kg/mol

        # Regression coefficients

        # Table 4: Pressure
        self.mTab4 = [3, 4, 4, 8, 1, 1, 4, 6]
        self.nTab4 = [0, 5, 6, 3, 0, 2, 6, 0]
        self.tTab4 = [0, 0, 0, 0, 1, 1, 1, 1]
        self.aTab4 = [-241.303, 19175000, -175521000, 32543000, 392.571, -2126.26, 185127000, 1912.16]

        # Table 5: Density
        self.mTab5 = [1, 1]
        self.tTab5 = [0, 6]
        self.aTab5 = [1.746, 4.709]

        # Table 6: Specific heat capacity (cp)
        self.mTab6 = [2, 3, 3, 3, 3, 2, 1, 1]
        self.nTab6 = [0, 0, 1, 2, 3, 0, 3, 2]
        self.tTab6 = [0, 0, 0, 0, 0, 2, 3, 4]
        self.aTab6 = [-14.2094, 40.4943, 111.135, 229.98, 1345.26, -0.014101, 0.0124977, -0.000683209]

        # Table 7: Enthalpy
        self.mTab7 = [1, 1, 2, 3, 6, 1, 3, 5, 4, 5, 5, 6, 6, 1, 2, 2, 2, 5, 6, 7, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1]
        self.nTab7 = [0, 1, 6, 6, 2, 0, 0, 4, 0, 4, 5, 5, 6, 0, 3, 5, 7, 0, 3, 1, 0, 4, 2, 6, 7, 0, 0, 1, 2, 3]
        self.tTab7 = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5]
        self.aTab7 = [
            2.27431, -7.99511, 385.239, -16394, -422.562, 0.113314, -8.33474, -17383.3,
            6.49763, 3245.52, -13464.3, 39932.2, -258877, -0.00193046, 2.80616, -40.4479,
            145.342, -2.74873, -449.743, -12.1794, -0.00583739, 0.23391, 0.341888, 8.85259,
            -17.8731, 0.0735179, -0.00017943, 0.00184261, -0.00624282, 0.00684765
        ]

        # Table 8: Entropy
        self.mTab8 = [1, 1, 2, 3, 6, 1, 3, 5, 1, 2, 2, 4, 5, 5, 6, 6, 1, 3, 5, 7, 1, 1, 1, 2, 3, 1, 1, 1, 1]
        self.nTab8 = [0, 1, 6, 6, 2, 0, 0, 4, 0, 0, 4, 0, 4, 5, 2, 5, 0, 4, 0, 1, 0, 2, 4, 7, 1, 0, 1, 2, 3]
        self.tTab8 = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5]
        self.aTab8 = [
            1.53091, -4.52564, 698.302, -21666.4, -1475.33, 0.0847012, -6.59523,
            -29533.1, 0.00956314, -0.188679, 9.31752, 5.78104, 13893.1, -17176.2,
            415.108, -55564.7, -0.00423409, 30.5242, -1.6762, 14.8283, 0.00303055,
            -0.040181, 0.149252, 2.5924, -0.177421, -0.000069965, 0.000605007,
            -0.00165228, 0.00122966
        ]
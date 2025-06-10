# Package Description

This package contains thermophysical property data for the following solutions used in absorption:
- NaOH-water
- LiBr-water
- LiCl-water
- CaCl2-water

If you want to use this package, please cite:

Höffner et al. 2025 - Potentials of absorption thermal energy storage systems in seasonal application

DOI: https://doi.org/10.18462/iir.tptpr2025.1130

### Installation and Unsage

```
pip install absorptionlib
```

```python
# Import
from absorptionlib import NaOH, LiBr, LiCl, CaCl2

# Example Usage
p = 100000 # [Pa]
x = 0.4    # [%]  = [kgNaOH / kgSolution]
t_sat = NaOH.saturation_temperature(x, p)
print(t_sat)
```
**Output**
```
129.75
```



# Quick documentation
Each of the submodules have their own "in-line" documentation, which can be called by the "documentation"-method. Each thermophysical property function has a separate docstring, which can be called by the "explain"-method.

```python
# How to get quick documentation on the modules
NaOH.documentation() # or
LiBr.documentation() # or
LiCl.documentation() # or
CaCl2.documentation()

# How to get quick documentation for functions
NaOH.explain("enthalpy")    # returns docstring for NaOH.enthalpy
LiCl.explain("pTDiagram")   # returns docstring for LiCl.pTDiagram
```


### Available Property-Functions

| Function Name              | Description                                                                                   |
|---------------------------|------------------------------------------------------------------------------------------------|
| saturation_temperature    | Calculate the boiling point temperature of solution.                |
| enthalpy                  | Calculate the enthalpy of the solution at a given temperature and concentration.       |
| differential_enthalpy_AD  | Calculates the differential enthalpy of the solution.                                      |
| saturation_pressure       | Calculate the equilibrium pressure of the solution.                                    |
| saturation_concentration  | Calculates the saturation concentration of the solution based on the temperature and pressure.|
| density                   | Calculate the density of a the solution.                                                |
| specific_heat_capacity    | Calculate the specific heat capacity of the solution.                                      |
| dynamic_viscosity         | Calculate the dynamic viscosity of the solution.                                            |
| diffusion_coefficient     | Computes the diffusion coefficient of the solution.                                   |
| solubility_temperature    | Calculate the crystallization temperature based on concentration.                           |
| hxDiagram                 | Plots the enthalpy-concentration diagram for the respective solution.                                 |
| pTDiagram                 | Plots the pressure-temperature diagram the respective solutions.                                 |
| crystallization_curve     | Plots the crystallization curve for the solution |


# Diagrams

As stated before, you can construct diagrams (pT-Diagram, hx-Diagram, crystallization-curve) with the package. For example:

```python
NaOH.pTDiagram()
```

![pT-Diagram](https://github.com/dorianhoeffner/absorptionlib/blob/main/graphics/pTDiagram_example.png)

Note: The plots can be styled with usual matplotlib syntax. If the plot should be editable, use ```NaOH.pTDiagram(editablePlot=True)```. To reproduce the same-looking plot, set up matplotlib using:

```python
# matplotlib font to geogia
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Georgia'
plt.rcParams['mathtext.it'] = 'Georgia:italic'
```


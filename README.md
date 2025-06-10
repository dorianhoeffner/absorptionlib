# Package Description

This package contains thermophysical property data for the following solutions used in absorption:
- NaOH-water
- LiBr-water
- LiCl-water
- CaCl2-water

If you want to use this package, please cite:
Höffner et al. 2025 - Potentials of absorption thermal energy storage systems in seasonal application
DOI: 10.18462/iir.tptpr2025.1130

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
t_sat = NaOH.saturation_temperature(p, x)
print(t_sat)
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

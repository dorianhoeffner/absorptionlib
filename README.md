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

| Function                         | Parameter 1           | Parameter 2           | Parameter 3    | Return Unit   |
|----------------------------------|-----------------------|-----------------------|----------------|---------------|
| saturation_temperature(P, x1)    | P (Pa)                | x (kgNaOH/kgSolution) | -              | °C            |
| saturation_pressure(x, T)        | x (kgNaOH/kgSolution) | T (°C)                | -              | Pa            |
| saturation_concentration(x, T)   | P (Pa)                | T (°C)                | -              | kg/kg         |
| enthalpy(x, T)                   | x (kgNaOH/kgSolution) | T (°C)                | -              | kJ/kg         |
| differential_enthalpy_AD(x, T)   | x (kgNaOH/kgSolution) | T (°C)                | -              | kJ/kg         |
| specific_heat_capacity(x, T)     | x (kgNaOH/kgSolution) | T (°C)                | -              | kJ/kg·K       |
| density(x, T)                    | x (kgNaOH/kgSolution) | T (°C)                | -              | kg/m³         |
| dynamic_viscosity(x, T)          | x (kgNaOH/kgSolution) | T (°C)                | -              | Pa·s          |
| diffusion_coefficient(x, T)      | x (kgNaOH/kgSolution) | T (°C)                | -              | m2/s          |
| thermal_conductivity(x, T, p)    | x (kgNaOH/kgSolution) | T (°C)                | p (Pa)         | W/m·K         |
| solubility_temperature(x)        | x (kgNaOH/kgSolution) | -                     | -              | W/m·K         |


### Available Utilities

| Function                 | Description                  | Boolean Options + Defaults                              |
|--------------------------|------------------------------|---------------------------------------------------------|
| hxDiagram()              | shows hx-Diagram             |                                                         |
| pTDiagram()              | shows pT-Diagram             | log=True, invT=True                                     |
|                          |                              | editablePlot=False, show_percentages=True               |
| crystallization_curve()  | shows crystallization curve  | return_data=False (if True, only data returned, no plot)|


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



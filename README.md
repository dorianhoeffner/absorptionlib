# absorptionlib

This package contains thermophysical property functions for the following aqueous solutions used in absorption systems:
- NaOH-water
- LiBr-water
- LiCl-water
- CaCl2-water

If you use this package, please cite:

Höffner et al. 2025 — Potentials of absorption thermal energy storage systems in seasonal application

DOI: https://doi.org/10.18462/iir.tptpr2025.1130

### Absorption Atlas — try it in the browser

Explore the property functions interactively, without installing anything:
p–T, h–x and solubility diagrams, a property calculator with matching
absorptionlib code, and a storage-cycle analysis comparing all four salts
(method: Höffner, Ziegler & Elbel 2026, *Energy Conversion and Management* 366,
121845, https://doi.org/10.1016/j.enconman.2026.121845).

**https://dorianhoeffner.github.io/absorptionlib/**

### Installation and Usage

```
pip install absorptionlib
```

```python
# Import
from absorptionlib import NaOH, LiBr, LiCl, CaCl2

# Example Usage
p = 100000  # [Pa]
x = 0.4     # [kg NaOH / kg solution]
t_sat = NaOH.saturation_temperature(x, p)
print(round(t_sat, 2))
```
**Output**
```
129.75
```

### Units

All functions use the same units throughout the package:

| Symbol | Meaning                                | Unit  |
|--------|----------------------------------------|-------|
| x      | Salt mass fraction                     | kg salt / kg solution |
| T      | Temperature                            | °C    |
| p      | Pressure                               | Pa    |

### Warnings and errors

Each function checks its inputs against the validity range of the underlying
correlation:

- Inputs outside the *validated* range emit an `OutOfRangeWarning` (the value
  is still computed and returned).
- State points below the crystallization line emit a `CrystallizationWarning`.
- Physically invalid inputs raise a `ValueError`.

Every property function accepts `prevent_errors=True`, which suppresses all
warnings and returns `float('nan')` instead of raising a `ValueError` — useful
inside optimizers and root finders:

```python
NaOH.enthalpy(0.4, 80, prevent_errors=True)
```

Warnings can also be controlled globally:

```python
import absorptionlib
absorptionlib.disable_warnings()   # silence all package warnings
absorptionlib.enable_warnings()    # restore them

# or with the standard library, e.g. turn warnings into exceptions:
import warnings
warnings.filterwarnings("error", category=absorptionlib.CrystallizationWarning)
```

# Quick documentation

Each submodule has its own "in-line" documentation, which can be called with the
`documentation()` function. Each property function has a docstring, which can be
printed with the `explain()` function.

```python
# How to get quick documentation on the modules
NaOH.documentation()  # or LiBr, LiCl, CaCl2

# How to get quick documentation for functions
NaOH.explain("enthalpy")    # prints docstring of NaOH.enthalpy
LiCl.explain("pTDiagram")   # prints docstring of LiCl.pTDiagram
```

### Available Property Functions

| Function Name             | Description                                                                    | NaOH | LiBr | LiCl | CaCl2 |
|---------------------------|--------------------------------------------------------------------------------|:----:|:----:|:----:|:-----:|
| saturation_temperature    | Boiling point temperature of the solution [°C]                                | x    | x    | x    | x     |
| saturation_pressure       | Equilibrium (vapor) pressure of the solution [Pa]                             | x    | x    | x    | x     |
| saturation_concentration  | Saturation concentration at given pressure and temperature [kg/kg]            | x    | x    | x    | x     |
| enthalpy                  | Specific enthalpy of the solution [kJ/kg]                                     | x    | x    | x    | x     |
| differential_enthalpy_AD  | Differential enthalpy of dilution [kJ/kg H2O]                                 |      |      | x    | x     |
| density                   | Density of the solution [kg/m³]                                               | x    | x    | x    | x     |
| specific_heat_capacity    | Specific heat capacity of the solution [kJ/(kg K)]                            | x    |      | x    | x     |
| dynamic_viscosity         | Dynamic viscosity of the solution [Pa s]                                      | x    |      | x    | x     |
| thermal_conductivity      | Thermal conductivity of the solution [W/(m K)]                                | x    |      |      |       |
| diffusion_coefficient     | Self diffusion coefficient of the solution [m²/s]                             |      |      | x    | x     |
| solubility_temperature    | Crystallization temperature at given concentration [°C]                       | x    | x    | x    | x     |
| hxDiagram                 | Plots the enthalpy-concentration diagram                                      | x    | x    | x    | x     |
| pTDiagram                 | Plots the pressure-temperature diagram                                        | x    | x    | x    | x     |
| crystallization_curve     | Plots (or returns) the crystallization curve                                  | x    | x    | x    | x     |

# Diagrams

You can construct diagrams (pT-diagram, hx-diagram, crystallization curve) with the package. For example:

```python
NaOH.pTDiagram()
```

![pT-Diagram](https://raw.githubusercontent.com/dorianhoeffner/absorptionlib/main/graphics/pTDiagram_example.png)

Note: The plots can be styled with the usual matplotlib syntax. If the plot should be editable, use `NaOH.pTDiagram(editablePlot=True)`. To reproduce the same-looking plot, set up matplotlib using:

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Georgia'
plt.rcParams['mathtext.it'] = 'Georgia:italic'
```

# Package Description

This file contains property data for aqueous sodium hydroxide solutions.

### Installation and Unsage

```
pip install propertiesNaOH
```

```python
# Import
import propertiesNaOH as NaOH

# Example Usage
p = 100000 # [Pa]
x = 0.4    # [%]  = [kgNaOH / kgSolution]
t_sat = NaOH.saturation_temperature(p, x)
print(t_sat)
```

### Available Property-Functions

| Function                         | Parameter 1           | Parameter 2           | Parameter 3    | Return Unit   |
|----------------------------------|-----------------------|-----------------------|----------------|---------------|
| saturation_temperature(P, x1)    | P (Pa)                | x (kgNaOH/kgSolution) | -              | °C            |
| saturation_pressure(x, T)        | x (kgNaOH/kgSolution) | T (°C)                | -              | Pa            |
| saturation_concentration(x, T)   | P (Pa)                | T (°C)                | -              | kg/kg         |
| enthalpy(x, T)                   | x (kgNaOH/kgSolution) | T (°C)                | -              | kJ/kg         |
| specific_heat_capacity(x, T)     | x (kgNaOH/kgSolution) | T (°C)                | -              | kJ/kg·K       |
| density(x, T)                    | x (kgNaOH/kgSolution) | T (°C)                | -              | kg/m³         |
| dynamic_viscosity(x, T)          | x (kgNaOH/kgSolution) | T (°C)                | -              | Pa·s          |
| thermal_conductivity(x, T, p)    | x (kgNaOH/kgSolution) | T (°C)                | p (Pa)         | W/m·K         |


### Available Utilities


| Function                 | Description                  | Boolean Options + Defaults                              |
|--------------------------|------------------------------|---------------------------------------------------------|
| pTDiagram()              | shows pT-Diagram             | log=True, invT=True                                     |
|                          |                              | editablePlot=False, show_percentages=True               |
| crystallization_curve()  | shows crystallization curve  | return_data=False (if True, only data returned, no plot)|

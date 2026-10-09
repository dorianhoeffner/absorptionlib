# absorptionlib 1.1.0 — Refactoring Changelog

Full review and refactor of the package (2026-07-03). Every numerical
correlation was regression-tested against the previous version on a grid of
state points: all values are unchanged except where a bug was deliberately
fixed (each such case is marked **[behavior change]** below).

## 1. Standardized warning and error system (all modules)

- All warnings previously emitted via `print()` now use Python's `warnings`
  module with two package-specific categories: `OutOfRangeWarning` (inputs
  outside a correlation's validated range; the value is still computed) and
  `CrystallizationWarning` (state point below the solubility line), both
  subclasses of `AbsorptionLibWarning`. They can be filtered, logged, or
  turned into exceptions with standard tooling.
- New global switches: `absorptionlib.disable_warnings()` /
  `absorptionlib.enable_warnings()`.
- **Every** property function now accepts `prevent_errors=False` (previously
  only some did). With `prevent_errors=True`, all warnings are suppressed and
  `ValueError` is replaced by a `float('nan')` return — safe for optimizers.
- All error/warning messages follow one format:
  `absorptionlib.<Module>.<function>: <description with values and valid range>`.
- Invalid-input returns standardized to `NaN` (previously a mix of `None`,
  `-1`, `-2`, and silent nonsense values). **[behavior change]**
- Plot functions no longer hijack `sys.stdout` to silence warnings (the old
  approach left stdout broken if an exception occurred mid-plot, and
  `LiCl.saturation_temperature` redirected stdout *to itself*, silencing
  nothing). They now use a proper warnings-suppression context.

## 2. Logical errors fixed

### NaOH
- `enthalpy`: the crystallization check was **inverted** — it only ran when
  `prevent_errors=True` and never in normal operation. Now it always runs
  (unless suppressed). **[behavior change]**
- `solubility_temperature`: error message was missing the `f` prefix
  (`'{x} must be less...'` printed literally).
- `saturation_temperature` / `saturation_pressure`: returned `None` below the
  crystallization line in normal mode; now warn and return the extrapolated
  value. **[behavior change]**
- `saturation_concentration`: crashed with `TypeError` or fell back to a slow
  0.001-step increment loop returning `x = 0.001` when no solution existed;
  now uses a bracketing root finder (brentq) and raises a clear `ValueError`
  (or returns NaN) when the pressure has no solution in range.
  **[behavior change]**
- `dynamic_viscosity`: warning text said "p > 15 MPa" while the check was
  against 7 MPa; unused `my_rel` variable removed.
- `density`: docstring said "Mole fraction" for a mass-fraction input.
- `thermal_conductivity`: result was multiplied by 1000 (the Alexandrov
  correlation already yields W/(m K)), e.g. 627.9 instead of 0.628 W/(m K) at
  x = 0.1, 20 °C. Factor removed. **[behavior change — results corrected]**

### LiBr
- `saturation_temperature`: unbounded `fsolve` replaced by bracketed `brentq`;
  previously it raised uncaught errors at pressures ≥ ~10 kPa (the solver
  stepped outside the correlation's range). Now returns correct values, e.g.
  `saturation_temperature(0.45, 100000) = 123.2 °C`. **[behavior change]**
- `saturation_concentration`: contained two sequential increment loops (the
  first was dead code — its result was discarded), returned `x = 0.001` when
  no solution existed, and could crash with `TypeError`. Replaced by `brentq`
  (also ~3-digit-accurate values instead of 0.001-step quantization).
  **[behavior change]**
- `saturation_pressure`: docstring claimed the input is in Kelvin
  (273.15–500 K) while the code expects °C — docs corrected to °C; error
  message now reports °C too.
- `enthalpy`: docstring claimed "returns -1 / -2" on invalid input while the
  code raises — corrected; parameters renamed `konz, tempC` → `x, T`
  (uniform API). The <40 % interpolation formula was rewritten as an
  equivalent, readable expression (identical results).
- `solubility_temperature`: dead unreachable code (the "-1 return" block
  after the range check that already raised) removed; array input supported.
- `enthalpy_PK(x, T)` → `enthalpy_PK(T)`: the `x` argument was never used;
  docstring completed (was a template with wrong parameter list).
  **[behavior change: signature]**
- Unused `CoolProp` import removed (`PropsSI` was imported but never used) —
  CoolProp is no longer a dependency.
- `Params_PK` is instantiated once at import instead of on every
  `saturation_pressure` call.

### LiCl
- `dynamic_viscosity`: the water viscosity for T > 0 °C was a hardcoded
  placeholder (`etaH2O_0 = 0.001`, marked "should be replaced") — results
  were ~50 % too high at 40 °C. Now uses pyXSteam (IAPWS-IF97), same as the
  CaCl2 module always did. **[behavior change — results corrected]**
- `saturation_temperature`: `contextlib.redirect_stdout(sys.stdout)` was a
  no-op (redirecting stdout to itself); unbounded `fsolve` silently returned
  the initial guess (20 °C!) at higher pressures, e.g.
  `saturation_temperature(0.35, 100000)` returned `20.0` instead of
  `127.1 °C`. Replaced by bracketed `brentq`. **[behavior change — results
  corrected]**
- `saturation_concentration`: could return physically impossible values
  (e.g. `x = 1.121` at 2 kPa / 120 °C) via the increment-loop fallback; now
  bounded to the correlation's validity (0–0.56) and errors clearly.
  **[behavior change]**
- `enthalpy`: no longer returns NaN silently below the solubility line; it
  warns (CrystallizationWarning) and returns the extrapolated value, matching
  the package-wide policy. Mass-fraction validity check (0–0.56) added.
  **[behavior change]**
- `crystallization_curve`: plot was labeled "CaCl2 Concentration" — now LiCl;
  curve range capped at the correlation limit 0.56 (was 0.6).
- `specific_heat_capacity` / `diffusion_coefficient`: now return `float` for
  scalar input (previously 0-d numpy arrays); docstring placeholder "XXXXXX"
  fixed; `hxDiagram` docstring said "NaOH".
- Duplicate imports (`sys`, `contextlib`, `fsolve`) removed.

### CaCl2
- `solubility_temperature`: array input returned only the **first element**
  (`ts = ts[0]` ran unconditionally because `np.max(t, axis=0)` is always
  1-D). Scalars still work; arrays now return arrays. **[behavior change for
  array input — corrected]**; also the indexing `t[k] = ...` (row-assign of
  scalar to full row) replaced by explicit per-column assignment.
- `enthalpy`: internal `quad` integration called `differential_enthalpy_AD`
  without `prevent_errors`, spamming a solubility warning for every
  integration point; inner calls are now silenced and one check is done at
  the top level. Variable `cp_LiCl_25` renamed (it is CaCl2); docstring
  mentioned LiCl ("x = m_LiCl / ...") — corrected.
- `saturation_temperature`: unbounded `fsolve` silently returned the initial
  guess (20 °C) at 1 bar; now `brentq` returns e.g. 112.2 °C at x = 0.35.
  **[behavior change — results corrected]**
- `saturation_concentration`: increment loop returning bogus `x = 0.001`
  replaced by `brentq`. **[behavior change]**
- `differential_enthalpy_AD`: gained `prevent_errors` (LiCl's version had it,
  CaCl2's didn't); domain check `x < 0.8` added (the formula divides by
  `0.8 - x`).
- `thermal_conductivity`: removed — it was an empty stub that printed "not
  implemented yet" and returned `None`. **[behavior change: removed]**
- Copy-paste docstrings fixed throughout (`saturation_pressure`, `density`,
  `saturation_concentration`, `dynamic_viscosity`, `pTDiagram` all described
  "NaOH solutions").

## 3. Standardized API and documentation

- Uniform parameter names and units everywhere: `x` [kg salt/kg solution],
  `T` [°C], `p` [Pa]; uniform argument order (`x, T`), (`x, p`), (`p, T`).
- All `dynamic_viscosity` functions now return **[Pa s]**. Previously
  LiCl/CaCl2 returned [mPa s] while NaOH returned [Pa s] — LiCl/CaCl2
  results are now a factor 1000 smaller. **[behavior change]**
- Uniform docstring structure: description, Parameters, Returns, validity
  range, Source, Author/History.
- `documentation()` tables are now generated from a per-module description
  dict — previously all four modules printed a copy of the LiBr table
  (NaOH's table described LiBr and CaCl2 functions, listed functions that
  didn't exist, and described `hxDiagram` as "pressure-temperature diagram").
- `explain()` now lists the available functions when the requested name is
  not found.
- Dead code removed: ~200 lines of commented-out lookup tables (NaOH),
  commented-out old density/TODO stubs (LiBr, CaCl2, LiCl), "INEFFICIENCIES"
  note block (NaOH).
- `pTDiagram(show_percentages=False)` no longer crashes with `NameError`
  (`label_pos` was only defined inside the percentage branch).

## 4. Packaging and repository cleanup

- `setup.py` replaced by `pyproject.toml` (PEP 621); version bumped to
  **1.1.0**; `coolprop` removed from dependencies; `python_requires >= 3.8`.
- New `absorptionlib/_common.py` with the shared warning/documentation
  machinery.
- `__init__.py`: added `__version__`, `__all__`, and re-exports of the
  warning categories and `enable_warnings`/`disable_warnings`.
- Removed committed build artifacts: `build/`, `dist/`, `absorptionlib.egg-info/`,
  all `__pycache__/` and `.ipynb_checkpoints/`, and the stray duplicates
  `absorptionlib/setup.py`, `absorptionlib/README.md`, `absorptionlib/LICENSE.txt`.
- Added `.gitignore`.
- README: fixed "Unsage" typo, wrong unit comment (`[%]` for a kg/kg value),
  added a units table, a warnings/errors section, and a per-salt function
  availability matrix.

## 5. Verification

- Regression test: all property values on a 4-salt grid (psat, h, rho, cp,
  tsat, xsat, tsol, mu, lambda, D, dh_AD) match the previous version to
  < 1e-6 relative, except the deliberate fixes listed above.
- Smoke tests: README example (129.75 °C) reproduced; warning categories,
  `prevent_errors`, `disable_warnings()`, array inputs, `documentation()`/
  `explain()`, and all 12 diagram functions run cleanly.
- `python -m build` produces a valid wheel (absorptionlib-1.1.0).

## Migration notes

- If your code relied on `None`/`-1`/`0.001` sentinel returns, switch to
  checking `math.isnan(...)` or catching `ValueError`.
- If you parsed printed warnings, use
  `warnings.filterwarnings(...)` with the new categories instead.
- `LiBr.enthalpy_PK` now takes only `T` (in K).
- `LiCl.dynamic_viscosity` and `CaCl2.dynamic_viscosity` now return [Pa s]
  instead of [mPa s] (multiply old downstream expectations by 1e-3).
- `CaCl2.thermal_conductivity` (non-functional stub) was removed.
- `LiCl.dynamic_viscosity` and the LiCl/CaCl2 `saturation_temperature`
  results changed because previous values were wrong (see above).

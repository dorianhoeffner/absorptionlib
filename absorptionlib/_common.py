"""
Shared utilities for absorptionlib.

Standardized warning categories, validity-check helpers, and the
``documentation()`` / ``explain()`` machinery used by all salt modules.

Warning / error policy (uniform across the package)
---------------------------------------------------
1. Results outside the validated range of a correlation are still computed
   and returned, but an ``OutOfRangeWarning`` is emitted.
2. Conditions below the crystallization (solubility) line emit a
   ``CrystallizationWarning``; the (extrapolated) value is still returned.
3. Physically meaningless inputs (e.g. x < 0, temperatures far outside the
   correlation) raise ``ValueError``.
4. Every property function accepts ``prevent_errors=False``. Setting it to
   True suppresses all warnings and converts ``ValueError`` into a
   ``float('nan')`` return value. This makes the functions safe to use
   inside optimizers and root finders.

Global control
--------------
>>> import absorptionlib
>>> absorptionlib.disable_warnings()          # silence all package warnings
>>> absorptionlib.enable_warnings()           # restore them

or, with the standard library:

>>> import warnings
>>> warnings.filterwarnings("ignore", category=absorptionlib.AbsorptionLibWarning)
>>> warnings.filterwarnings("error",  category=absorptionlib.CrystallizationWarning)

Author: Dorian Höffner
"""

import contextlib
import warnings


# ---------------------------------------------------------------------------
# Warning categories
# ---------------------------------------------------------------------------

class AbsorptionLibWarning(UserWarning):
    """Base category for all warnings emitted by absorptionlib."""


class OutOfRangeWarning(AbsorptionLibWarning):
    """Inputs are outside the validated range of the underlying correlation."""


class CrystallizationWarning(AbsorptionLibWarning):
    """State point lies below the crystallization (solubility) line."""


def enable_warnings():
    """Re-enable all absorptionlib warnings (undo ``disable_warnings``)."""
    warnings.filterwarnings("default", category=AbsorptionLibWarning)


def disable_warnings():
    """Globally silence all absorptionlib warnings."""
    warnings.filterwarnings("ignore", category=AbsorptionLibWarning)


# ---------------------------------------------------------------------------
# Internal helpers (used by the salt modules)
# ---------------------------------------------------------------------------

def _msg(module, func, text):
    return f"absorptionlib.{module}.{func}: {text}"


def warn_out_of_range(module, func, text, prevent_errors=False):
    """Emit a standardized OutOfRangeWarning unless suppressed."""
    if not prevent_errors:
        warnings.warn(_msg(module, func, text), OutOfRangeWarning, stacklevel=3)


def warn_crystallization(module, func, T, t_sol, x, prevent_errors=False):
    """Emit a standardized CrystallizationWarning unless suppressed."""
    if not prevent_errors:
        warnings.warn(
            _msg(module, func,
                 f"T = {T:.2f} °C is below the crystallization temperature "
                 f"t_sol = {t_sol:.2f} °C at x = {x:.4f}. The solution is not "
                 f"in a stable liquid state; the returned value is an "
                 f"extrapolation."),
            CrystallizationWarning, stacklevel=3)


@contextlib.contextmanager
def suppress_warnings():
    """Context manager that silences absorptionlib warnings (internal use)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AbsorptionLibWarning)
        yield


def print_documentation(module_name, salt, descriptions):
    """Print the standardized module documentation table."""
    name_w = max(len(n) for n in descriptions) + 1
    lines = [
        f"\nThis module contains functions for calculating properties of "
        f"{salt}-H2O solutions:\n",
        f"| {'Function Name'.ljust(name_w)} | Description",
        f"|{'-' * (name_w + 2)}|{'-' * 80}",
    ]
    for name, desc in descriptions.items():
        lines.append(f"| {name.ljust(name_w)} | {desc}")
    lines += [
        "",
        f"For more information use: {module_name}.explain(\"function_name\")",
        f"For example: {module_name}.explain(\"enthalpy\")",
        "",
        "All property functions accept prevent_errors=True to suppress "
        "warnings and return NaN instead of raising errors (useful in "
        "optimizers).",
    ]
    print("\n".join(lines))


def explain_function(module_globals, module_name, function_name, descriptions):
    """Print the docstring of one public function of a salt module."""
    func = module_globals.get(function_name)
    if not callable(func) or function_name not in descriptions:
        available = ", ".join(descriptions)
        print(f"Function '{function_name}' not found in {module_name}. "
              f"Available functions: {available}")
        return
    print(func.__doc__)

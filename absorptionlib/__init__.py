"""
absorptionlib — thermophysical properties of salt solutions for absorption
systems (NaOH, LiBr, LiCl, CaCl2 in water).

Usage:
    >>> from absorptionlib import NaOH
    >>> NaOH.saturation_temperature(x=0.4, p=100000)

Warnings can be controlled globally:
    >>> import absorptionlib
    >>> absorptionlib.disable_warnings()
or per call with the ``prevent_errors=True`` keyword available on every
property function.
"""

from .CaCl2 import functions as CaCl2
from .LiBr import functions as LiBr
from .LiCl import functions as LiCl
from .NaOH import functions as NaOH

from ._common import (
    AbsorptionLibWarning,
    OutOfRangeWarning,
    CrystallizationWarning,
    enable_warnings,
    disable_warnings,
)

__version__ = "1.1.0"

__all__ = [
    "CaCl2",
    "LiBr",
    "LiCl",
    "NaOH",
    "AbsorptionLibWarning",
    "OutOfRangeWarning",
    "CrystallizationWarning",
    "enable_warnings",
    "disable_warnings",
    "__version__",
]

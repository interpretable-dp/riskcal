"""riskcal: Privacy risk analysis and calibration for differential privacy.

This library provides tools for computing f-DP trade-off curves for differential
privacy mechanisms, and calibrating their noise to operational privacy risk
(attack accuracy/advantage, or attack TPR and FPR) instead of (epsilon, delta).

Recommended usage:
    from riskcal import analysis  # For computing privacy metrics
    from riskcal import calibration  # For noise calibration
    from riskcal import accountants  # For privacy accountants
"""

# New recommended API
from riskcal import analysis
from riskcal import calibration
from riskcal import accountants

# Lazy deprecation for legacy modules and top-level names (PEP 562).
# Warnings are emitted only when deprecated names are actually accessed.

_DEPRECATED_MODULES = {
    "dpsgd": (
        "The 'riskcal.dpsgd' module is deprecated and will be removed in v2.0.0. "
        "Use 'riskcal.calibration.dpsgd' for calibration functions and "
        "'riskcal.accountants' for CTDAccountant instead."
    ),
    "conversions": (
        "The 'riskcal.conversions' module is deprecated and will be removed in v2.0.0. "
        "Use 'riskcal.analysis' instead."
    ),
    "blackbox": (
        "The 'riskcal.blackbox' module is deprecated and will be removed in v2.0.0. "
        "Use 'riskcal.calibration.blackbox' instead."
    ),
    "plrv": (
        "The 'riskcal.plrv' module is deprecated and will be removed in v2.0.0. "
        "Use 'riskcal.analysis' instead."
    ),
}

_DEPRECATED_TOPLEVEL = {
    # (new_module_path, migration_message)
    "get_advantage_from_pld": (
        "riskcal.analysis",
        "Use 'from riskcal.analysis import get_advantage_from_pld' instead.",
    ),
    "get_beta_from_pld": (
        "riskcal.analysis",
        "Use 'from riskcal.analysis import get_beta_from_pld' instead.",
    ),
    "get_advantage_for_dpsgd": (
        "riskcal.calibration.dpsgd",
        "Use 'from riskcal.calibration.dpsgd import get_advantage_for_dpsgd' instead.",
    ),
    "get_beta_for_dpsgd": (
        "riskcal.calibration.dpsgd",
        "Use 'from riskcal.calibration.dpsgd import get_beta_for_dpsgd' instead.",
    ),
    "find_noise_multiplier_for_advantage": (
        "riskcal.calibration.dpsgd",
        "Use 'from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage' instead.",
    ),
    "find_noise_multiplier_for_err_rates": (
        "riskcal.calibration.dpsgd",
        "Use 'from riskcal.calibration.dpsgd import find_noise_multiplier_for_err_rates' instead.",
    ),
    "CTDAccountant": (
        "riskcal.accountants",
        "Use 'from riskcal.accountants import CTDAccountant' instead.",
    ),
}


def __getattr__(name):
    import importlib
    import warnings

    if name in _DEPRECATED_MODULES:
        warnings.warn(_DEPRECATED_MODULES[name], DeprecationWarning, stacklevel=2)
        # Suppress the module-level warning to avoid double warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return importlib.import_module(f"riskcal.{name}")

    if name in _DEPRECATED_TOPLEVEL:
        module_path, msg = _DEPRECATED_TOPLEVEL[name]
        warnings.warn(
            f"Accessing 'riskcal.{name}' is deprecated and will be removed in v2.0.0. {msg}",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(module_path)
        return getattr(mod, name)

    raise AttributeError(f"module 'riskcal' has no attribute {name!r}")


__all__ = [
    # New API (recommended)
    "analysis",
    "calibration",
    "accountants",
    # Legacy modules (deprecated, lazy)
    "dpsgd",
    "conversions",
    "blackbox",
    "plrv",
    # Legacy top-level exports (deprecated, lazy)
    "get_advantage_from_pld",
    "get_beta_from_pld",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    "CTDAccountant",
]

try:
    from riskcal.analysis.rdp.converter import get_FNR
    _CYTHON_AVAILABLE = True
except Exception:
    from riskcal.analysis.rdp.pure import get_FNR
    _CYTHON_AVAILABLE = False


def is_cython_available() -> bool:
    """Return True if the compiled Cython extension for RDP conversion is loaded."""
    return _CYTHON_AVAILABLE

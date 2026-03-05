"""Privacy risk analysis module.

This module provides functions for computing privacy metrics from various
privacy representations (PLDs, Gaussian mechanisms, epsilon-delta parameters).
"""

# Export PLRVs dataclass for advanced users
from riskcal.analysis._plrv import PLRVs, get_beta as get_beta_from_plrvs

# Export all functions from conversions module
from riskcal.analysis.conversions import (
    # Internal conversions
    plrvs_from_pld,
    pld_to_plrvs,
    # Beta functions
    get_beta_from_pld,
    get_beta_from_gdp,
    get_beta_from_rdp,
    get_beta_from_zcdp,
    get_beta_from_adp,
    get_beta_for_mu,
    get_beta_for_epsilon_delta,
    # Advantage functions
    get_advantage_from_pld,
    get_advantage_from_gdp,
    get_advantage_from_adp,
    get_advantage_from_zcdp,
    get_mu_from_zcdp_approx,
    get_beta_from_zcdp_approx,
    get_advantage_from_zcdp_approx,
    get_advantage_for_mu,
    get_advantage_for_epsilon_delta,
    # Bayes risk functions
    get_bayes_risk_from_pld,
    get_bayes_risk_from_gdp,
    get_bayes_risk_for_mu,
    # ADP
    get_epsilon_from_err_rates,
    get_epsilon_for_err_rates,
)

__all__ = [
    # Data structures
    "PLRVs",
    # Internal conversions
    "plrvs_from_pld",
    "pld_to_plrvs",
    # Beta (FNR) functions
    "get_beta_from_pld",
    "get_beta_from_gdp",
    "get_beta_from_adp",
    "get_beta_from_rdp",
    "get_beta_from_zcdp",
    "get_beta_for_mu",  # Deprecated alias
    "get_beta_for_epsilon_delta",  # Deprecated alias
    "get_beta_from_plrvs",
    # Advantage functions
    "get_advantage_from_pld",
    "get_advantage_from_gdp",
    "get_advantage_from_adp",
    "get_advantage_from_zcdp",
    "get_mu_from_zcdp_approx",  # Deprecated
    "get_beta_from_zcdp_approx",  # Deprecated
    "get_advantage_from_zcdp_approx",  # Deprecated
    "get_advantage_for_mu",  # Deprecated alias
    "get_advantage_for_epsilon_delta",  # Deprecated alias
    # Bayes risk functions
    "get_bayes_risk_from_pld",
    "get_bayes_risk_from_gdp",
    "get_bayes_risk_for_mu",  # Deprecated alias
    # Conversions
    "get_epsilon_from_err_rates",
    "get_epsilon_for_err_rates",  # Deprecated alias
]

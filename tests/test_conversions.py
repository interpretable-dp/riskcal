import pytest
import numpy as np
import math

from scipy.optimize import minimize_scalar
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

import riskcal


def _get_bayes_risk_from_kaissis_definition(pld, prior):
    """Optimization-based Bayes risk. Kaissis et al. (2024), Eq. 2. https://arxiv.org/abs/2406.08918"""
    from riskcal.utils import _ensure_array

    prior, is_scalar = _ensure_array(prior)
    bayes_risk = []
    for prior_val in prior:
        result = minimize_scalar(
            lambda x: prior_val * x
            + (1 - prior_val) * riskcal.analysis.get_beta_from_pld(pld, alpha=x),
            bounds=(0, 1),
            method="bounded",
        )
        bayes_risk.append(result.fun)
    if is_scalar:
        return bayes_risk[0]
    return np.array(bayes_risk)


# Global tolerances for test assertions
REL_TOLERANCE = 1e-4  # Relative tolerance for numerical comparisons
ABS_TOLERANCE = 1e-12  # Absolute tolerance for small numbers


def get_gaussian_plrv_data():
    mu = 2.0
    pld = from_gaussian_mechanism(
        standard_deviation=1.0 / mu, value_discretization_interval=1e-3
    )
    return mu, pld, riskcal.analysis.conversions.plrvs_from_pld(pld)


def get_subsampled_gaussian_plrv_data():
    mu = 2.5
    pld = from_gaussian_mechanism(
        standard_deviation=1.0 / mu, sampling_prob=0.1
    ).self_compose(5)
    return mu, pld, riskcal.analysis.conversions.plrvs_from_pld(pld)


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_beta_matches_analytic_curve(alpha):
    mu, pld, plrvs = get_gaussian_plrv_data()
    analytic_beta = riskcal.analysis.conversions.get_beta_for_mu(mu, alpha)
    numerical_beta1 = riskcal.plrv.get_beta(plrvs, alpha=alpha)
    numerical_beta2 = riskcal.analysis.conversions.get_beta_from_pld(pld, alpha=alpha)

    # check that numerical beta is close to analytic beta
    assert analytic_beta == pytest.approx(
        numerical_beta1, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )
    assert analytic_beta == pytest.approx(
        numerical_beta2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


def test_deprecated_get_beta_interface():
    mu, pld, plrvs = get_gaussian_plrv_data()
    alpha = 0.5
    analytic_beta = riskcal.analysis.conversions.get_beta_for_mu(mu, alpha)

    with pytest.warns(DeprecationWarning):
        numerical_beta1 = riskcal.plrv.get_beta(plrvs, alphas=alpha)
        numerical_beta2 = riskcal.analysis.conversions.get_beta_from_pld(
            pld, alphas=alpha
        )

    # check that numerical beta is close to analytic beta
    assert analytic_beta == pytest.approx(
        numerical_beta1, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )
    assert analytic_beta == pytest.approx(
        numerical_beta2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


def test_get_beta_symmetrized_in_the_middle():
    mu, pld, plrvs = get_subsampled_gaussian_plrv_data()
    analytic_advantage = pld.get_delta_for_epsilon(0)
    analytic_fixed_point_beta = 0.5 * (1 - analytic_advantage)

    alpha = np.linspace(0, 1, 100_000)
    beta = riskcal.analysis.conversions.get_beta_from_pld(
        pld,
        alpha=alpha,
    )
    numerical_fixed_point_idx = np.argmin(np.abs(beta - analytic_fixed_point_beta))
    numerical_fixed_point = beta[numerical_fixed_point_idx]

    assert analytic_fixed_point_beta == pytest.approx(
        numerical_fixed_point, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


def test_get_advantage_matches_analytic_matching_expression():
    mu, pld, plrvs = get_gaussian_plrv_data()
    analytic_advantage = riskcal.analysis.conversions.get_advantage_for_mu(mu)
    numerical_advantage = riskcal.analysis.conversions.get_advantage_from_pld(pld)
    assert analytic_advantage == pytest.approx(
        numerical_advantage, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


@pytest.mark.parametrize(
    "prior",
    [
        0.5,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_bayes_risk_from_gdp_matches_pld(prior):
    mu, pld, plrvs = get_gaussian_plrv_data()

    bayes_risk_from_pld = riskcal.analysis.conversions.get_bayes_risk_from_pld(
        pld, prior
    )
    bayes_risk_from_gdp = riskcal.analysis.conversions.get_bayes_risk_for_mu(mu, prior)
    assert bayes_risk_from_pld == pytest.approx(
        bayes_risk_from_gdp, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )

    # When the prior is 0.5, check that we correctly recover TV privacy.
    if not hasattr(prior, "__len__") and prior == 0.5:
        advantage = riskcal.analysis.conversions.get_advantage_from_gdp(mu)
        risk_from_advantage = 0.5 - 0.5 * advantage
        assert bayes_risk_from_pld == pytest.approx(
            risk_from_advantage, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
        )
        assert bayes_risk_from_gdp == pytest.approx(
            risk_from_advantage, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
        )


@pytest.mark.parametrize(
    "prior",
    [
        0.1,
        0.2,
        0.25,
        0.75,
        0.8,
        0.9,
    ],
)
def test_bayes_risk_is_symmetric(prior):
    mu, pld, plrvs = get_gaussian_plrv_data()

    numerical_risk1 = riskcal.analysis.conversions.get_bayes_risk_from_pld(pld, prior)
    numerical_risk2 = riskcal.analysis.conversions.get_bayes_risk_from_pld(
        pld, 1 - prior
    )
    assert numerical_risk1 == pytest.approx(
        numerical_risk2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )

    analytic_risk1 = riskcal.analysis.conversions.get_bayes_risk_for_mu(mu, prior)
    analytic_risk2 = riskcal.analysis.conversions.get_bayes_risk_for_mu(mu, 1 - prior)
    assert analytic_risk1 == pytest.approx(
        analytic_risk2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


@pytest.mark.parametrize(
    "prior",
    [
        0.5,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.75, 0.8, 0.9]),
    ],
)
def test_bayes_risk_from_pld_matches_kaissis_definition(prior):
    mu, pld, plrvs = get_gaussian_plrv_data()

    closed_form_risk_from_pld = riskcal.analysis.conversions.get_bayes_risk_from_pld(
        pld, prior
    )
    closed_form_risk_from_gdp = riskcal.analysis.conversions.get_bayes_risk_from_gdp(
        mu, prior
    )
    kaissis_risk = _get_bayes_risk_from_kaissis_definition(pld, prior)
    assert closed_form_risk_from_pld == pytest.approx(
        kaissis_risk, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )
    assert closed_form_risk_from_gdp == pytest.approx(
        kaissis_risk, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


@pytest.mark.parametrize("rho", list(np.linspace(0.005, 6, 10)))
def test_zcdp_beta_bounds_close_to_fitted_gdp(rho):
    tol = 0.025
    alpha = np.linspace(0, 1, 100)
    beta_gdp = riskcal.analysis.conversions.get_beta_from_zcdp_approx(
        rho=rho, alpha=alpha
    )
    beta_zcdp = riskcal.analysis.get_beta_from_zcdp(rho=rho, alpha=alpha)

    for i in range(len(alpha)):
        assert beta_gdp[i] == pytest.approx(beta_zcdp[i], abs=tol)


@pytest.mark.parametrize("rho", list(np.linspace(0.005, 6, 10)))
def test_zcdp_advantage_bounds_close_to_fitted_gdp(rho):
    tol = 0.025
    adv_gdp = riskcal.analysis.conversions.get_advantage_from_zcdp_approx(rho=rho)
    adv_zcdp = riskcal.analysis.get_advantage_from_zcdp(rho=rho)
    assert adv_gdp == pytest.approx(adv_zcdp, abs=tol)


def test_advantage_from_zcdp_decreases_with_smaller_rho():
    """Test that advantage decreases as privacy increases (rho decreases)."""
    rhos = [2.0, 1.0, 0.5, 0.1]
    advantages = [riskcal.analysis.get_advantage_from_zcdp(rho=rho) for rho in rhos]

    # Advantages should be in decreasing order
    for i in range(len(advantages) - 1):
        assert advantages[i] > advantages[i + 1], (
            f"Advantage should decrease as rho decreases: "
            f"rho={rhos[i]} gave {advantages[i]}, "
            f"rho={rhos[i+1]} gave {advantages[i+1]}"
        )

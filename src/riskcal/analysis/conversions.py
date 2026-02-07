"""Privacy risk metrics and conversions.

This module provides functions for computing privacy metrics (beta, advantage,
Bayes risk) from different privacy representations (PLDs, ADP, GDP,
RDP, zCDP).
"""

from typing import Union, Optional
import warnings
import numpy as np
from scipy import stats, optimize
from dp_accounting.pld import privacy_loss_distribution

from riskcal.analysis import _plrv
from riskcal.utils import _ensure_array


# =============================================================================
# Internal conversions
# =============================================================================


def pld_to_plrvs(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> _plrv.PLRVs:
    """
    Convert PLD to internal PLRVs representation.

    Converts a Google dp_accounting Privacy Loss Distribution object
    into the PLRVs (Privacy Loss Random Variables) format used internally.

    Args:
        pld: Privacy loss distribution from Google's dp_accounting library.

    Returns:
        PLRVs object containing the privacy loss random variables.

    Note:
        This is an internal conversion function. Most users should use
        the higher-level `get_beta_from_pld()`, `get_advantage_from_pld()`, etc.
    """

    def _get_plrv(pld):
        pld = pld.to_dense_pmf()
        pmf = pld._probs
        lower_loss = pld._lower_loss
        infinity_mass = pld._infinity_mass
        return lower_loss, infinity_mass, pmf

    lower_loss_Y, infinity_mass_Y, pmf_Y = _get_plrv(pld._pmf_remove)
    lower_loss_Z, infinity_mass_Z, pmf_Z = _get_plrv(pld._pmf_add)

    upper_loss_Z = lower_loss_Z + len(pmf_Z) - 1

    # clean pmfs. Sometimes float errors cause probs to be negative
    pmf_Y = np.where(pmf_Y < 0, 0, pmf_Y)
    pmf_Z = np.where(pmf_Z < 0, 0, pmf_Z)

    pmf_Y = pmf_Y * (1 - infinity_mass_Y) / np.sum(pmf_Y)
    pmf_Z = pmf_Z * (1 - infinity_mass_Z) / np.sum(pmf_Z)

    is_symmetric = pld._symmetric
    return _plrv.PLRVs(
        y0=lower_loss_Y,
        x0=-upper_loss_Z,
        pmf_Y=pmf_Y,
        pmf_X=pmf_Z[::-1],
        minus_infinity_mass_X=infinity_mass_Z,
        infinity_mass_Y=infinity_mass_Y,
        is_symmetric=is_symmetric,
    )


# Backward compatibility alias
plrvs_from_pld = pld_to_plrvs
plrvs_from_pld.__deprecated__ = "2.0.0"


# =============================================================================
# PLD (Privacy Loss Distribution)
# =============================================================================


def get_beta_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    alpha: Union[float, np.ndarray] = None,
    alphas: Union[float, np.ndarray] = None,  # Deprecated
) -> Union[float, np.ndarray]:
    """
    Compute false negative rate (FNR) for given false positive rate (FPR) from PLD.

    Uses the direct method from Algorithm 1 (Kulynych et al., 2024) to compute
    the optimal trade-off between FNR (beta) and FPR (alpha).

    .. deprecated:: 1.2.0
        Parameter 'alphas' is deprecated and will be removed in version 2.0.0.
        Use 'alpha' instead.

    Args:
        pld: Privacy loss distribution from Google's dp_accounting library.
        alpha: False positive rate(s) in [0, 1]. Can be scalar or array.
        alphas: (Deprecated) Use 'alpha' instead.

    Returns:
        False negative rate(s) corresponding to input alpha.

    Example:
        >>> from dp_accounting.pld import privacy_loss_distribution as pld_module
        >>> pld = pld_module.from_gaussian_mechanism(1.0)
        >>> beta = get_beta_from_pld(pld, alpha=0.01)

    References:
        Kulynych & Gomez et al. (2024), Algorithm 1. https://arxiv.org/abs/2407.02191
    """
    if alpha is None and alphas is None:
        raise ValueError("Must specify alpha.")
    elif alpha is not None and alphas is not None:
        raise ValueError("Must pass either alpha or alphas.")
    elif alphas is not None:
        warnings.warn(
            "Parameter 'alphas' is deprecated. Use 'alpha' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        alpha = alphas

    return _plrv.get_beta(pld_to_plrvs(pld), alpha)


def get_advantage_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> float:
    """
    Compute attack advantage from PLD.

    Advantage is the maximum value of (TPR - FPR) achievable by any attacker,
    which equals delta at epsilon=0.

    Args:
        pld: Privacy loss distribution from Google's dp_accounting library.

    Returns:
        Maximum attack advantage.

    References:
        Kulynych & Gomez et al. (2024). https://arxiv.org/abs/2407.02191
    """
    return pld.get_delta_for_epsilon(0)


def get_bayes_risk_from_pld(pld, prior):
    """
    Compute Bayes Risk from PLD for given prior probability.

    Bayes risk shows the maximum accuracy of an attack against privacy of a single
    record under a binary prior (e.g., accuracy of attribute inference).

    Args:
        pld: Privacy loss distribution from Google's dp_accounting library.
        prior: Prior probability (scalar or array). Probability that the
            sensitive attribute takes value 1.

    Returns:
        Bayes risk value(s). Float if prior is scalar, array if prior is array.

    Example:
        >>> from dp_accounting.pld import privacy_loss_distribution as pld_module
        >>> pld = pld_module.from_laplace_mechanism(1.0)
        >>> risk = get_bayes_risk_from_pld(pld, prior=0.5)

    References:
        Kulynych et al. (2025), Proposition D.1 / Eq. 35.
        https://arxiv.org/abs/2507.06969
    """
    prior, is_scalar = _ensure_array(prior)
    epsilon = np.log(prior / (1 - prior))
    delta = np.array([pld.get_delta_for_epsilon(e) for e in epsilon])
    bayes_risk = (1 - prior) * (1 - delta)

    if is_scalar:
        return bayes_risk[0]
    return np.array(bayes_risk)


# =============================================================================
# GDP (Gaussian Differential Privacy)
# =============================================================================


def get_beta_from_gdp(
    mu: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute FNR for FPR using the analytical formula for Gaussian DP.

    Args:
        mu: Gaussian noise scale parameter (sigma).
        alpha: False positive rate(s) in [0, 1].

    Returns:
        False negative rate(s) corresponding to input alpha.

    References:
        Dong et al. (2019), Eq. 6. https://arxiv.org/abs/1905.02383
    """
    return stats.norm.cdf(-stats.norm.ppf(alpha) - mu)


def get_advantage_from_gdp(mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute attack advantage using analytical formula for Gaussian mechanism.

    Args:
        mu: Gaussian noise scale parameter (sigma). Can be scalar or array.

    Returns:
        Attack advantage value(s).

    References:
        Dong et al. (2019), Corollary 2.13. https://arxiv.org/abs/1905.02383
    """
    return stats.norm.cdf(mu / 2) - stats.norm.cdf(-mu / 2)


def get_bayes_risk_from_gdp(mu, prior):
    """
    Compute Bayes Risk for Gaussian mechanism using analytical formula.

    Args:
        mu: Gaussian noise scale parameter (sigma).
        prior: Prior probability (scalar or array).

    Returns:
        Bayes risk value(s). Float if prior is scalar, array if prior is array.

    References:
        Kulynych et al. (2025), Proposition D.1 / Eq. 35.
        https://arxiv.org/abs/2507.06969
        Dong et al. (2019), Corollary 2.13. https://arxiv.org/abs/1905.02383
    """
    assert mu >= 0, "mu must be >= 0"
    prior, is_scalar = _ensure_array(prior)
    epsilon = np.log(prior / (1 - prior))
    # Gaussian privacy profile: Dong et al. (2019), Corollary 2.13
    delta = stats.norm.cdf(-epsilon / mu + mu / 2) - np.exp(epsilon) * stats.norm.cdf(
        -epsilon / mu - mu / 2
    )
    bayes_risk = (1 - prior) * (1 - delta)

    if is_scalar:
        return bayes_risk[0]
    return np.array(bayes_risk)


# Backward compatibility aliases
get_beta_for_mu = get_beta_from_gdp
get_beta_for_mu.__deprecated__ = "2.0.0"
get_advantage_for_mu = get_advantage_from_gdp
get_advantage_for_mu.__deprecated__ = "2.0.0"
get_bayes_risk_for_mu = get_bayes_risk_from_gdp
get_bayes_risk_for_mu.__deprecated__ = "2.0.0"


# =============================================================================
# ADP (Approximate Differential Privacy)
# =============================================================================


def get_beta_from_adp(
    epsilon: float, delta: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute FNR for FPR from (epsilon, delta)-DP parameters.

    Args:
        epsilon: Privacy parameter epsilon.
        delta: Privacy parameter delta.
        alpha: False positive rate(s) in [0, 1].

    Returns:
        False negative rate(s) corresponding to input alpha.

    Example:
        >>> import numpy as np
        >>> np.round(get_beta_from_adp(1.0, 0.001, 0.8), 3)
        0.073

    References:
        Dong et al. (2019), Eq. 5. https://arxiv.org/abs/1905.02383
    """
    form1 = np.array(1 - delta - np.exp(epsilon) * alpha)
    form2 = np.array(np.exp(-epsilon) * (1 - delta - alpha))
    return np.maximum.reduce([form1, form2, np.zeros_like(form1)])


def get_advantage_from_adp(epsilon: float, delta: float) -> float:
    """
    Compute attack advantage from (epsilon, delta)-DP parameters.

    Args:
        epsilon: Privacy parameter epsilon.
        delta: Privacy parameter delta.

    Returns:
        Attack advantage.

    Example:
        >>> import numpy as np
        >>> np.round(get_advantage_from_adp(0., 0.001), 3)
        0.001

    References:
        Dong et al. (2019). https://arxiv.org/abs/1905.02383
    """
    return (np.exp(epsilon) + 2 * delta - 1) / (np.exp(epsilon) + 1)


def get_epsilon_from_err_rates(delta: float, alpha: float, beta: float) -> float:
    """
    Convert f-DP error rates (alpha, beta) to epsilon for (epsilon, delta)-DP.

    Args:
        delta: Target delta parameter for (epsilon, delta)-DP.
        alpha: False positive rate (FPR) from f-DP.
        beta: False negative rate (FNR) from f-DP.

    Returns:
        Epsilon value corresponding to the error rates.

    Example:
        >>> import numpy as np
        >>> np.round(get_epsilon_from_err_rates(0.001, 0.001, 0.8), 3)
        5.293

    References:
        Dong et al. (2019). https://arxiv.org/abs/1905.02383
    """
    epsilon1 = np.log((1 - delta - alpha) / beta)
    epsilon2 = np.log((1 - delta - beta) / alpha)
    return np.maximum.reduce([epsilon1, epsilon2, np.zeros_like(epsilon1)])


# Backward compatibility aliases
get_beta_for_epsilon_delta = get_beta_from_adp
get_beta_for_epsilon_delta.__deprecated__ = "2.0.0"
get_advantage_for_epsilon_delta = get_advantage_from_adp
get_advantage_for_epsilon_delta.__deprecated__ = "2.0.0"
get_epsilon_for_err_rates = get_epsilon_from_err_rates
get_epsilon_for_err_rates.__deprecated__ = "2.0.0"


# =============================================================================
# RDP (Renyi Differential Privacy)
# =============================================================================


def _check_renyi_constraints(epsilon: float, y: float, x: float, order: float) -> bool:
    # In the Zhu et al.'s notation:
    # alpha (from input) -> x (Type I error)
    # beta (the return value) -> y (Type II error)
    # order -> alpha (Rényi divergence order)

    # Precompute terms in log space
    logx = np.log(x)
    log1mx = np.log1p(-x)
    logy = np.log(y)
    log1my = np.log1p(-y)

    # Case where order != 1
    if order != 1:
        upper = (order - 1) * epsilon
        sign_order = np.sign(order - 1)
        one_minus_order = 1 - order

        log_f2 = np.logaddexp(
            order * logx + one_minus_order * log1my,
            order * log1mx + one_minus_order * logy,
        )
        constraint2 = sign_order * (upper - log_f2) >= 0

        log_f1 = np.logaddexp(
            order * log1my + one_minus_order * logx,
            order * logy + one_minus_order * log1mx,
        )
        constraint1 = sign_order * (upper - log_f1) >= 0

    # Case where order == 1
    else:
        upper = epsilon
        f1 = x * (logx - log1my) + (1 - x) * (log1mx - logy)
        constraint1 = upper - f1 >= 0
        f2 = y * (logy - log1mx) + (1 - y) * (log1my - logx)
        constraint2 = upper - f2 >= 0

    return constraint1 and constraint2


def get_beta_from_rdp(
    epsilon: float,
    alpha: float,
    order: float,
    linear_search_step: float = 1e-3,
    max_bisection_steps: int = 50,
    tol: float = 1e-5,
) -> float:
    """
    Compute FNR for FPR from Renyi DP parameters.

    Uses linear search to find a feasible region, then bisection search to
    refine the optimal beta (FNR) for a given alpha (FPR). This directly
    implements the hypothesis testing trade-off for Renyi DP.

    Args:
        epsilon: Renyi divergence parameter (privacy budget).
        alpha: False positive rate (FPR) in [0, 1].
        order: Order of Renyi divergence (alpha in Renyi DP literature).
        linear_search_step: Step size for linear search phase. Smaller values
            give more precision but slower execution.
        max_bisection_steps: Maximum iterations for bisection refinement.
        tol: Numerical tolerance for convergence and avoiding log(0).

    Returns:
        False negative rate (FNR) corresponding to input alpha.

    Example:
        >>> import numpy as np
        >>> # Renyi DP with epsilon=1.0 at order 2
        >>> beta = get_beta_from_rdp(epsilon=1.0, alpha=0.1, order=2.0)
        >>> np.round(beta, 3)
        0.507

    References:
        Zhu et al. (2022), Appendix F.1. https://arxiv.org/abs/2106.08567
    """
    # Use linear search to find the feasible region boundaries
    # Search from below (beta=0 upward)
    beta1 = tol  # Start slightly above 0 to avoid log(0)
    while beta1 < 1 and not _check_renyi_constraints(epsilon, beta1, alpha, order):
        beta1 += linear_search_step
    beta1 = min(beta1, 1.0)

    # Search from above (beta=1 downward)
    beta2 = 1 - tol  # Start slightly below 1 to avoid log(0)
    while beta2 > 0 and not _check_renyi_constraints(epsilon, beta2, alpha, order):
        beta2 -= linear_search_step
    beta2 = max(beta2, 0.0)

    # Set bisection bracket
    beta_low = min(beta1, beta2)
    beta_high = max(beta1, beta2)

    # Ensure we have a valid bracket
    if beta_low >= beta_high:
        # Check edge cases
        if _check_renyi_constraints(epsilon, beta_low, alpha, order):
            return beta_low
        elif _check_renyi_constraints(epsilon, beta_high, alpha, order):
            return beta_high
        else:
            return 0.0

    # Bisection to find the minimum beta that satisfies constraints
    for _ in range(max_bisection_steps):
        if np.abs(beta_high - beta_low) <= tol:
            break
        beta_mid = (beta_low + beta_high) / 2
        if _check_renyi_constraints(epsilon, beta_mid, alpha, order):
            beta_high = beta_mid
        else:
            beta_low = beta_mid

    # Return a conservative solution
    if _check_renyi_constraints(epsilon, beta_low, alpha, order):
        return beta_low
    elif _check_renyi_constraints(epsilon, beta_high, alpha, order):
        return beta_high
    else:
        return 0.0


# =============================================================================
# zCDP (Zero-Concentrated Differential Privacy)
# =============================================================================


def get_beta_from_zcdp(
    rho: float,
    alpha: Union[float, np.ndarray],
    max_bisection_steps: int = 100,
    linear_search_step: float = 5e-4,
    tol: float = 1e-4,
    max_order: Optional[float] = None,
    order_grid_size: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Compute FNR for FPR from zCDP parameter rho.

    Zero-Concentrated Differential Privacy (zCDP) is characterized by a single
    parameter rho. This function computes the optimal trade-off between false
    negative rate (beta) and false positive rate (alpha) by optimizing over
    Renyi divergence orders.

    Args:
        rho: Zero-concentrated differential privacy parameter.
        alpha: False positive rate(s) in [0, 1]. Can be scalar or array.
        max_bisection_steps: Maximum number of bisection iterations for beta search.
            Increase for higher precision at the cost of computation time.
        linear_search_step: Step size for linear search to find feasible region.
            Smaller values give more precision but slower execution.
        tol: Tolerance for numerical convergence and avoiding log(0).
        max_order: Maximum Renyi DP order to consider. If None, chosen adaptively
            based on rho (higher orders for smaller rho values).
        order_grid_size: Number of grid points for initial order search. If None,
            chosen adaptively based on rho.

    Returns:
        False negative rate(s) corresponding to input alpha. Returns float if
        alpha is scalar, array if alpha is array.

    Example:
        >>> import numpy as np
        >>> # Single alpha value
        >>> beta = get_beta_from_zcdp(rho=0.5, alpha=0.1)
        >>> np.round(beta, 3)
        0.517
        >>> # Multiple alpha values
        >>> betas = get_beta_from_zcdp(rho=0.5, alpha=np.array([0.1, 0.2, 0.3]))
        >>> np.round(betas, 3)
        array([0.517, 0.34, 0.232])

    References:
        Bun & Steinke (2016). https://arxiv.org/abs/1605.02065
        Zhu et al. (2022), Appendix F.1. https://arxiv.org/abs/2106.08567
    """
    HIGH_ORDER_THRESH = 0.05
    LOW_MAX_ORDER = 20
    HIGH_MAX_ORDER = 50

    HIGH_GRID_THRESH = 1.5
    HIGH_GRID_SIZE = 100
    LOW_GRID_SIZE = 10

    beta = []
    for alpha_val in np.atleast_1d(alpha):
        if alpha_val <= tol:
            beta.append(1.0)
        elif alpha_val >= 1 - tol:
            beta.append(0.0)
        else:
            # Adaptive order optimization heuristics
            if max_order is None:
                if rho <= HIGH_ORDER_THRESH:
                    max_order = HIGH_MAX_ORDER
                else:
                    max_order = LOW_MAX_ORDER

            if order_grid_size is None:
                if rho <= HIGH_GRID_THRESH:
                    order_grid_size = LOW_GRID_SIZE
                else:
                    order_grid_size = HIGH_GRID_SIZE

            # Initial coarse grid search
            orders = np.logspace(np.log10(0.5), np.log10(max_order), order_grid_size)
            betas_for_orders = np.array(
                [
                    get_beta_from_rdp(
                        epsilon=rho * order,
                        alpha=alpha_val,
                        order=order,
                        linear_search_step=linear_search_step,
                        max_bisection_steps=max_bisection_steps,
                        tol=tol,
                    )
                    for order in orders
                ]
            )

            # Find the order that gives maximum beta
            best_idx = np.argmax(betas_for_orders)
            best_beta = betas_for_orders[best_idx]

            # Refinement: use bounded optimization around the best order
            if best_idx > 0 and best_idx < len(orders) - 1:
                order_low = orders[best_idx - 1]
                order_high = orders[best_idx + 1]

                # Try bounded optimization to refine
                result = optimize.minimize_scalar(
                    lambda order: -get_beta_from_rdp(
                        epsilon=rho * order,
                        alpha=alpha_val,
                        order=order,
                        linear_search_step=linear_search_step,
                        max_bisection_steps=max_bisection_steps,
                        tol=tol,
                    ),
                    bounds=(order_low, order_high),
                    method="bounded",
                    options={"xatol": tol},
                )

                if result.success and (abs(result.x - max_order) <= tol):
                    warnings.warn(
                        "Optimal order is close to the maximum order. Consider increasing max_order"
                    )

                # Only use optimization result if it's successful AND better than grid search
                if result.success and -result.fun >= best_beta:
                    best_beta = -result.fun

            beta.append(best_beta)

    return beta[0] if isinstance(alpha, (int, float)) else np.array(beta)


def get_advantage_from_zcdp(
    rho: float,
    max_bisection_steps: int = 100,
    linear_search_step: float = 5e-4,
    tol: float = 1e-4,
    max_order: Optional[float] = None,
    order_grid_size: Optional[int] = None,
) -> float:
    """
    Compute attack advantage from zCDP parameter rho.

    Advantage is the maximum value of (TPR - FPR) achievable by any attacker.
    This function optimizes over all possible threshold choices to find the
    maximum advantage.

    Args:
        rho: Zero-concentrated differential privacy parameter.
        max_bisection_steps: Maximum number of bisection iterations for beta search.
        linear_search_step: Step size for linear search to find feasible region.
        tol: Tolerance for convergence.
        max_order: Max Renyi DP order. If None, choose based on a heuristic.
        order_grid_size: Number of grid points for initial order search. If None,
            choose based on a heuristic.

    Returns:
        Maximum attack advantage.

    Example:
        >>> import numpy as np
        >>> adv = get_advantage_from_zcdp(rho=0.5)
        >>> np.round(adv, 3)
        0.47

    References:
        Bun & Steinke (2016). https://arxiv.org/abs/1605.02065
    """
    result = optimize.minimize_scalar(
        lambda alpha: -(
            1
            - alpha
            - get_beta_from_zcdp(
                rho=rho,
                alpha=alpha,
                max_bisection_steps=max_bisection_steps,
                linear_search_step=linear_search_step,
                tol=tol,
                max_order=max_order,
                order_grid_size=order_grid_size,
            )
        ),
        bounds=(0, 1),
        method="bounded",
    )
    if not result.success:
        warnings.warn("Optimization failed for advantage calculation")
        return np.nan
    else:
        return -result.fun


def get_mu_from_zcdp_approx(rho: float) -> float:
    """
    Approximate mapping from zCDP parameter rho to GDP parameter mu.

    Uses a fitted polynomial approximation to convert zCDP rho to an equivalent
    GDP mu parameter. This approximation is much faster than exact conversion
    methods while maintaining high accuracy.

    Args:
        rho: Zero-concentrated differential privacy parameter.

    Returns:
        Approximate GDP parameter mu.

    Example:
        >>> import numpy as np
        >>> mu = get_mu_from_zcdp_approx(rho=0.5)
        >>> np.round(mu, 3)
        1.214

    Note:
        Fitted for rho values in [0.005, 8.5]. Maximum pointwise difference
        in resulting trade-off curve is <0.025 for FNR values >0.01.

    References:
        Bun & Steinke (2016). https://arxiv.org/abs/1605.02065
        Dong et al. (2019), Corollary 2.13. https://arxiv.org/abs/1905.02383
    """
    return (
        1.5822558654881096 * np.sqrt(rho)
        + 0.08064620797681155 * rho
        + 0.05485600531538526
    )


def get_beta_from_zcdp_approx(
    rho: float,
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Approximate FNR for FPR from zCDP parameter rho using GDP approximation.

    Computes the trade-off curve by first approximating zCDP with GDP, then
    using the analytical GDP formula. This is much faster than exact zCDP
    computation while maintaining high accuracy.

    Args:
        rho: Zero-concentrated differential privacy parameter.
        alpha: False positive rate(s) in [0, 1]. Can be scalar or array.

    Returns:
        False negative rate(s) corresponding to input alpha. Returns float if
        alpha is scalar, array if alpha is array.

    Example:
        >>> import numpy as np
        >>> # Single alpha value
        >>> beta = get_beta_from_zcdp_approx(rho=0.5, alpha=0.1)
        >>> np.round(beta, 3)
        0.527
        >>> # Multiple alpha values
        >>> betas = get_beta_from_zcdp_approx(rho=0.5, alpha=np.array([0.1, 0.2, 0.3]))
        >>> np.round(betas, 3)
        array([0.527, 0.355, 0.245])

    Note:
        Fitted for rho values in [0.005, 8.5]. Maximum pointwise error
        is <0.025 for FNR values >0.01. For higher accuracy, use
        `get_beta_from_zcdp()`.

    References:
        Bun & Steinke (2016). https://arxiv.org/abs/1605.02065
        Dong et al. (2019), Eq. 6. https://arxiv.org/abs/1905.02383
    """
    return get_beta_from_gdp(get_mu_from_zcdp_approx(rho), alpha=alpha)


def get_advantage_from_zcdp_approx(
    rho: float,
) -> float:
    """
    Approximate attack advantage from zCDP parameter rho using GDP approximation.

    Computes advantage by first approximating zCDP with GDP, then using the
    analytical GDP formula. This is much faster than exact zCDP computation
    while maintaining high accuracy.

    Args:
        rho: Zero-concentrated differential privacy parameter.

    Returns:
        Maximum attack advantage.

    Example:
        >>> import numpy as np
        >>> adv = get_advantage_from_zcdp_approx(rho=0.5)
        >>> np.round(adv, 3)
        0.456

    Note:
        Fitted for rho values in [0.005, 8.5]. Approximation error is
        typically <0.025. For higher accuracy, use `get_advantage_from_zcdp()`.

    References:
        Bun & Steinke (2016). https://arxiv.org/abs/1605.02065
        Dong et al. (2019), Corollary 2.13. https://arxiv.org/abs/1905.02383
    """
    return get_advantage_from_gdp(get_mu_from_zcdp_approx(rho))

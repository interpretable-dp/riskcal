# imports for python
import numpy as np
import math
import warnings
from typing import List, Union

# Replacement for libc.math and cython types
from math import fabs, pow, exp, log, log1p

INFINITY = float("inf")

ArrayLike = Union[np.ndarray, List[float]]

"""
Notation:

    FPR = false positive rate
    FNR = false negative rate
    alpha = another name for FPR
    beta = another name for FNR
    order = Renyi order
    epsilon = value of Renyi divergence


The main python facing function is get_FNR.
This function takes as input an single
(order, epsilon)-RDP guarantee and a single (or multiple)
FPR values alpha and returns the corresponding FNR values of
the tradeoff curve

The function does this by calling one of 3 cython functions,
each optimized for a particular regime of Renyi order (order).
They are:

get_FNR_case1_vec: handles order > 1
get_FNR_case2_vec: handles order = 1
get_FNR_case3_vec: handles 0 < order < 1

Each of these functions perform a for loop (in cython) over
the list/array of given FPR values. For each FPR value, the functions
call a correspinding _core_logic function, which implements the actual
conversion. The conversion is itself a root-finding problem, and various
numerical techniques are used to speed up the compuation. The main doc
string that explains these techniques are in _core_logic_case1.
The doc strings of all other core logic functions only mention techniques
different from those in _core_logic_case1.
"""


###############
###############
### CASE 1 ####
###############
###############
def get_fixed_point_case1(order, upper, tol=1e-7):
    """
    See _core_logic_case1 doc string for context on why we need this
    function.

    Assuming order > 1, this function returns the fixed point of the function beta(alpha)
    implicitly defined via:

    (1-alpha)^{1-order} beta^order     + alpha^{1-order} (1-beta)^order     <= upper
    (1-alpha)^{order}   beta^{1-order} + alpha^{order}   (1-beta)^{1-order} <= upper

    which can be easily seen to be the smallest value of beta such that:

    (1-beta)^{order} beta^{1-order} + beta^{order} (1-beta)^{1-order} <= upper

    Let the lhs of this equation by F(beta). F can be shown to be convex and
    decreasing. Since we know this function computes the fixed point of a
    tradeoff function, we know that 0 <= beta <= 1/2.

    Therefore, this function uses bisection over [0,1/2] to find F(beta) >= upper.
    Let beta* denote the true inverse of F at upper. It is guaranteed that
    beta* < beta < beta* + tol thanks to bisection. Since this function is not called
    as often as _core_logic_case1, it is not as performance critical, and we
    intentionally use standard bisection.
    """

    # init [low, high]  for bisection
    low = 0
    high = 1 / 2

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = 0.5 * (low + high)
        F_mid = (1 - mid) ** (1 - order) * mid**order + mid ** (1 - order) * (
            1 - mid
        ) ** order

        if F_mid > upper:  # True means "move low up"
            low = mid
        else:  # False means "move high down"
            high = mid

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def _core_logic_case1(alpha, upper, order, one_minus_order, fixed_point, tol):
    """
    For fixed RDP guarantee (order, epsilon) with order > 1 and FPR alpha,
    this function returns the smallest value of beta in [0,1] that simultaneously satisfies:

    (1-alpha)^{1-order} beta^order     + alpha^{1-order} (1-beta)^order     <= upper
    (1-alpha)^{order}   beta^{1-order} + alpha^{order}   (1-beta)^{1-order} <= upper

    Let lhs1 denote the lhs of the first inequality, and lhs2 denote the second.
    Let M(beta) = max(lhs1, lhs2). The above two inequalities simplify to M(beta) <= upper.
    Let fixed_point denote the unique value such that (alpha = fixed_point, beta = fixed_point)
    satisfies the two inequalities above. It can be shown that M(beta) = lhs1(beta) if alpha < fixed_point
    and M(beta) = lhs2(beta) if alpha > fixed_point. The code below uses this fact to focus on
    only one of the two inequalities above, depending on the value of FPR alpha.

    Note that if alpha = 0, then only beta = 1 satisfies the inequalities. And if alpha = 1, then all values
    of beta satisfy the inequalities, therefore we choose the smallest: beta = 0. These two edge
    cases are hard coded into the function.

    M(beta) is a convex decreasing function of beta, and our goal is to return the smallest
    value of beta such that M(beta) <= upper. Let the result of this be denoted beta = f(alpha).
    Since we know this is a tradeoff curve, we have that 0 <= beta <= 1 - alpha.
    Therefore, we can use bisection over [0, 1-alpha] to numerically invert M(beta) <= upper.

    Let beta* denote the true inverse of M at upper, i.e. M(beta*) = upper. Let beta denote the
    output of this function. This function is guaranteed to return beta such that
    beta* < beta < beta* + tol.

    This function implements one more trick for speed: since M is convex and decreasing,
    using Newton's Method is guaranteed to yield a lower bound on beta*, and using the secant
    method is guaranteed to yield an upper bound on beta*. Therefore, we use both techniques to
    improve the bisection interval when possible.

    Args:
        alpha: FPR value
        upper: exp( (order-1) * epsilon). The value of the upper bound
        order: Renyi order
        one_minus_order: one minus the Renyi order
        fixed_point: fixed point of the tradeoff curve
        tol: tolerance. See doc string for meaning

    Returns:
        beta: FNR at FPR
    """

    # handle alpha = 0 or alpha = 1 explicitly
    if alpha == 0:
        return 1.0
    elif alpha == 1:
        return 0.0

    # save variables to reduce computation of M for later
    one_minus_alpha = 1.0 - alpha
    c1 = 0.0
    c2 = 0.0
    expo = 0.0

    if alpha <= fixed_point:
        expo = order
        c1 = pow(one_minus_alpha, one_minus_order)
        c2 = pow(alpha, one_minus_order)
    else:
        expo = one_minus_order
        c1 = pow(one_minus_alpha, order)
        c2 = pow(alpha, order)

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_alpha

    # M_low, M_high below assume low = 0 and high = 1 - alpha
    # if alpha <= fixed_point, expo > 0 and M_low is finite, if expo < 0, M_low is infinite
    M_low = c2 if alpha <= fixed_point else INFINITY

    # M_high: cython.double = c1 * pow(high, expo) + c2 * pow(1 - high, expo)
    M_high = c1 * pow(high, expo) + c2 * pow(alpha, expo)

    # init all other vars for bisection
    mid = 0.5 * (high - low)
    one_minus_mid = 0.0
    term1 = 0.0
    term2 = 0.0
    M_mid = 0.0
    grad = 0.0
    newton = 0.0
    secant = 0.0

    # begin bisection
    while (high - low) > tol:
        one_minus_mid = 1.0 - mid

        # M(mid) = term1 + term2
        term1 = c1 * pow(mid, expo)  # used in Newton calc
        term2 = c2 * pow(one_minus_mid, expo)  # used in Newton calc
        M_mid = term1 + term2

        # update bisection interval [low, high]
        if M_mid > upper:  # True means "move low up"
            low = mid
            M_low = M_mid  # used in Secant calc
        else:  #  False means "move high down"
            high = mid
            M_high = M_mid  # used in Secant calc

        # attempt to use Newton's method to increase low
        grad = expo * (term1 / mid - term2 / one_minus_mid)
        newton = mid - (M_mid - upper) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < beta* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton
            M_low = c1 * pow(low, expo) + c2 * pow(1.0 - low, expo)

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15:  # avoid division by 0
            secant = low + (upper - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < beta* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant
                M_high = c1 * pow(high, expo) + c2 * pow(1.0 - high, expo)

        # update mid for next iteration of bisection
        mid = 0.5 * (low + high)

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def get_FNR_case1_vec(alpha_array, order, epsilon, tol=1e-7):

    # compute upper and one_minus_order once to save compute in core_logic
    log_upper = (order - 1.0) * epsilon
    upper = exp(log_upper)
    one_minus_order = 1.0 - order

    # compute fixed point (hard code 1e-15 for stability)
    fixed_point = get_fixed_point_case1(order, upper, tol=1e-15)

    # allocate output array
    n = alpha_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case1(
            alpha_array[i], upper, order, one_minus_order, fixed_point, tol
        )

    return result_np


###############
###############
### CASE 2 ####
###############
###############
def get_fixed_point_case2(epsilon, tol=1e-7):
    """
    See _core_logic_case1 and get_fixed_point_case1 doc string for context on
    why we need this function. Only difference from get_fixed_point_case1 is
    that here, the fixed point equation is:

    beta log( beta/(1-beta) ) + (1-beta) log( (1-beta)/beta) <= epsilon

    The same bisection algo used in get_fixed_point_case1 is used here
    """

    # init [low, high]  for bisection
    low = 0
    high = 1 / 2

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = 0.5 * (low + high)
        F_mid = mid * (log(mid) - log1p(-mid)) + (1 - mid) * (log1p(-mid) - log(mid))
        if F_mid > epsilon:  # True means "move low up"
            low = mid
        else:  # False means "move high down"
            high = mid

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def _core_logic_case2(alpha, epsilon, fixed_point, tol):
    """
    See _core_logic_case1 doc string for details. The only difference
    between case1 and this function is that here, order = 1,
    where the inequalities are

    beta log( beta/(1-alpha) ) + (1-beta) log( (1-beta)/alpha) <= epsilon
    alpha log( alpha/(1-beta) ) + (1-alpha) log( (1-alpha)/beta) <= epsilon

    Still, M(beta) defined as in _core_logic_case1 is convex and decreasing,
    so the implementation of bisection + Newton + Secant is the same.
    """
    # handle alpha = 0 or alpha = 1 explicitly
    if alpha == 0:
        return 1.0
    elif alpha == 1:
        return 0.0

    # save variables to reduce computation of M for later
    one_minus_alpha = 1.0 - alpha
    log_one_minus_alpha = log1p(-alpha)
    log_alpha = log(alpha)

    less_than_fixed_point = alpha <= fixed_point

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_alpha

    M_low = 0.0
    M_high = 0.0

    if less_than_fixed_point:
        # M_low = M(0) = -log(alpha) (limit as beta -> 0+)
        M_low = -log_alpha
        M_high = high * (log(high) - log_one_minus_alpha) + (1.0 - high) * (
            log1p(-high) - log_alpha
        )
    else:
        # M(0) = +inf (because -(1-alpha)*log(beta) term blows up)
        M_low = INFINITY
        M_high = alpha * (log_alpha - log1p(-high)) + one_minus_alpha * (
            log_one_minus_alpha - log(high)
        )

    # begin bisection
    mid = 0.5 * (low + high)
    one_minus_mid = 0.0
    M_mid = 0.0
    grad = 0.0
    newton = 0.0
    secant = 0.0

    while (high - low) > tol:
        one_minus_mid = 1.0 - mid

        # compute M(mid) and M'(mid)
        if less_than_fixed_point:
            M_mid = mid * (log(mid) - log_one_minus_alpha) + one_minus_mid * (
                log1p(-mid) - log_alpha
            )
            grad = (log(mid) - log_one_minus_alpha) - (log1p(-mid) - log_alpha)
        else:
            M_mid = alpha * (log_alpha - log1p(-mid)) + one_minus_alpha * (
                log_one_minus_alpha - log(mid)
            )
            grad = (mid - one_minus_alpha) / (mid * one_minus_mid)

        # update bisection interval [low, high]
        if M_mid > epsilon:  # True means "move low up"
            low = mid
            M_low = M_mid  # used in Secant calc
        else:  #  False means "move high down"
            high = mid
            M_high = M_mid  # used in Secant calc

        # attempt to use Newton's method to increase low
        newton = mid - (M_mid - epsilon) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < beta* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton

            # recompute M_low
            if less_than_fixed_point:
                M_low = low * (log(low) - log_one_minus_alpha) + (1.0 - low) * (
                    log1p(-low) - log_alpha
                )
            else:
                M_low = alpha * (log_alpha - log1p(-low)) + one_minus_alpha * (
                    log_one_minus_alpha - log(low)
                )

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15:  # avoid division by 0
            secant = low + (epsilon - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < beta* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant

                # recompute M_high
                if less_than_fixed_point:
                    M_high = high * (log(high) - log_one_minus_alpha) + (1.0 - high) * (
                        log1p(-high) - log_alpha
                    )
                else:
                    M_high = alpha * (log_alpha - log1p(-high)) + one_minus_alpha * (
                        log_one_minus_alpha - log(high)
                    )

        # Next bisection midpoint
        mid = 0.5 * (low + high)

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def get_FNR_case2_vec(alpha_array, epsilon, tol=1e-7):

    # compute fixed point (hard code 1e-15 tol for stability)
    fixed_point = get_fixed_point_case2(epsilon, tol=1e-15)

    # allocate output array
    n = alpha_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core_logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case2(alpha_array[i], epsilon, fixed_point, tol)

    return result_np


###############
###############
### CASE 3 ####
###############
###############
def get_fixed_point_case3(order, lower, alpha_max, tol=1e-7):
    """
    See _core_logic_case1 and get_fixed_point_case1 doc string for context on
    why we need this function. Only difference from get_fixed_point_case1 is
    that here, the fixed point equation is:

    (1-beta)^{order} beta^{1-order} + beta^{order} (1-beta)^{1-order} >= lower

    Since order < 1 for case 3, now F(beta) (the lhs of the inequality) is
    increasing and concave, as opposed to decreasing and convex as in case 1.
    Moreover, in get_fixed_point_case1 all we knew was the the solution had
    to be between [0,1/2]. Here, we know that the fixed point must be less than
    alpha_max, so we search between [0,alpha_max].

    The code below returns a value of beta that is guaranteed to have the property
    F(beta) >= lower. It does this by bisection. Let beta* denote the true inverse of
    F at lower. It is guaranteed that beta* < beta < beta* + tol.
    """

    # init [low, high]  for bisection
    low = 0
    high = alpha_max

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = (low + high) / 2.0
        F_mid = (1 - mid) ** (1 - order) * mid**order + mid ** (1 - order) * (
            1 - mid
        ) ** order
        if F_mid > lower:
            high = mid  # True means "move high down"
        else:
            low = mid  # False means "move low up"

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def _core_logic_case3(
    alpha, lower, order, one_minus_order, fixed_point, alpha_max, tol
):
    """
    See _core_logic_case1 doc string for details. The only difference
    between case 1 and this function is that here, order < 1,
    where the inequalities are

    (1-alpha)^{1-order} beta^order     + alpha^{1-order} (1-beta)^order     >= lower
    (1-alpha)^{order}   beta^{1-order} + alpha^{order}   (1-beta)^{1-order} >= lower

    In particular, they are identical except with the inequality fliped.
    This has a much bigger impact than one may imagine. For example, M(beta) is
    now concave and increasing since we need to take the min(lhs1, lhs2) and
    not the max. Note that in case 1 and 2, M(beta) is convex and decreasing.
    More importantly, however, is that for order < 1, the inequalities
    have very different behaviors at alpha = 0.

    At alpha = 0, the inequality simplifies to beta >= e^{ (order-1) / order * epsilon }.
    Therefore, the smallest beta must be e^{ (order-1) / order * epsilon }, which is not 1.

    Due to symmetry, the same effect occurs at beta = 0, in that we get
    alpha >= e^{ (order-1) / order * epsilon }. This leads to the somewhat counter intuitive
    situation where if the input alpha satisfies alpha >= e^{ (order-1) / order * epsilon }, then
    it must be that beta = 0. This is hard coded into the code below (alpha_max is this value).

    The above implies the tradeoff function for RDP when order < 1 has the property
    f(0) < 1 and f(alpha') = 0 for alpha' in some non zero interval that contains 1. This
    implies this tradeoff function catastrophically fails (i.e. has infinite privacy loss).
    We leave further exploration of this to future work.
    """

    # handle alpha = 0 and alpha >= alpha_max explicitly
    if alpha == 0:
        return alpha_max
    elif alpha >= alpha_max:
        return 0.0

    # save variables to reduce computation of M for later
    one_minus_alpha = 1.0 - alpha
    c1 = 0.0
    c2 = 0.0
    expo = 0.0

    if alpha <= fixed_point:
        expo = order
        c1 = pow(one_minus_alpha, one_minus_order)
        c2 = pow(alpha, one_minus_order)
    else:
        expo = one_minus_order
        c1 = pow(one_minus_alpha, order)
        c2 = pow(alpha, order)

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_alpha

    # two lines below assume low = 0 and high = 1 - alpha
    M_low = 0
    M_high = c1 * pow(high, expo) + c2 * pow(alpha, expo)

    # init all other vars for bisection
    mid = 0.5 * (high - low)
    one_minus_mid = 0.0
    term1 = 0.0
    term2 = 0.0
    M_mid = 0.0
    grad = 0.0
    newton = 0.0
    secant = 0.0

    # begin bisection
    while (high - low) > tol:
        one_minus_mid = 1.0 - mid

        # M(mid) = term1 + term2
        term1 = c1 * pow(mid, expo)  # used in Newton calc
        term2 = c2 * pow(one_minus_mid, expo)  # used in Newton calc
        M_mid = term1 + term2

        # update bisection interval [low, high]
        if M_mid > lower:  # True means "move high down"
            high = mid
            M_high = M_mid  # used in Secant calc
        else:  #  False means "move low up"
            low = mid
            M_low = M_mid  # used in Secant calc

        # attempt to use Newton's method to increase low
        grad = expo * (term1 / mid - term2 / one_minus_mid)
        newton = mid - (M_mid - lower) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < beta* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton
            M_low = c1 * pow(low, expo) + c2 * pow(1.0 - low, expo)

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15:  # avoid division by 0
            secant = low + (lower - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < beta* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant
                M_high = c1 * pow(high, expo) + c2 * pow(1.0 - high, expo)

        # update mid for next iteration of bisection
        mid = 0.5 * (low + high)

    # return high for formal guarantee that beta* < beta < beta* + tol
    return high


def get_FNR_case3_vec(alpha_array, order, epsilon, tol=1e-7):

    # compute lower, one_minus_order, alpha_max once to save compute in core_logic
    log_lower = (order - 1.0) * epsilon
    lower = exp(log_lower)
    one_minus_order = 1.0 - order
    alpha_max = exp((order - 1) / order * epsilon)

    # compute fixed point (hard code 1e-15 for stability)
    fixed_point = get_fixed_point_case3(order, lower, alpha_max, tol=1e-15)

    # allocate output array
    n = alpha_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case3(
            alpha_array[i], lower, order, one_minus_order, fixed_point, alpha_max, tol
        )

    return result_np


def get_FNR(
    alpha_array: Union[ArrayLike, float],
    order: float,
    epsilon: float,
    tol: float = 1e-7,
):
    """
    Args:
        alpha_array: input FPR(s). Can be a scalar or array like.
        order: Renyi order
        epsilon: Value that upper bounds the Renyi divergence
        tol: Tolerance for the returned FNR.

    Returns:
        FNRs at corresponding FPRs. Same shape as alpha_array.
    """

    # check if epsilon is valid
    if not (epsilon > 0):
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    # check if FPRs alpha are valid, and convert alpha to contiguous 1D numpy array
    is_scalar = np.isscalar(alpha_array)
    alpha_array = np.ascontiguousarray(np.atleast_1d(alpha_array))

    if not (np.all(0 <= alpha_array) and np.all(alpha_array <= 1)):
        raise ValueError(f"FPRs alpha must be between 0 and 1, got {alpha_array}")

    # exp((order - 1.0) * epsilon) is used in all functions; guard against overflow.
    log_val = (order - 1.0) * epsilon
    if log_val >= 200.0:
        warnings.warn(
            f"(order - 1) * epsilon = {log_val:.1f} >= 200: exp would overflow. "
            f"The RDP bound is vacuously loose; returning 0 for all FPR values.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = np.zeros(len(alpha_array), dtype=np.float64)
        return float(result[0]) if is_scalar else result

    # 0 < order < 0.5 are not needed by symmetry
    # per Zhu et al https://arxiv.org/pdf/2106.08567
    if order < 0.5:
        warnings.warn(
            f"RDP order < 0.5 is not needed by symmetry. " f"Got order = {order}"
        )

    # make sure order is valid for the Renyi divergence
    if not (0 < order):
        raise ValueError(f"RDP order must be positive, got {order}")

    if order > 1:
        output = get_FNR_case1_vec(alpha_array, order, epsilon, tol)
    elif order == 1:
        output = get_FNR_case2_vec(alpha_array, epsilon, tol)
    elif order < 1:
        output = get_FNR_case3_vec(alpha_array, order, epsilon, tol)

    if is_scalar:  # if original input alpha_array was a scalar
        return float(output[0])
    else:  # original alpha_array was ArrayLike
        return output

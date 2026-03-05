# imports for python
import numpy as np
import math
import warnings
from typing import List, Union

# Replacement for libc.math and cython types
from math import fabs, pow, exp, log, log1p
INFINITY = float('inf')

ArrayLike = Union[np.ndarray, List[float]]

'''
Notation:

    FPR = false positive rate
    FNR = false negative rate
    x = another name for FPR
    y = another name for FNR
    alpha = Renyi order
    rho = value of Renyi divergence


The main python facing function is get_FNR.
This function takes as input an single
(alpha, rho)-RDP guarantee and a single (or multiple)
FPR values x and returns the corresponding FNR values of
the tradeoff curve

The function does this by calling one of 3 cython functions,
each optimized for a particular regime of Renyi order alpha.
They are:

get_FNR_case1_vec: handles alpha > 1
get_FNR_case2_vec: handles alpha = 1
get_FNR_case3_vec: handles 0 < alpha < 1

Each of these functions perform a for loop (in cython) over
the list/array of given FPR values. For each FPR value, the functions
call a correspinding _core_logic function, which implements the actual
conversion. The conversion is itself a root-finding problem, and various
numerical techniques are used to speed up the compuation. The main doc
string that explains these techniques are in _core_logic_case1.
The doc strings of all other core logic functions only mention techniques
different from those in _core_logic_case1.
'''

###############
###############
### CASE 1 ####
###############
###############
def get_fixed_point_case1(alpha,
                          upper,
                          tol = 1e-7):
    '''
    See _core_logic_case1 doc string for context on why we need this
    function.

    Assuming alpha > 1, this function returns the fixed point of the function y(x)
    implicitly defined via:

    (1-x)^{1-alpha} y^alpha     + x^{1-alpha} (1-y)^alpha     <= upper
    (1-x)^{alpha}   y^{1-alpha} + x^{alpha}   (1-y)^{1-alpha} <= upper

    which can be easily seen to be the smallest value of y such that:

    (1-y)^{alpha} y^{1-alpha} + y^{alpha} (1-y)^{1-alpha} <= upper

    Let the lhs of this equation by F(y). F can be shown to be convex and
    decreasing. Since we know this function computes the fixed point of a
    tradeoff function, we know that 0 <= y <= 1/2.

    Therefore, this function uses bisection over [0,1/2] to find F(y) >= upper.
    Let y* denote the true inverse of F at upper. It is guaranteed that
    y* < y < y* + tol thanks to bisection. Since this function is not called
    as often as _core_logic_case1, it is not as performance critical, and we
    intentionally use standard bisection.
    '''

    # init [low, high]  for bisection
    low = 0
    high = 1/2

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = 0.5 * (low + high)
        F_mid = (1-mid) ** (1 - alpha) * mid ** alpha + mid ** (1-alpha) * (1-mid) ** alpha

        if F_mid > upper: # True means "move low up"
            low = mid
        else: # False means "move high down"
            high = mid

    # return high for formal guarantee that y* < y < y* + tol
    return high

def _core_logic_case1(x,
                      upper,
                      alpha,
                      one_minus_alpha,
                      fixed_point,
                      tol):
    '''
    For fixed RDP guarantee (alpha, rho) with alpha > 1 and FPR x,
    this function returns the smallest value of y in [0,1] that simultaneously satisfies:

    (1-x)^{1-alpha} y^alpha     + x^{1-alpha} (1-y)^alpha     <= upper
    (1-x)^{alpha}   y^{1-alpha} + x^{alpha}   (1-y)^{1-alpha} <= upper

    Let lhs1 denote the lhs of the first inequality, and lhs2 denote the second.
    Let M(y) = max(lhs1, lhs2). The above two inequalities simplify to M(y) <= upper.
    Let fixed_point denote the unique value such that (x = fixed_point, y = fixed_point)
    satisfies the two inequalities above. It can be shown that M(y) = lhs1(y) if x < fixed_point
    and M(y) = lhs2(y) if x > fixed_point. The code below uses this fact to focus on
    only one of the two inequalities above, depending on the value of FPR x.

    Note that if x = 0, then only y = 1 satisfies the inequalities. And if x = 1, then all values
    of y satisfy the inequalities, therefore we choose the smallest: y = 0. These two edge
    cases are hard coded into the function.

    M(y) is a convex decreasing function of y, and our goal is to return the smallest
    value of y such that M(y) <= upper. Let the result of this be denoted y = f(x).
    Since we know this is a tradeoff curve, we have that 0 <= y <= 1 - x.
    Therefore, we can use bisection over [0, 1-x] to numerically invert M(y) <= upper.

    Let y* denote the true inverse of M at upper, i.e. M(y*) = upper. Let y denote the
    output of this function. This function is guaranteed to return y such that
    y* < y < y* + tol.

    This function implements one more trick for speed: since M is convex and decreasing,
    using Newton's Method is guaranteed to yield a lower bound on y*, and using the secant
    method is guaranteed to yield an upper bound on y*. Therefore, we use both techniques to
    improve the bisection interval when possible.

    Args:
        x: FPR value
        upper: exp( (alpha-1) * rho). The value of the upper bound
        alpha: Renyi order
        one_minus_alpha: one minus the Renyi order
        fixed_point: fixed point of the tradeoff curve
        tol: tolerance. See doc string for meaning

    Returns:
        y: FNR at FPR
    '''

    # handle x = 0 or x = 1 explicitly
    if x == 0: return 1.0
    elif x == 1: return 0.0

    # save variables to reduce computation of M for later
    one_minus_x = 1.0 - x
    c1 = 0.0
    c2 = 0.0
    expo = 0.0

    if x <= fixed_point:
        expo = alpha
        c1 = pow(one_minus_x, one_minus_alpha)
        c2 = pow(x, one_minus_alpha)
    else:
        expo = one_minus_alpha
        c1 = pow(one_minus_x, alpha)
        c2 = pow(x, alpha)

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_x

    # M_low, M_high below assume low = 0 and high = 1 - x
    # if x <= fixed_point, expo > 0 and M_low is finite, if expo < 0, M_low is infinite
    M_low = c2 if x <= fixed_point else INFINITY

    #M_high: cython.double = c1 * pow(high, expo) + c2 * pow(1 - high, expo)
    M_high = c1 * pow(high, expo) + c2 * pow(x, expo)

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
        term1 = c1 * pow(mid, expo) # used in Newton calc
        term2 = c2 * pow(one_minus_mid, expo) # used in Newton calc
        M_mid = term1 + term2

        # update bisection interval [low, high]
        if M_mid > upper: # True means "move low up"
            low = mid
            M_low = M_mid # used in Secant calc
        else: #  False means "move high down"
            high = mid
            M_high = M_mid # used in Secant calc

        # attempt to use Newton's method to increase low
        grad = expo * (term1 / mid - term2 / one_minus_mid)
        newton = mid - (M_mid - upper) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < y* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton
            M_low = c1 * pow(low, expo) + c2 * pow(1.0 - low, expo)

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15: # avoid division by 0
            secant = low + (upper - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < y* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant
                M_high = c1 * pow(high, expo) + c2 * pow(1.0 - high, expo)

        # update mid for next iteration of bisection
        mid = 0.5 * (low + high)

    # return high for formal guarantee that y* < y < y* + tol
    return high

def get_FNR_case1_vec(x_array,
                    alpha,
                    rho,
                    tol = 1e-7):

    # compute upper and one_minus_alpha once to save compute in core_logic
    log_upper = (alpha - 1.0) * rho
    upper = exp(log_upper)
    one_minus_alpha = 1.0 - alpha

    # compute fixed point (hard code 1e-15 for stability)
    fixed_point = get_fixed_point_case1(alpha, upper, tol = 1e-15)

    # allocate output array
    n = x_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case1(x_array[i], upper, alpha, one_minus_alpha, fixed_point, tol)

    return result_np


###############
###############
### CASE 2 ####
###############
###############
def get_fixed_point_case2(rho,
                          tol = 1e-7):
    '''
    See _core_logic_case1 and get_fixed_point_case1 doc string for context on
    why we need this function. Only difference from get_fixed_point_case1 is
    that here, the fixed point equation is:

    y log( y/(1-y) ) + (1-y) log( (1-y)/y) <= rho

    The same bisection algo used in get_fixed_point_case1 is used here
    '''

    # init [low, high]  for bisection
    low = 0
    high = 1/2

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = 0.5 * (low + high)
        F_mid = mid * (log(mid) - log1p(-mid)) + (1-mid) * (log1p(-mid) - log(mid))
        if F_mid > rho:   # True means "move low up"
            low = mid
        else: # False means "move high down"
            high = mid

    # return high for formal guarantee that y* < y < y* + tol
    return high


def _core_logic_case2(x,
                      rho,
                      fixed_point,
                      tol):
    '''
    See _core_logic_case1 doc string for details. The only difference
    between case1 and this function is that here, alpha = 1,
    where the inequalities are

    y log( y/(1-x) ) + (1-y) log( (1-y)/x) <= rho
    x log( x/(1-y) ) + (1-x) log( (1-x)/y) <= rho

    Still, M(y) defined as in _core_logic_case1 is convex and decreasing,
    so the implementation of bisection + Newton + Secant is the same.
    '''
    # handle x = 0 or x = 1 explicitly
    if x == 0: return 1.0
    elif x == 1: return 0.0

    # save variables to reduce computation of M for later
    one_minus_x = 1.0 - x
    log_one_minus_x = log1p(-x)
    log_x = log(x)

    less_than_fixed_point = (x <= fixed_point)

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_x

    M_low = 0.0
    M_high = 0.0

    if less_than_fixed_point:
        # M_low = M(0) = -log(x) (limit as y -> 0+)
        M_low = -log_x
        M_high = high * (log(high) - log_one_minus_x) + (1.0 - high) * (log1p(-high) - log_x)
    else:
        # M(0) = +inf (because -(1-x)*log(y) term blows up)
        M_low = INFINITY
        M_high = x * (log_x - log1p(-high)) + one_minus_x * (log_one_minus_x - log(high))

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
            M_mid = mid * (log(mid) - log_one_minus_x) + one_minus_mid * (log1p(-mid) - log_x)
            grad = (log(mid) - log_one_minus_x) - (log1p(-mid) - log_x)
        else:
            M_mid = x * (log_x - log1p(-mid)) + one_minus_x * (log_one_minus_x - log(mid))
            grad = (mid - one_minus_x) / (mid * one_minus_mid)

        # update bisection interval [low, high]
        if M_mid > rho: # True means "move low up"
            low = mid
            M_low = M_mid # used in Secant calc
        else:  #  False means "move high down"
            high = mid
            M_high = M_mid # used in Secant calc

        # attempt to use Newton's method to increase low
        newton = mid - (M_mid - rho) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < y* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton

            # recompute M_low
            if less_than_fixed_point:
                M_low = low * (log(low) - log_one_minus_x) + (1.0 - low) * (log1p(-low) - log_x)
            else:
                M_low = x * (log_x - log1p(-low)) + one_minus_x * (log_one_minus_x - log(low))

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15: # avoid division by 0
            secant = low + (rho - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < y* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant

                # recompute M_high
                if less_than_fixed_point:
                    M_high = high * (log(high) - log_one_minus_x) + (1.0 - high) * (log1p(-high) - log_x)
                else:
                    M_high = x * (log_x - log1p(-high)) + one_minus_x * (log_one_minus_x - log(high))

        # Next bisection midpoint
        mid = 0.5 * (low + high)

    # return high for formal guarantee that y* < y < y* + tol
    return high

def get_FNR_case2_vec(x_array,
                    rho,
                    tol = 1e-7):

    # compute fixed point (hard code 1e-15 tol for stability)
    fixed_point = get_fixed_point_case2(rho, tol = 1e-15)

    # allocate output array
    n = x_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core_logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case2(x_array[i], rho, fixed_point, tol)

    return result_np



###############
###############
### CASE 3 ####
###############
###############
def get_fixed_point_case3(alpha,
                          lower,
                          x_max,
                          tol = 1e-7):
    '''
    See _core_logic_case1 and get_fixed_point_case1 doc string for context on
    why we need this function. Only difference from get_fixed_point_case1 is
    that here, the fixed point equation is:

    (1-y)^{alpha} y^{1-alpha} + y^{alpha} (1-y)^{1-alpha} >= lower

    Since alpha < 1 for case 3, now F(y) (the lhs of the inequality) is
    increasing and concave, as opposed to decreasing and convex as in case 1.
    Moreover, in get_fixed_point_case1 all we knew was the the solution had
    to be between [0,1/2]. Here, we know that the fixed point must be less than
    x_max, so we search between [0,x_max].

    The code below returns a value of y that is guaranteed to have the property
    F(y) >= lower. It does this by bisection. Let y* denote the true inverse of
    F at lower. It is guaranteed that y* < y < y* + tol.
    '''

    # init [low, high]  for bisection
    low = 0
    high = x_max

    # init all other vars
    mid = 0.0
    F_mid = 0.0

    # run bisection
    while high - low > tol:
        mid = (low + high) / 2.0
        F_mid = (1-mid) ** (1 - alpha) * mid ** alpha + mid ** (1-alpha) * (1-mid) ** alpha
        if F_mid > lower:
            high = mid # True means "move high down"
        else:
            low = mid # False means "move low up"

    # return high for formal guarantee that y* < y < y* + tol
    return high


def _core_logic_case3(x,
                      lower,
                      alpha,
                      one_minus_alpha,
                      fixed_point,
                      x_max,
                      tol):
    '''
    See _core_logic_case1 doc string for details. The only difference
    between case 1 and this function is that here, alpha < 1,
    where the inequalities are

    (1-x)^{1-alpha} y^alpha     + x^{1-alpha} (1-y)^alpha     >= lower
    (1-x)^{alpha}   y^{1-alpha} + x^{alpha}   (1-y)^{1-alpha} >= lower

    In particular, they are identical except with the inequality fliped.
    This has a much bigger impact than one may imagine. For example, M(y) is
    now concave and increasing since we need to take the min(lhs1, lhs2) and
    not the max. Note that in case 1 and 2, M(y) is convex and decreasing.
    More importantly, however, is that for alpha < 1, the inequalities
    have very different behaviors at x = 0.

    At x = 0, the inequality simplifies to y >= e^{ (alpha-1) / alpha * rho }.
    Therefore, the smallest y must be e^{ (alpha-1) / alpha * rho }, which is not 1.

    Due to symmetry, the same effect occurs at y = 0, in that we get
    x >= e^{ (alpha-1) / alpha * rho }. This leads to the somewhat counter intuitive
    situation where if the input x satisfies x >= e^{ (alpha-1) / alpha * rho }, then
    it must be that y = 0. This is hard coded into the code below (x_max is this value).

    The above implies the tradeoff function for RDP when alpha < 1 has the property
    f(0) < 1 and f(x') = 0 for x' in some non zero interval that contains 1. This
    implies this tradeoff function catastrophically fails (i.e. has infinite privacy loss).
    We leave further exploration of this to future work.
    '''

    # handle x = 0 and x >= x_max explicitly
    if x == 0: return x_max
    elif x >= x_max: return 0.0

    # save variables to reduce computation of M for later
    one_minus_x = 1.0 - x
    c1 = 0.0
    c2 = 0.0
    expo = 0.0

    if x <= fixed_point:
        expo = alpha
        c1 = pow(one_minus_x, one_minus_alpha)
        c2 = pow(x, one_minus_alpha)
    else:
        expo = one_minus_alpha
        c1 = pow(one_minus_x, alpha)
        c2 = pow(x, alpha)

    # init [low, high] and M(low), M(high) for bisection
    low = 0.0
    high = one_minus_x

    # two lines below assume low = 0 and high = 1 - x
    M_low = 0
    M_high = c1 * pow(high, expo) + c2 * pow(x, expo)

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
        term1 = c1 * pow(mid, expo) # used in Newton calc
        term2 = c2 * pow(one_minus_mid, expo) # used in Newton calc
        M_mid = term1 + term2

        # update bisection interval [low, high]
        if M_mid > lower: # True means "move high down"
            high = mid
            M_high = M_mid # used in Secant calc
        else: #  False means "move low up"
            low = mid
            M_low = M_mid # used in Secant calc

        # attempt to use Newton's method to increase low
        grad = expo * (term1 / mid - term2 / one_minus_mid)
        newton = mid - (M_mid - lower) / grad

        # if newton point fits inside bisection interval [low, high],
        # then we know that low < newton < y* < high, therefore we
        # let low = newton to improve bisection interval.
        if low < newton < high:
            low = newton
            M_low = c1 * pow(low, expo) + c2 * pow(1.0 - low, expo)

        # attempt to use Secant method to decrease high
        if fabs(M_high - M_low) < 1e-15: # avoid division by 0
            secant = low + (lower - M_low) * (high - low) / (M_high - M_low)

            # if secant point fits inside bisection interval [low, high],
            # then we know that low < y* < secant < high, therefore we
            # let high = secant to improve bisection interval.
            if low < secant < high:
                high = secant
                M_high = c1 * pow(high, expo) + c2 * pow(1.0 - high, expo)

        # update mid for next iteration of bisection
        mid = 0.5 * (low + high)

    # return high for formal guarantee that y* < y < y* + tol
    return high

def get_FNR_case3_vec(x_array,
                    alpha,
                    rho,
                    tol = 1e-7):

    # compute lower, one_minus_alpha, x_max once to save compute in core_logic
    log_lower = (alpha - 1.0) * rho
    lower = exp(log_lower)
    one_minus_alpha = 1.0 - alpha
    x_max = exp( (alpha-1)/alpha * rho)

    # compute fixed point (hard code 1e-15 for stability)
    fixed_point = get_fixed_point_case3(alpha, lower, x_max, tol = 1e-15)

    # allocate output array
    n = x_array.shape[0]
    result_np = np.empty(n, dtype=np.float64)

    # run core logic in C for loop
    for i in range(n):
        result_np[i] = _core_logic_case3(x_array[i], lower, alpha, one_minus_alpha, fixed_point, x_max, tol)

    return result_np

def get_FNR(x_array: Union[ArrayLike, float],
            alpha: float,
            rho: float,
            tol: float = 1e-7):
    '''
    Args:
        x_array: input FPR(s). Can be a scalar or array like.
        alpha: Renyi order
        rho: Value that upper bounds the Renyi divergence
        tol: Tolerance for the returned FNR.

    Returns:
        FNRs at corresponding FPRs. Same shape as x_array.
    '''

    # check if rho is valid
    if not (rho > 0):
        raise ValueError(f"rho must be positive, got {rho}")

    # check if FPRs x are valid, and convert x to contiguous 1D numpy array
    is_scalar = np.isscalar(x_array)
    x_array = np.ascontiguousarray(np.atleast_1d(x_array))

    if not (np.all(0 <= x_array) and np.all(x_array <= 1)):
        raise ValueError(f"FPRs x must be between 0 and 1, got {x_array}")

    # exp((alpha - 1.0) * rho) is used in all functions; guard against overflow.
    log_val = (alpha - 1.0) * rho
    if log_val >= 200.0:
        warnings.warn(
            f"(alpha - 1) * rho = {log_val:.1f} >= 200: exp would overflow. "
            f"The RDP bound is vacuously loose; returning 0 for all FPR values.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = np.zeros(len(x_array), dtype=np.float64)
        return float(result[0]) if is_scalar else result

    # 0 < alpha < 0.5 are not needed by symmetry
    # per Zhu et al https://arxiv.org/pdf/2106.08567
    if alpha < 0.5:
        warnings.warn(f"RDP order alpha < 0.5 is not needed by symmetry. "
                      f"Got alpha = {alpha}")

    # make sure alpha is valid for the Renyi divergence
    if not (0 < alpha):
        raise ValueError(f"RDP order alpha must be positive, got {alpha}")

    if alpha > 1:
        output = get_FNR_case1_vec(x_array, alpha, rho, tol)
    elif alpha == 1:
        output = get_FNR_case2_vec(x_array, rho, tol)
    elif alpha < 1:
        output = get_FNR_case3_vec(x_array, alpha, rho, tol)

    if is_scalar: # if original input x_array was a scalar
        return float(output[0])
    else: # original x_array was ArrayLike
        return output

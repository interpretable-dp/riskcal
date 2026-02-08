## riskcal

[![pypi](https://img.shields.io/pypi/v/riskcal)](https://pypi.org/project/riskcal/)
[![docs](https://app.readthedocs.org/projects/riskcal/badge/?version=latest)](https://riskcal.readthedocs.io/latest/)
[![CI](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2407.02191-b31b1b.svg)](https://arxiv.org/abs/2407.02191)
[![arXiv](https://img.shields.io/badge/arXiv-2507.06969-b31b1b.svg)](https://arxiv.org/abs/2507.06969)

---

⚠️  This is a research prototype. Avoid or be extra careful when using in production.

---

The library provides tools for computing f-DP trade-off curves for differentially private
algorithms, and calibrating their noise scale to operational privacy risk measures (attack
advantage, or attack TPR and FPR). The library enables reducing noise scale while maintaining
the same level of targeted attack risk.

### Background

#### Privacy risk and f-DP

[Differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) (DP) protects against
information leakage from machine learning models, datasets, and statistical releases by adding
controlled random noise. Traditional DP uses epsilon-delta parameters that are difficult to
interpret. In practice, we want to understand privacy in terms of concrete attack risks membership
inference, re-identification, and attribute reconstruction.

**f-Differential privacy (f-DP)** directly quantifies these attack risks using:
- **Trade-off curve**: False positive rate (FPR, `alpha`) vs. false negative rate (FNR, `beta`) for
  the worst-case attacker. Note that true positive rate (TPR) is `1 - beta`.
- **Advantage**: Maximum value of `TPR - FPR` = `1 - beta - alpha`, achieved at the optimal
  threshold. This bound is also known as total variation (TV) privacy.

The f-DP trade-off curve and advantage correspond directly to operational attack metrics:

| Measure | Membership Inference | Singling out / Reconstruction / Attribute Inference |
|---------|----------------------|-----------------------------------------------------|
| **Trade-off curve (beta/alpha)** | Min FNR at FPR | Max success probability bounded by TPR at baseline `alpha` |
| **Advantage** | Max TPR - FPR | Max probability increase over baseline |

See Kulynych & Gomez et al. (2024)[^1] for membership inference and Kulynych et al. (2025)[^2] for
the unified framework connecting re-identification, attribute inference, and reconstruction.


#### Methods
The library implements methods described by Kulynych & Gomez et al., 2024[^1].

- The direct method for computing the trade-off curve based on privacy loss
  random variables is described in Algorithm 1.
- The mapping between f-DP and operational privacy risk, and the idea of direct
  noise calibration to risk instead of the standard calibration to a given
  (epsilon, delta) is described in Sections 2 and 3.

#### References

[^1]: [Attack-Aware Noise Calibration for Differential
Privacy](https://arxiv.org/abs/2407.02191). NeurIPS 2024.
[^2]: [Unifying Re-Identification, Attribute Inference, and Data Reconstruction
Risks in Differential Privacy](https://arxiv.org/abs/2507.06969). NeurIPS 2025.

If you make use of the library or methods, please cite:
```bibtex
@article{kulynych2024attack,
  title={Attack-aware noise calibration for differential privacy},
  author={Kulynych, Bogdan and Gomez, Juan F and Kaissis, Georgios and du Pin Calmon, Flavio and Troncoso, Carmela},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}

@article{kulynych2025unifying,
  title={Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy},
  author = {Kulynych, Bogdan and Gomez, Juan Felipe and Kaissis, Georgios and Hayes, Jamie and Balle, Borja and du Pin Calmon, Flavio and Raisaro, Jean Louis},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

### Installation

Install with pip:
<!--pytest.mark.skip-->
```bash
pip install riskcal
```

For local development, clone the repository and run:
<!--pytest.mark.skip-->
```bash
uv sync --dev
```

### Quickstart

#### Analysis and Conversions

The library supports computing privacy risk metrics from different privacy representations:

| Source Privacy Notion | Trade-off (f-DP) | Advantage (TV) | Bayes Risk |
|-----------------------|------------------|----------------|------------|
| **PLD** (Privacy Loss Distribution) | ✓ | ✓ | ✓ |
| **GDP** (Gaussian DP) | ✓ | ✓ | ✓ |
| **ADP** (Approximate DP) | ✓* | ✓* | – |
| **RDP** (Renyi DP) | ✓ | – | – |
| **zCDP** (Zero-Concentrated DP) | ✓ | ✓ | – |

* The conversion from a single pair of (epsilon, delta)-ADP guarantees is very lossy for most
  mechanisms[^1]. For meaningful risk bounds, use other available representations.

_Minimal Example._ Here's a complete example showing how to evaluate privacy risk for a simple mechanism:

```python
from riskcal import analysis
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

# Create a Gaussian mechanism with noise scale 1.0
pld = from_gaussian_mechanism(1.0)

# What's the worst-case attack advantage?
advantage = analysis.get_advantage_from_pld(pld)
print(f"Maximum attack advantage: {advantage:.3f}")

# What's the maximum attack TPR at 1% FPR?
beta = analysis.get_beta_from_pld(pld, alpha=0.01)
tpr_bound = 1 - beta
print(f"Max TPR at 1% FPR: {tpr_bound:.3f}")
```

_Computing Trade-Off Curves._ The trade-off curve shows the relationship between false positive rate
(FPR, `alpha`) and false negative rate (FNR, `beta`) for the worst-case attacker. For DP-SGD:

```python
from riskcal.analysis import get_beta_from_pld
from dp_accounting.pld import privacy_loss_distribution as pld_module
import numpy as np

# DP-SGD parameters
noise_multiplier = 0.5
sample_rate = 0.002
num_steps = 10000

# Create PLD for DP-SGD using composition
pld = pld_module.from_gaussian_mechanism(
    standard_deviation=noise_multiplier,
    sampling_prob=sample_rate,
    use_connect_dots=True,
).self_compose(num_steps)

# Compute FNR (beta) at various FPR (alpha) values
alpha = np.array([0.01, 0.05, 0.1])
beta = get_beta_from_pld(pld, alpha=alpha)
print(f"Trade-off curve: {list(zip(alpha, beta))}")
```

You can also use the Opacus-compatible accountant for easier integration with DP-SGD training:

```python
from riskcal.accountants import CTDAccountant
import numpy as np

# Track privacy over training
acct = CTDAccountant()
noise_multiplier = 0.5
sample_rate = 0.002
num_steps = 100
for _ in range(num_steps):
    acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

# Get the trade-off curve
alpha = np.array([0.01, 0.05, 0.1])
beta = acct.get_beta(alpha=alpha)

# Get the maximum advantage
advantage = acct.get_advantage()
print(f"Maximum attack advantage: {advantage:.3f}")
```

The library works with any DP mechanism
[supported](https://github.com/google/differential-privacy/tree/main/python/dp_accounting) supported
by `dp_accounting` library, e.g., arbitrary adaptive compositions of (Subsampled) Gaussian, Discrete
Gaussian, Laplace mechanisms:

```python
from riskcal.analysis import get_beta_from_pld, get_advantage_from_pld
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism, from_laplace_mechanism

# Compose multiple mechanisms
pld = from_gaussian_mechanism(1.0).compose(from_laplace_mechanism(0.5))

# Analyze the composed mechanism
advantage = get_advantage_from_pld(pld)
beta = get_beta_from_pld(pld, alpha=0.1)
print(f"Advantage: {advantage:.3f}, Beta at alpha=0.1: {beta:.3f}")
```

_Gaussian Differential Privacy (GDP)._ Gaussian differential privacy tightly characterizes many DP
mechanisms based on the Gaussian mechanism or its composition. For a given GDP parameter `mu`, get
the advantage and the trade-off curve:

```python
from riskcal.analysis import get_advantage_from_gdp, get_beta_from_gdp
import numpy as np

mu = 2.0  # GDP parameter

# Get advantage directly from mu
advantage = get_advantage_from_gdp(mu)

# Get trade-off curve
alpha = np.linspace(0, 1, 100)
beta = get_beta_from_gdp(mu, alpha)
```
This is faster than PLD-based computation.See also a dedicated library for privacy accounting of DP
mechanisms in terms of GDP, [`gdpnum`](https://github.com/Felipe-Gomez/gdp-numeric).

_Zero-Concentrated Differential Privacy (zCDP) and Renyi DP (RDP)._ Zero-Concentrated Differential
Privacy (zCDP) is characterized by a single parameter `rho`.  Renyi DP is parameterized by an
`epsilon` value at a specific divergence `order`.

Get the trade-off curve from zCDP:

```python
from riskcal.analysis import get_advantage_from_zcdp, get_beta_from_zcdp
import numpy as np

rho = 0.5  # zCDP parameter

# Get advantage directly from rho
advantage = get_advantage_from_zcdp(rho)
print(f"Maximum attack advantage: {advantage:.3f}")

# Get trade-off curve for various FPR values
alpha = np.linspace(0.01, 0.1, 10)
beta = get_beta_from_zcdp(rho, alpha)
print(f"Trade-off curve: {list(zip(alpha, beta))}")
```

Get the trade-off curve from RDP:

```python
from riskcal.analysis import get_beta_from_rdp
import numpy as np

epsilon = 1.0  # Renyi divergence parameter
order = 2.0    # Renyi divergence order (alpha in Renyi DP literature)

# Get FNR at specific FPR
beta = get_beta_from_rdp(epsilon=epsilon, alpha=0.1, order=order)
print(f"Beta (FNR) at alpha=0.1: {beta:.3f}")
```

_Computing Bayes Risk._ Bayes risk measures the maximum accuracy of attacks under a binary prior.
This is useful for attribute inference (assuming a record has one of two attributes) or membership
inference (with a prior probability of membership):

```python
from riskcal.analysis import get_bayes_risk_from_pld
from dp_accounting.pld.privacy_loss_distribution import from_laplace_mechanism
import numpy as np

pld = from_laplace_mechanism(1.0)

# Compute attack accuracy for different prior probabilities
prior = np.array([0.5, 0.8, 0.95])
risk = get_bayes_risk_from_pld(pld, prior=prior)
print(f"Bayes risk at priors {prior}: {risk}")
```

#### Calibration

You can directly calibrate noise to bound attack success rates.

_Calibrating Noise for DP-SGD._ Calibrate to maximum attack advantage:

```python
from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage

sample_rate = 0.002
num_steps = 10000

# Find noise multiplier that bounds advantage at 10%
noise_multiplier = find_noise_multiplier_for_advantage(
    advantage=0.1,
    sample_rate=sample_rate,
    num_steps=num_steps,
)
print(f"Required noise multiplier: {noise_multiplier:.3f}")
```

Calibrate to bound attack TPR at a specific FPR:

```python
from riskcal.calibration.dpsgd import find_noise_multiplier_for_err_rates

# Bound attack to max 5% TPR at 1% FPR
noise_multiplier = find_noise_multiplier_for_err_rates(
    beta=0.95,  # FNR = 1 - TPR = 1 - 0.05
    alpha=0.01,  # FPR
    sample_rate=0.002,
    num_steps=10000,
    grid_step=1e-2,  # Lower resolution for the sake of running the example faster.
)
print(f"Required noise multiplier: {noise_multiplier:.3f}")
```

_Calibrating Generic Mechanisms._ For custom mechanisms beyond DP-SGD, use the generic calibration
framework. You provide an evaluator function that computes privacy metrics for a given parameter
value:

```python
from riskcal.calibration.core import (
    calibrate_parameter,
    PrivacyEvaluator,
    PrivacyMetrics,
    CalibrationTarget,
    CalibrationConfig,
)
from riskcal.analysis import get_advantage_from_pld, get_beta_from_pld
from dp_accounting.pld.privacy_loss_distribution import from_laplace_mechanism

# Define evaluator for Laplace mechanism
def evaluate_laplace(scale: float) -> PrivacyMetrics:
    """Compute privacy metrics for Laplace mechanism with given scale."""
    pld = from_laplace_mechanism(sensitivity=1.0, parameter=scale)
    advantage = get_advantage_from_pld(pld)
    beta = get_beta_from_pld(pld, alpha=0.01)  # For FPR=1%
    return PrivacyMetrics(advantage=advantage, alpha=0.01, beta=beta)

# Calibrate to advantage target
target = CalibrationTarget(kind="advantage", advantage=0.1)
config = CalibrationConfig(param_min=0.1, param_max=30.0, increasing=False)
result = calibrate_parameter(
    evaluator=evaluate_laplace,
    target=target,
    config=config,
    parameter_name="scale"
)
print(f"Required Laplace scale: {result.parameter_value:.3f}")
print(f"Achieved advantage: {result.achieved_advantage:.3f}")
```
This approach works with any mechanism where you can compute privacy metrics as a function
of a tunable parameter (noise scale, sampling rate, etc.).

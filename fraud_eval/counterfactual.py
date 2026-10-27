"""Counterfactual evaluation of fraud models under selective and delayed labels.

This module is a clean-room rebuild of the evaluation protocol from the
paper "Counterfactual Evaluation of Fraud Models under Selective and
Delayed Labels" (fraud-model evaluation under a deterministic threshold
champion). It contains no paper code and no real data: every order is
simulated, and the only labels an estimator ever sees are the ones a
production log would actually contain.

The problem
-----------
A deployed champion declines every order scoring above a threshold, so
declined orders never reveal whether they were fraudulent, and chargebacks
arrive weeks later, so recent orders are only partly labeled. Offline
metrics computed on the labeled remainder are therefore biased: they
describe the orders the champion approved, not the population a challenger
would serve.

The protocol
------------
Each order has features ``X``, value ``V > 0``, fraud status ``Y`` (the
status if the order completes), chargeback delay ``D``, and order time
``T``. The champion declines orders above a score threshold. ``R = 1`` if
the order completes. An exploration layer completes a declined order in
stratum ``k`` (risk-score bins crossed with order-value bands) with
probability ``p_k`` and logs ``p_k``, so the propensity is ``e(X, V) = 1``
for champion-approved orders and ``p_k`` for declined orders. At
evaluation time an order of age ``a`` has visible label
``Y_vis = Y * 1{D <= a}``, available only if ``R = 1``.

Estimands (target population: orders at least ``a_min`` old, 14 days by
default). For a candidate flag ``phi_j(X)`` and lost-margin ratio ``lambda``
(the cost of blocking a legitimate order, as a fraction of order value),
dollar-weighted net savings are ``theta_j = E[g_j]`` with
``g_j = phi_j(X) * V * ((1 + lambda) * Y - lambda)``. Precision at capacity
and AUC are defined on the same population.

Estimators
----------
* ``naive``: visible-label contributions averaged over completed orders.
  Descriptive only; biased whenever declined orders differ from approved
  ones or labels are immature.
* ``ipw``: delay-corrected inverse-propensity weighting. Contributions use
  ``Y_vis / F(a)`` (``F`` = fitted lognormal delay law, floored at 0.05)
  and weights ``R / e``, clipped at a configurable cap.
* ``dr``: delay-corrected doubly robust estimation. An outcome model
  ``mu_hat`` for ``E[g_tilde | X, V]`` is fitted on completed orders by
  weighted maximum likelihood, where the probability of a *visible* fraud
  label is ``sigmoid(x . beta) * F(a)``; the estimate is
  ``mean(mu_hat + (R / e) * (g_tilde - mu_hat))``.

Identification
--------------
If a candidate flags a declined order whose stratum has zero exploration
propensity, that order's contribution is unidentified: IPW and DR return
``identified=False`` and ``value=None`` instead of silently extrapolating
or dividing by zero. Use :func:`partial_identification_bounds` for the
sharp bounds that remain available (paper Proposition 2).

Promotion gate
--------------
The gate promotes a challenger only when a lower confidence bound on its
paired advantage over the champion is positive. The interval here is the
normal approximation on paired doubly-robust pseudo-values
(``advantage - z_{1-alpha/J} * SE`` with a Bonferroni correction when
``J`` challengers are screened). It is deterministic and fast; the paper
also reports a bootstrap-t bound, which is more conservative at small
exploration budgets and is not implemented here. Intervals ignore
nuisance-estimation error (delay law and outcome model are fitted, not
known), so treat them as approximate, exactly as the paper does.

Ground truth
------------
``OrderLog.fraud``, ``OrderLog.delays`` (except at visible-fraud indices,
where the delay is genuinely observed), and ``OrderLog.logit`` are
simulation-only. Estimators never read them; the ``ground_truth_*``
functions exist so tests and examples can validate against known truth.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

# Floor on the delay CDF, as in the paper. The correction divides by F(a),
# so an unfloored F near zero would explode individual contributions.
FLOOR_F = 0.05
DEFAULT_A_MIN = 14.0
DEFAULT_LAMBDA = 0.15
DEFAULT_CLIP = 100.0
DEFAULT_ALPHA = 0.05

# Simulator defaults, matching the paper's headline design: ten features,
# lognormal values ~1.65x larger for fraud, lognormal chargeback delays
# with a 25-day median.
_N_FEATURES = 10
_FEATURE_COEF = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05)
_FRAUD_INTERCEPT = -5.0
_LOG_VALUE_MEAN = 4.0
_LOG_VALUE_SD = 1.0
_FRAUD_VALUE_SHIFT = math.log(1.65)
_CHAMPION_NOISE = 0.5

_NORM = statistics.NormalDist()


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _z_value(alpha: float) -> float:
    """(1 - alpha) quantile of the standard normal."""
    return _NORM.inv_cdf(1.0 - alpha)


def _quantile_cutpoints(sorted_vals: list[float], n_bins: int) -> list[float]:
    n = len(sorted_vals)
    cuts = []
    for k in range(1, n_bins):
        cuts.append(sorted_vals[min(n - 1, int(k * n / n_bins))])
    return cuts


def _assign_bin(x: float, cuts: list[float]) -> int:
    b = 0
    for c in cuts:
        if x >= c:
            b += 1
        else:
            break
    return b


@dataclass
class OrderLog:
    """One simulated evaluation log.

    Fields ``fraud``, ``delays`` and ``logit`` are ground truth and exist
    only so simulations can be validated. Estimators read only
    ``features``, ``values``, ``completed``, ``propensity``, ``ages``,
    ``visible``, ``champion_flag`` and ``stratum`` -- plus ``delays`` at
    visible-fraud indices, where the delay is genuinely observed (the
    chargeback arrived). Never feed true labels into an estimator and
    expect a production answer.
    """

    features: list[list[float]]
    values: list[float]
    fraud: list[int]              # TRUE Y; simulation validation only
    completed: list[int]          # R: 1 if the order completed
    propensity: list[float]       # e(X, V): 1.0 if approved, p_k if declined
    ages: list[float]             # age in days at evaluation time
    visible: list[int]            # Y_vis: 1 only if completed, fraud, delay <= age
    delays: list[float]           # TRUE D; only visible-fraud entries are observed
    logit: list[float]            # TRUE fraud logit; simulation validation only
    champion_flag: list[int]      # 1 if the champion declines the order
    champion_score: list[float]
    stratum: list[int]            # stratum index if declined, -1 if approved
    exploration_by_stratum: dict[int, float] = field(default_factory=dict)
    median_delay: float = 25.0
    delay_sigma: float = 0.9

    def __len__(self) -> int:
        return len(self.values)


def simulate_orders(
    n: int = 20000,
    seed: int = 7,
    decline_rate: float = 0.05,
    exploration: float = 0.20,
    n_score_bins: int = 5,
    n_value_bands: int = 4,
    median_delay: float = 25.0,
    delay_sigma: float = 0.9,
    age_window: float = 120.0,
    fraud_intercept: float = _FRAUD_INTERCEPT,
    champion_noise: float = _CHAMPION_NOISE,
) -> OrderLog:
    """Simulate an order log with known ground truth.

    Fraud logit is linear in ten standard-normal features; values are
    lognormal (fraud orders average 1.65x larger); the champion score is
    the true logit plus Gaussian noise and declines the top
    ``decline_rate`` fraction; declined orders complete with probability
    ``exploration`` (uniform across strata); delays are lognormal with
    the given median. Orders arrive uniformly over ``age_window`` days
    before evaluation, so age = ``age_window`` - arrival time.

    With ``exploration=0`` no declined order completes: the declined
    region is then unidentified by design (see module docstring), and
    IPW/DR report that instead of crashing.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= decline_rate <= 1.0:
        raise ValueError("decline_rate must be in [0, 1]")
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be in [0, 1]")
    rng = random.Random(seed)

    features: list[list[float]] = []
    values: list[float] = []
    fraud: list[int] = []
    logit: list[float] = []
    delays: list[float] = []
    ages: list[float] = []
    champion_score: list[float] = []
    for _ in range(n):
        x = [rng.gauss(0.0, 1.0) for _ in range(_N_FEATURES)]
        lg = fraud_intercept + sum(c * v for c, v in zip(_FEATURE_COEF, x))
        p = _sigmoid(lg)
        y = 1 if rng.random() < p else 0
        log_v = rng.gauss(_LOG_VALUE_MEAN, _LOG_VALUE_SD)
        if y:
            log_v += _FRAUD_VALUE_SHIFT
        features.append(x)
        values.append(math.exp(log_v))
        fraud.append(y)
        logit.append(lg)
        delays.append(math.exp(rng.gauss(math.log(median_delay), delay_sigma)))
        ages.append(age_window - rng.uniform(0.0, age_window))
        champion_score.append(lg + rng.gauss(0.0, champion_noise))

    # Champion: deterministic threshold on its own score.
    order = sorted(range(n), key=lambda i: champion_score[i])
    n_decline = int(round(decline_rate * n))
    declined = set(order[n - n_decline:]) if n_decline > 0 else set()
    champion_flag = [1 if i in declined else 0 for i in range(n)]

    # Strata among declined orders: score bins x value bands.
    stratum = [-1] * n
    exploration_by_stratum: dict[int, float] = {}
    if declined:
        dec = sorted(declined)
        score_cuts = _quantile_cutpoints(
            sorted(champion_score[i] for i in dec), n_score_bins
        )
        value_cuts = _quantile_cutpoints(
            sorted(values[i] for i in dec), n_value_bands
        )
        for i in dec:
            s = _assign_bin(champion_score[i], score_cuts)
            b = _assign_bin(values[i], value_cuts)
            stratum[i] = s * n_value_bands + b
        for s in set(stratum[i] for i in dec):
            exploration_by_stratum[s] = exploration

    completed: list[int] = []
    propensity: list[float] = []
    visible: list[int] = []
    for i in range(n):
        if champion_flag[i] == 0:
            completed.append(1)
            propensity.append(1.0)
        else:
            p_k = exploration_by_stratum[stratum[i]]
            completed.append(1 if rng.random() < p_k else 0)
            propensity.append(p_k)
        visible.append(
            1 if (completed[i] == 1 and fraud[i] == 1 and delays[i] <= ages[i]) else 0
        )

    return OrderLog(
        features=features,
        values=values,
        fraud=fraud,
        completed=completed,
        propensity=propensity,
        ages=ages,
        visible=visible,
        delays=delays,
        logit=logit,
        champion_flag=champion_flag,
        champion_score=champion_score,
        stratum=stratum,
        exploration_by_stratum=exploration_by_stratum,
        median_delay=median_delay,
        delay_sigma=delay_sigma,
    )


def flags_from_scores(scores: list[float], capacity: float) -> list[int]:
    """Flag the top ``capacity`` fraction of orders by score.

    ``capacity`` is the share of orders the candidate would block (the
    paper's capacity constraint). Ties at the threshold may flag slightly
    more than ``capacity``; that is documented, not hidden.
    """
    if not 0.0 < capacity <= 1.0:
        raise ValueError("capacity must be in (0, 1]")
    if not scores:
        return []
    ordered = sorted(scores)
    idx = min(len(ordered) - 1, int((1.0 - capacity) * len(ordered)))
    threshold = ordered[idx]
    return [1 if s >= threshold else 0 for s in scores]


# ---------------------------------------------------------------- delay law


@dataclass
class DelayLaw:
    """Fitted lognormal chargeback-delay law.

    Fit by truncated maximum likelihood on visible fraud delays: a
    visible delay ``d`` at age ``a`` contributes ``log f(d) - log F(a)``.
    Only delays of visible frauds are used -- those are the delays a
    production log actually observes. ``F`` is floored at 0.05 whenever
    it is used as a divisor, as in the paper.
    """

    mu: float
    sigma: float
    n_visible_frauds: int

    def cdf(self, age: float) -> float:
        if age <= 0:
            return FLOOR_F
        z = (math.log(age) - self.mu) / self.sigma
        return max(FLOOR_F, _phi(z))

    @property
    def median(self) -> float:
        return math.exp(self.mu)


def _neg_trunc_loglik(
    mu: float, log_sigma: float, log_d: list[float], log_a: list[float]
) -> float:
    sigma = math.exp(log_sigma)
    total = 0.0
    for ld, la in zip(log_d, log_a):
        z_d = (ld - mu) / sigma
        log_f = -ld - log_sigma - 0.5 * math.log(2.0 * math.pi) - 0.5 * z_d * z_d
        f_a = _phi((la - mu) / sigma)
        if f_a <= 0.0:
            f_a = 1e-300
        total += log_f - math.log(f_a)
    return -total


def _nelder_mead_2d(
    fn, x0: tuple[float, float], step: float = 0.1, max_iter: int = 400
) -> tuple[float, float]:
    """Tiny deterministic Nelder-Mead for the 2-parameter delay fit."""
    simplex = [x0, (x0[0] + step, x0[1]), (x0[0], x0[1] + step)]
    vals = [fn(*p) for p in simplex]
    for _ in range(max_iter):
        order = sorted(range(3), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[2] - vals[0]) < 1e-10:
            break
        cx = 0.5 * (simplex[0][0] + simplex[1][0])
        cy = 0.5 * (simplex[0][1] + simplex[1][1])
        worst = simplex[2]
        rx = (cx + (cx - worst[0]), cy + (cy - worst[1]))
        fr = fn(*rx)
        if fr < vals[0]:
            ex = (cx + 2.0 * (cx - worst[0]), cy + 2.0 * (cy - worst[1]))
            fe = fn(*ex)
            simplex[2], vals[2] = (ex, fe) if fe < fr else (rx, fr)
        elif fr < vals[1]:
            simplex[2], vals[2] = rx, fr
        else:
            kx = (cx + 0.5 * (worst[0] - cx), cy + 0.5 * (worst[1] - cy))
            fk = fn(*kx)
            if fk < vals[2]:
                simplex[2], vals[2] = kx, fk
            else:
                for i in (1, 2):
                    simplex[i] = (
                        simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0]),
                        simplex[0][1] + 0.5 * (simplex[i][1] - simplex[0][1]),
                    )
                    vals[i] = fn(*simplex[i])
    best = min(range(3), key=lambda i: vals[i])
    return simplex[best]


def fit_delay_law(log: OrderLog) -> DelayLaw:
    """Fit the lognormal delay law on visible frauds.

    Uses all completed visible frauds, including orders younger than
    ``a_min`` -- young orders carry delay information even though they
    are excluded from the target estimand. Raises ``ValueError`` if
    fewer than 3 visible frauds exist: with no visible delays the law
    is simply not estimable, and guessing would silently bias every
    corrected estimate.
    """
    pairs = [
        (log.delays[i], log.ages[i])
        for i in range(len(log))
        if log.completed[i] == 1 and log.visible[i] == 1
    ]
    if len(pairs) < 3:
        raise ValueError(
            f"cannot fit delay law: only {len(pairs)} visible fraud(s); "
            "need at least 3"
        )
    log_d = [math.log(d) for d, _ in pairs]
    log_a = [math.log(max(a, 1e-6)) for _, a in pairs]
    mu0 = sum(log_d) / len(log_d)
    var0 = sum((v - mu0) ** 2 for v in log_d) / len(log_d)
    sig0 = max(math.sqrt(var0), 0.1)
    mu, log_sigma = _nelder_mead_2d(
        lambda m, ls: _neg_trunc_loglik(m, ls, log_d, log_a),
        (mu0, math.log(sig0)),
    )
    sigma = min(max(math.exp(log_sigma), 0.05), 5.0)
    return DelayLaw(mu=mu, sigma=sigma, n_visible_frauds=len(pairs))


# ---------------------------------------------------------------- estimators


@dataclass
class Estimate:
    """One estimated quantity.

    ``identified=False`` with ``value=None`` means the data cannot
    identify this quantity -- the canonical case is a candidate flagging
    declined orders in a zero-exploration stratum. That is a property
    of the design, not an error, and it must surface, not crash.
    """

    value: float | None
    identified: bool
    method: str
    n_target: int
    note: str = ""


def _target_indices(log: OrderLog, a_min: float) -> list[int]:
    return [i for i in range(len(log)) if log.ages[i] >= a_min]


def _delay_corrected_label(log: OrderLog, i: int, law: DelayLaw) -> float:
    """Y_vis / F(a): an unbiased stand-in for the true label Y."""
    return log.visible[i] / law.cdf(log.ages[i])


def _contribution(
    log: OrderLog, i: int, flags: list[int], lam: float, label: float
) -> float:
    """g (or g_tilde) for one order given a label value."""
    return flags[i] * log.values[i] * ((1.0 + lam) * label - lam)


def _unidentified(log: OrderLog, flags: list[int], target: list[int]) -> bool:
    """True when a flagged target order sits in a zero-propensity stratum."""
    return any(flags[i] == 1 and log.propensity[i] <= 0.0 for i in target)


def naive_net_savings(
    log: OrderLog,
    flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
) -> Estimate:
    """Visible-label average over completed target orders.

    Descriptive only. It inherits both distortions: declined orders are
    missing, and immature labels undercount fraud. Compare against
    :func:`ground_truth_net_savings` to see the bias.
    """
    target = _target_indices(log, a_min)
    completed = [i for i in target if log.completed[i] == 1]
    if not completed:
        return Estimate(None, True, "naive", len(target), "no completed orders")
    total = sum(
        _contribution(log, i, flags, lam, float(log.visible[i])) for i in completed
    )
    return Estimate(total / len(completed), True, "naive", len(target))


def ground_truth_net_savings(
    log: OrderLog,
    flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
) -> float:
    """Exact net savings from true labels. Simulation validation only."""
    target = _target_indices(log, a_min)
    if not target:
        return 0.0
    total = sum(
        _contribution(log, i, flags, lam, float(log.fraud[i])) for i in target
    )
    return total / len(target)


def ipw_net_savings(
    log: OrderLog,
    flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
    delay_law: DelayLaw | None = None,
    clip: float = DEFAULT_CLIP,
) -> Estimate:
    """Delay-corrected inverse-propensity estimate of net savings."""
    target = _target_indices(log, a_min)
    if not target:
        return Estimate(None, True, "ipw", 0, "empty target population")
    if _unidentified(log, flags, target):
        return Estimate(
            None,
            False,
            "ipw",
            len(target),
            "candidate flags declined orders with zero exploration propensity; "
            "see partial_identification_bounds",
        )
    law = delay_law if delay_law is not None else fit_delay_law(log)
    total = 0.0
    for i in target:
        if log.completed[i] == 0:
            continue
        w = min(1.0 / log.propensity[i], clip)
        total += w * _contribution(
            log, i, flags, lam, _delay_corrected_label(log, i, law)
        )
    return Estimate(total / len(target), True, "ipw", len(target))


# ------------------------------------------------------------ outcome model


@dataclass
class OutcomeModel:
    """P(fraud | X) as a logistic model fitted on completed orders.

    The fitted probability is for a *visible* fraud label,
    ``sigmoid(x . beta) * F(a)``; ``fraud_probability`` strips the ``F(a)``
    factor back off, giving the underlying fraud probability the DR
    estimator needs. Features are standardized with the completed-order
    mean/std stored here.
    """

    intercept: float
    coef: list[float]
    mean: list[float]
    std: list[float]

    def fraud_probability(self, x: list[float]) -> float:
        z = self.intercept + sum(
            b * (v - m) / s
            for b, v, m, s in zip(self.coef, x, self.mean, self.std)
        )
        return _sigmoid(z)


def fit_outcome_model(
    log: OrderLog,
    delay_law: DelayLaw,
    clip: float = DEFAULT_CLIP,
    max_iter: int = 300,
    ridge: float = 1e-3,
) -> OutcomeModel:
    """Weighted logistic fit on completed orders (paper Appendix B.1).

    Maximizes ``sum w_i [y_vis log q_i + (1 - y_vis) log (1 - q_i)]``
    with ``q_i = sigmoid(x_i . beta) * F(a_i)`` and weights
    ``w_i = R_i / e_i`` (clipped). A small ridge penalty stabilizes the
    fit. Optimized by Adam; deterministic, no randomness. Fitted without
    cross-fitting, as in the paper -- so DR intervals that use this model
    are approximate.
    """
    rows = [i for i in range(len(log)) if log.completed[i] == 1]
    if not rows:
        raise ValueError("cannot fit outcome model: no completed orders")
    d = len(log.features[0])
    mean = [sum(log.features[i][j] for i in rows) / len(rows) for j in range(d)]
    std = []
    for j in range(d):
        var = sum((log.features[i][j] - mean[j]) ** 2 for i in rows) / len(rows)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    xs = [
        [(log.features[i][j] - mean[j]) / std[j] for j in range(d)] for i in rows
    ]
    fs = [delay_law.cdf(log.ages[i]) for i in rows]
    ws = [min(1.0 / log.propensity[i], clip) for i in rows]
    ys = [float(log.visible[i]) for i in rows]

    beta0 = 0.0
    beta = [0.0] * d
    m0, v0 = 0.0, 0.0
    m = [0.0] * d
    v = [0.0] * d
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05
    for t in range(1, max_iter + 1):
        g0 = 0.0
        g = [0.0] * d
        for x, f, w, y in zip(xs, fs, ws, ys):
            p = _sigmoid(beta0 + sum(b * v for b, v in zip(beta, x)))
            q = min(max(p * f, 1e-9), 1.0 - 1e-9)
            coeff = w * (y - q) / (q * (1.0 - q)) * f * p * (1.0 - p)
            g0 += coeff
            for j in range(d):
                g[j] += coeff * x[j]
        m0 = b1 * m0 + (1.0 - b1) * g0
        v0 = b2 * v0 + (1.0 - b2) * g0 * g0
        beta0 += lr * (m0 / (1.0 - b1**t)) / (
            math.sqrt(v0 / (1.0 - b2**t)) + eps
        )
        for j in range(d):
            gj = g[j] - 2.0 * ridge * beta[j]
            m[j] = b1 * m[j] + (1.0 - b1) * gj
            v[j] = b2 * v[j] + (1.0 - b2) * gj * gj
            beta[j] += lr * (m[j] / (1.0 - b1**t)) / (
                math.sqrt(v[j] / (1.0 - b2**t)) + eps
            )
    return OutcomeModel(intercept=beta0, coef=beta, mean=mean, std=std)


def _dr_pseudo_values(
    log: OrderLog,
    flags: list[int],
    lam: float,
    a_min: float,
    law: DelayLaw,
    model: OutcomeModel,
    clip: float,
) -> tuple[list[int], list[float]]:
    """Per-target-order DR pseudo-values for one candidate."""
    target = _target_indices(log, a_min)
    values: list[float] = []
    for i in target:
        mu = _contribution(
            log, i, flags, lam, model.fraud_probability(log.features[i])
        )
        if log.completed[i] == 1:
            w = min(1.0 / log.propensity[i], clip)
            g_tilde = _contribution(
                log, i, flags, lam, _delay_corrected_label(log, i, law)
            )
            values.append(mu + w * (g_tilde - mu))
        else:
            values.append(mu)
    return target, values


def dr_net_savings(
    log: OrderLog,
    flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
    delay_law: DelayLaw | None = None,
    clip: float = DEFAULT_CLIP,
    max_iter: int = 300,
    model: OutcomeModel | None = None,
) -> Estimate:
    """Delay-corrected doubly robust estimate of net savings."""
    target = _target_indices(log, a_min)
    if not target:
        return Estimate(None, True, "dr", 0, "empty target population")
    if _unidentified(log, flags, target):
        return Estimate(
            None,
            False,
            "dr",
            len(target),
            "candidate flags declined orders with zero exploration propensity; "
            "see partial_identification_bounds",
        )
    law = delay_law if delay_law is not None else fit_delay_law(log)
    model = (
        model
        if model is not None
        else fit_outcome_model(log, law, clip=clip, max_iter=max_iter)
    )
    _, values = _dr_pseudo_values(log, flags, lam, a_min, law, model, clip)
    return Estimate(sum(values) / len(values), True, "dr", len(target))


def paired_advantage(
    log: OrderLog,
    challenger_flags: list[int],
    champion_flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
    delay_law: DelayLaw | None = None,
    clip: float = DEFAULT_CLIP,
    max_iter: int = 300,
    model: OutcomeModel | None = None,
) -> tuple[float, float]:
    """Paired DR advantage (challenger - champion) and its standard error.

    Uses one shared delay law and one outcome model per candidate, with
    per-order pseudo-value differences -- the pairing is what makes the
    gate's standard error small relative to unpaired comparisons.
    """
    law = delay_law if delay_law is not None else fit_delay_law(log)
    model_c = (
        model
        if model is not None
        else fit_outcome_model(log, law, clip=clip, max_iter=max_iter)
    )
    # The outcome model predicts P(fraud | X), independent of the
    # candidate; only the flag factor differs, so one model serves both.
    _, psi_j = _dr_pseudo_values(
        log, challenger_flags, lam, a_min, law, model_c, clip
    )
    _, psi_c = _dr_pseudo_values(
        log, champion_flags, lam, a_min, law, model_c, clip
    )
    diffs = [a - b for a, b in zip(psi_j, psi_c)]
    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0
    if n < 2:
        return mean, 0.0
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(var / n)


def promotion_gate(
    advantage: float,
    se: float,
    alpha: float = DEFAULT_ALPHA,
    n_challengers: int = 1,
) -> bool:
    """Promote only when the lower confidence bound on the advantage is
    positive. With ``n_challengers`` screened, the level is Bonferroni
    corrected to ``alpha / n_challengers`` (paper Section 4.3)."""
    if n_challengers < 1:
        raise ValueError("n_challengers must be >= 1")
    level = alpha / n_challengers
    return advantage - _z_value(level) * se > 0.0


# ------------------------------------------------------- precision and AUC


def precision_at_capacity(
    log: OrderLog,
    flags: list[int],
    a_min: float = DEFAULT_A_MIN,
    delay_law: DelayLaw | None = None,
    method: str = "ipw",
    clip: float = DEFAULT_CLIP,
) -> Estimate:
    """Precision among flagged orders: fraction of flagged that are fraud.

    ``naive`` averages visible labels over completed flagged orders.
    ``ipw`` weights flagged orders by ``R / e`` and uses delay-corrected
    labels, so it estimates precision on the full flagged population.
    """
    target = _target_indices(log, a_min)
    if method not in ("naive", "ipw"):
        raise ValueError("method must be 'naive' or 'ipw'")
    if method == "naive":
        flagged = [i for i in target if flags[i] == 1 and log.completed[i] == 1]
        if not flagged:
            return Estimate(None, True, "precision-naive", len(target),
                            "no completed flagged orders")
        total = sum(float(log.visible[i]) for i in flagged)
        return Estimate(total / len(flagged), True, "precision-naive",
                        len(target))
    if _unidentified(log, flags, target):
        return Estimate(None, False, "precision-ipw", len(target),
                        "flagged declined orders with zero propensity")
    law = delay_law if delay_law is not None else fit_delay_law(log)
    num = 0.0
    den = 0.0
    for i in target:
        if flags[i] == 0 or log.completed[i] == 0:
            continue
        w = min(1.0 / log.propensity[i], clip)
        num += w * _delay_corrected_label(log, i, law)
        den += w
    if den <= 0.0:
        return Estimate(None, True, "precision-ipw", len(target),
                        "no flagged orders in target population")
    return Estimate(num / den, True, "precision-ipw", len(target))


def ground_truth_precision(
    log: OrderLog, flags: list[int], a_min: float = DEFAULT_A_MIN
) -> float:
    """Exact precision from true labels. Simulation validation only."""
    target = [i for i in _target_indices(log, a_min) if flags[i] == 1]
    if not target:
        return 0.0
    return sum(float(log.fraud[i]) for i in target) / len(target)


def _weighted_auc(
    scores: list[float], labels: list[int], weights: list[float]
) -> float | None:
    """Mann-Whitney AUC with weights; ties get 0.5 credit."""
    pos = sum(w for l, w in zip(labels, weights) if l == 1)
    neg = sum(w for l, w in zip(labels, weights) if l == 0)
    if pos <= 0.0 or neg <= 0.0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    cum_neg = 0.0
    concordant = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        group = order[i:j]
        pos_w = sum(weights[k] for k in group if labels[k] == 1)
        neg_w = sum(weights[k] for k in group if labels[k] == 0)
        concordant += pos_w * (cum_neg + 0.5 * neg_w)
        cum_neg += neg_w
        i = j
    return concordant / (pos * neg)


def auc_score(
    log: OrderLog,
    scores: list[float],
    a_min: float = DEFAULT_A_MIN,
    method: str = "ipw",
    clip: float = DEFAULT_CLIP,
) -> Estimate:
    """AUC of a score against fraud status.

    ``naive`` uses visible labels on completed orders only. ``ipw``
    weights completed orders by ``R / e``. Both use visible labels, so
    the IPW variant corrects selection but is delay-approximate: for
    delay-sensitive AUC work, raise ``a_min`` so most labels have
    matured (``F(a)`` near 1) instead of relying on this estimate at
    young ages.
    """
    target = _target_indices(log, a_min)
    if method not in ("naive", "ipw"):
        raise ValueError("method must be 'naive' or 'ipw'")
    if method == "naive":
        idx = [i for i in target if log.completed[i] == 1]
        value = _weighted_auc(
            [scores[i] for i in idx],
            [log.visible[i] for i in idx],
            [1.0] * len(idx),
        )
        return Estimate(value, True, "auc-naive", len(target),
                        "" if value is not None else "no pos/neg examples")
    idx = target
    weights = [
        min(1.0 / log.propensity[i], clip) if log.completed[i] == 1 else 0.0
        for i in idx
    ]
    value = _weighted_auc(
        [scores[i] for i in idx],
        [log.visible[i] for i in idx],
        weights,
    )
    return Estimate(value, True, "auc-ipw", len(target),
                    "" if value is not None else "no pos/neg examples")


def ground_truth_auc(
    log: OrderLog, scores: list[float], a_min: float = DEFAULT_A_MIN
) -> float | None:
    """Exact AUC from true labels. Simulation validation only."""
    target = _target_indices(log, a_min)
    return _weighted_auc(
        [scores[i] for i in target],
        [log.fraud[i] for i in target],
        [1.0] * len(target),
    )


# ------------------------------------------------------------- gate bounds


def partial_identification_bounds(
    log: OrderLog,
    flags: list[int],
    lam: float = DEFAULT_LAMBDA,
    a_min: float = DEFAULT_A_MIN,
    delay_law: DelayLaw | None = None,
) -> tuple[float, float]:
    """Sharp bounds on net savings without exploration (paper Prop. 2).

    ``[E[1{R=1} g_tilde] - lam * E[1{R=0} phi V],
       E[1{R=1} g_tilde] + E[1{R=0} phi V]]`` over the target population,
    with delay-corrected completed contributions. Width is
    ``(1 + lam) * E[1{R=0} phi V]``: it grows with the flagged value
    sitting in the declined region, which is exactly why exploration is
    needed before a promotion decision.
    """
    target = _target_indices(log, a_min)
    if not target:
        return (0.0, 0.0)
    law = delay_law if delay_law is not None else fit_delay_law(log)
    identified_part = 0.0
    unidentified_value = 0.0
    for i in target:
        if log.completed[i] == 1:
            identified_part += _contribution(
                log, i, flags, lam, _delay_corrected_label(log, i, law)
            )
        else:
            unidentified_value += flags[i] * log.values[i]
    n = len(target)
    lo = identified_part / n - lam * unidentified_value / n
    hi = identified_part / n + unidentified_value / n
    return (lo, hi)

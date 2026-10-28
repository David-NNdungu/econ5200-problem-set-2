# Phase 4 — Sequential-testing record
**David Ndungu · ECON 5200**  
**Model:** ChatGPT / Codex  
**Date:** October 27, 2026

## My aim
I need a test that lets the client monitor an experiment as observations arrive
without repeatedly applying an ordinary fixed-look p-value. I use the P.R.I.M.E.
framework to specify the task, review the first design, refine its assumptions
and check the result.

I retain the two design specifications and the returned Python code below.
My comments explain what changed and why. The code blocks preserve the generated
source; the surrounding discussion is my revised review of the two design
iterations from the working session.

## 1. Initial specification
```text
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.
```

## 2. Initial returned code
```python
"""Initial AI design: two-sided normal-mixture test on standardized observations."""
import numpy as np


def sequential_v1(observations, alpha=0.05, rho=312.5):
    """Inputs must be independent N(mu, 1); outputs latch after rejection."""
    x = np.asarray(observations, dtype=float)
    t = np.arange(1, len(x) + 1)
    s = np.cumsum(x)
    log_e = 0.5 * np.log(rho / (rho + t)) + s**2 / (2 * (rho + t))
    return np.maximum.accumulate(log_e >= np.log(1 / alpha))
```

## 3. My review and revised specification
My shipping question concerns positive effects. The initial symmetric mixture
can reject a large negative effect, so I cannot use every rejection as a ship
decision. It also assumes unit-variance observations, while my paired revenue
differences have known SD sqrt(2) × 10.

My revised specification fixes the direction, raw scale, log columns, look
schedule and seed policy. I also ask for uncertainty around the null rejection
rate and distinguish an error bound from an exact calibration target.

```text
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.

[Revision for this portfolio]
The log columns are experiment_id, true_effect, is_null, n_per_arm,
n_total, estimate, se, p_value, design_effect, design_power, oracle_power,
client_ship and power_tier. A deployable rule cannot read truth.
Each arm has known SD 10; paired differences have known SD sqrt(2)*10.
The shipping alternative is POSITIVE. Version 1 is symmetric and assumes
unit variance, leaving raw-scale log integration unspecified.

Use a positive half-normal mixture, accept known sigma, fix
rho=(sqrt(2)*10/0.80)**2=312.5 before testing, monitor every observation,
and return log evidence, first crossing and latched decisions.
Validate inputs and optional increasing integer looks; use log_ndtr.

With seed 20261027 and one generator, generate 1,000 independent
5,000-observation pure-null streams. Compare both versions, naive
one-sided p<.05 at any look, and fixed-final-look p<.05 on the same streams.
Report counts, rates, Monte Carlo SEs and an exact binomial interval.
Interpret 5% as an upper bound, not equality; do not tune rho or seeds.

Replay the 200-test log with conditional Gaussian residual paths
preserving its original terminal estimates and SEs. Label these as
simulated possible histories, separate from the unconditional null check.
Test direction, scale, latching, indexing and invalid inputs.
Explain the stopping lines and why revisions stop.
```

## 4. Revised returned code
```python
"""One-sided Gaussian mixture sequential test; NumPy and SciPy only.

Known variance is essential. Standardize independent treatment-minus-control
observations using a pre-specified SD, never the running estimated SD.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr


def sequential_test(
    observations: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
    rho: float = 312.5,
    looks: np.ndarray | None = None,
) -> dict[str, np.ndarray | int | bool | None]:
    """Test H0: mean <= 0 against mean > 0 at specified cumulative looks.

    Under independent Gaussian observations with known SD sigma, the half-normal
    mixture is an e-process. P(ever reject a true null) <= alpha, including
    data-dependent stopping. rho is the inverse variance of the half-normal
    mixing distribution on the standardized mean and must be fixed in advance.
    Returns cumulative sample sizes, log e-values, latched decisions, and first
    crossing. The first crossing rejects; no crossing means no rejection.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("observations must be a finite nonempty 1D array")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not (0 < alpha < 1) or not np.isfinite(rho) or rho <= 0:
        raise ValueError("require 0 < alpha < 1 and finite rho > 0")
    if looks is None:
        t = np.arange(1, x.size + 1, dtype=int)
    else:
        raw = np.asarray(looks)
        if (raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all()
                or np.any(raw != np.floor(raw))):
            raise ValueError("looks must be a nonempty array of integer counts")
        t = raw.astype(int)
        if t[0] < 1 or t[-1] > x.size or np.any(np.diff(t) <= 0):
            raise ValueError("looks must be strictly increasing within the stream")
    s = np.cumsum(x / sigma)[t - 1]
    v = rho + t
    # Integral over theta >= 0 of exp(theta*S - t*theta**2/2)
    # against a HalfNormal(scale=1/sqrt(rho)) mixing density.
    log_e = (np.log(2.0) + 0.5 * (np.log(rho) - np.log(v))
             + s**2 / (2.0 * v) + log_ndtr(s / np.sqrt(v)))
    crossing = log_e >= -np.log(alpha)
    reject_by_look = np.maximum.accumulate(crossing)
    first = int(t[np.flatnonzero(crossing)[0]]) if crossing.any() else None
    return {"n": t, "log_e": log_e, "reject_by_look": reject_by_look,
            "first_reject_n": first, "rejected": bool(crossing.any())}


def replay_portfolio(
    log: object, sigma_pair: float, seed: int, alpha: float = 0.05,
    rho: float = 312.5,
) -> list[dict[str, int | float | bool | None]]:
    """Reconstruct Gaussian paths conditional on the SAME logged final estimates.

    log must provide itertuples() with experiment_id, n_per_arm, estimate, se.
    A normal residual vector minus its mean, plus the logged estimate, is a
    conditional Gaussian path whose terminal mean equals that estimate. This
    uses extra simulated path information, not recovered real historical data.
    """
    rng = np.random.default_rng(seed)
    out = []
    for row in log.itertuples(index=False):
        n = int(row.n_per_arm)
        if n < 2 or not np.isclose(row.se, sigma_pair / np.sqrt(n)):
            raise ValueError("log standard errors do not match known sigma_pair")
        residual = rng.normal(0.0, sigma_pair, size=n)
        stream = residual - residual.mean() + row.estimate
        assert np.isclose(stream.mean(), row.estimate)
        result = sequential_test(stream, sigma_pair, alpha, rho)
        out.append({"experiment_id": int(row.experiment_id),
                    "estimate": float(stream.mean()), "n_per_arm": n,
                    "sequential_ship": bool(result["rejected"]),
                    "first_reject_n": result["first_reject_n"]})
    return out
```

## 5. My changes and reasons
| Change | My reason |
|---|---|
| Positive half-normal mixture | My ship decision needs evidence of a positive effect. |
| Explicit known sigma | My raw paired differences are not unit-variance inputs. |
| Rho fixed at 312.5 | I set the mixture scale from my planning effect before inspecting null results. |
| Input and integer-look checks | I need invalid streams and look schedules to fail clearly. |
| Log-space evidence and log_ndtr | I avoid overflow and unstable tail calculations. |
| First crossing and latched decisions | I need a stopping time and a decision that persists after stopping. |
| Conditional Gaussian path replay | I preserve my original terminal estimates and sample sizes. |
| Four methods on the same null streams | I compare error rates without changing the simulated inputs between methods. |
| Monte Carlo SE and exact binomial interval | I report sampling noise in the measured rejection rate. |
| Direction, scaling, latching and indexing checks | I check whether the implementation follows the intended rule. |

The revised source is the same source saved in src/sequential_test.py. My
notebook writes that module and imports it before checking the test.

## 6. My verification results
I compare the methods on 1,000 pure-null streams, with 5,000 paired observations
per stream. I use seed 20261027 and a single generator so streams do not repeat
because of accidental reseeding. These are the executed results from my notebook.

### Verification code

```python
from src import sequential_test as st
importlib.reload(st)
R, max_n = CONFIG['sequential_R'], CONFIG['sequential_max_n']
null_rng = np.random.default_rng(CONFIG['sequential_seed'])
sigma_pair = np.sqrt(2)*CONFIG['sigma']
counts = {'Version 1: symmetric mixture':0, 'Version 2: positive mixture':0,
          'Naive: p < .05 at any look':0, 'Fixed horizon: final p < .05':0}
first_stop = []
for rep in range(R):
    # Different independent streams from ONE generator, never reset inside loop.
    raw = null_rng.normal(0,sigma_pair,size=max_n)
    standardized = raw/sigma_pair
    counts['Version 1: symmetric mixture'] += bool(sequential_v1(
        standardized,CONFIG['alpha'],CONFIG['rho'])[-1])
    seq_result = st.sequential_test(raw,sigma_pair,CONFIG['alpha'],CONFIG['rho'])
    counts['Version 2: positive mixture'] += seq_result['rejected']
    first_stop.append(seq_result['first_reject_n'])
    z_path = np.cumsum(standardized)/np.sqrt(np.arange(1,max_n+1))
    p_path = stats.norm.sf(z_path)
    counts['Naive: p < .05 at any look'] += np.any(p_path < CONFIG['alpha'])
    counts['Fixed horizon: final p < .05'] += p_path[-1] < CONFIG['alpha']
null_table = pd.DataFrame({'false_positives':counts})
null_table['streams'] = R
null_table['rate'] = null_table.false_positives/R
null_table['Monte_Carlo_SE'] = np.sqrt(null_table.rate*(1-null_table.rate)/R)
null_table['nominal_upper_bound'] = CONFIG['alpha']
display(null_table)
seq_count = counts['Version 2: positive mixture']
seq_rate = seq_count/R
seq_mcse = np.sqrt(seq_rate*(1-seq_rate)/R)
seq_exact_ci = stats.binomtest(seq_count,R).proportion_ci(.95,method='exact')
boundary_mcse = np.sqrt(CONFIG['alpha']*(1-CONFIG['alpha'])/R)
print('Version 2 exact 95% binomial interval:',tuple(seq_exact_ci))
print('MC SE under a hypothetical exactly 5% null rate:',boundary_mcse)
print('One-sided binomial evidence of EXCEEDING the bound:',
    stats.binomtest(seq_count,R,CONFIG['alpha'],alternative='greater').pvalue)
display(Markdown(f'**Measured check:** {seq_count}/{R} = **{seq_rate:.1%}** '
    f'(MC SE **{seq_mcse:.2%}**, or {100*seq_mcse:.2f} percentage points). '
    f'The exact binomial interval is [{seq_exact_ci.low:.2%}, {seq_exact_ci.high:.2%}]. '
    'This finite-horizon test is conservative relative to the 5% ceiling. '
    'A low rate does not violate an upper-bound guarantee; it is not evidence of exact 5% calibration.'))
```
```text
Version 2 exact 95% binomial interval: (0.01950593732438462, 0.04138471911644505)
MC SE under a hypothetical exactly 5% null rate: 0.006892024376045111
One-sided binomial evidence of EXCEEDING the bound: 0.9996082859980967

```
```text
                              false_positives  streams   rate  Monte_Carlo_SE  \
Version 1: symmetric mixture               31     1000  0.031          0.0055   
Version 2: positive mixture                29     1000  0.029          0.0053   
Naive: p < .05 at any look                521     1000  0.521          0.0158   
Fixed horizon: final p < .05               49     1000  0.049          0.0068   

                              nominal_upper_bound  
Version 1: symmetric mixture                 0.05  
Version 2: positive mixture                  0.05  
Naive: p < .05 at any look                   0.05  
Fixed horizon: final p < .05                 0.05  
```
**Measured check:** 29/1000 = **2.9%** (MC SE **0.53%**, or 0.53 percentage points). The exact binomial interval is [1.95%, 4.14%]. This finite-horizon test is conservative relative to the 5% ceiling. A low rate does not violate an upper-bound guarantee; it is not evidence of exact 5% calibration.

### Verification code

```python
# Direction, scale, latching, look schedule, and invalid-input checks.
zero = st.sequential_test(np.zeros(100),sigma_pair,CONFIG['alpha'],CONFIG['rho'])
assert not zero['rejected']
positive = st.sequential_test(np.full(300,3*sigma_pair),sigma_pair,CONFIG['alpha'],CONFIG['rho'])
negative = st.sequential_test(np.full(300,-3*sigma_pair),sigma_pair,CONFIG['alpha'],CONFIG['rho'])
assert positive['rejected'] and not negative['rejected']
assert sequential_v1(np.full(300,-3.0),CONFIG['alpha'],CONFIG['rho'])[-1]
scale_probe = np.random.default_rng(8801).normal(size=400)
a = st.sequential_test(scale_probe,1,CONFIG['alpha'],CONFIG['rho'])
b = st.sequential_test(scale_probe*sigma_pair,sigma_pair,CONFIG['alpha'],CONFIG['rho'])
assert np.allclose(a['log_e'],b['log_e'])
assert np.all(np.diff(positive['reject_by_look'].astype(int)) >= 0)
chosen_looks = np.array([10,50,100,200,400])
subset = st.sequential_test(scale_probe,1,CONFIG['alpha'],CONFIG['rho'],chosen_looks)
assert np.allclose(subset['log_e'],a['log_e'][chosen_looks-1])
for kwargs in [{'sigma':0}, {'sigma':1,'looks':np.array([2,1])}]:
    try:
        st.sequential_test(np.ones(10),**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input was accepted')
print('Eight sequential behavior checks passed.')

# Reuse the SAME 200 terminal estimates, SEs, sample sizes and truths.
paths = pd.DataFrame(st.replay_portfolio(log,sigma_pair,CONFIG['path_seed'],
    CONFIG['alpha'],CONFIG['rho']))
assert np.allclose(paths.estimate,log.estimate)
sequential_log = log.merge(paths[['experiment_id','sequential_ship','first_reject_n']],
    on='experiment_id',validate='one_to_one')
seq_portfolio = audit_mask(sequential_log,sequential_log.sequential_ship)
display(pd.DataFrame({'Fixed horizon':audit_mask(log,log.client_ship),
                      'Always-valid monitoring':seq_portfolio}).T)
log.to_csv('output/experiment_log.csv',index=False)
sequential_log.to_csv('output/sequential_log.csv',index=False)
null_table.to_csv('output/null_check.csv')
comparison.to_csv('output/rule_comparison.csv')
```
```text
Eight sequential behavior checks passed.

```
```text
                         realised_return  ship_count  false_ship_count  \
Fixed horizon                    16.1460        26.0               8.0   
Always-valid monitoring          12.3105        14.0               1.0   

                         false_ship_rate  
Fixed horizon                     0.3077  
Always-valid monitoring           0.0714  
```
## 7. My interpretation
The positive-mixture test rejects 29 of 1,000 null streams: **2.9%**, with
Monte Carlo SE **0.53 percentage points**. Its exact 95% binomial interval is
approximately **1.95%–4.14%**. The nominal claim is a probability of ever rejecting
a true null of **at most 5%**. My finite-horizon result is conservative; it is
not evidence that the method should be tuned until it rejects exactly 5%.

The ordinary final-look test rejects 4.9%, while repeated ordinary testing
rejects 52.1%. This comparison shows why the client's habit of stopping when
results look good needs a different boundary.

My test assumes independent Gaussian paired differences with known variance.
It does not establish validity for unknown variance or dependent client data,
and it does not control false discoveries across all experiments.

## 8. My explanation of stopping
I compare log_e with -log(alpha) to decide whether the evidence crosses the
boundary. I use maximum.accumulate to keep the rejection true after its first
crossing, even if later evidence falls. The first crossing supplies the stopping
sample size; a stream with no crossing has not rejected the null.

## 9. My judgment
My direction, scale, latching and look-schedule checks pass, and the null check
does not show error inflation under the stated model. I stop revising at that
point instead of changing the seed or mixture scale to get a preferred rate.

I would use AI for this implementation again, provided I review the assumptions
and test the output. My main responsibility is deciding whether the statistical
question, input scale and stopping rule match the client's process.

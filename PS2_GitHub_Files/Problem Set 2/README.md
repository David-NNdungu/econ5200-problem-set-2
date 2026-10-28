# The Underpowered Portfolio
**David Ndungu · ECON 5200 · Problem Set 2**

## My objective
I investigate why a client's reported experiment wins overstate the revenue
they deliver. I reconstruct a simulated portfolio, measure the winner's curse,
compare a power-floor rule and check an always-valid sequential test.

## My approach
- I simulate 200 experiments with known effects, varying sample sizes and a fixed seed.
- I audit the client's positive-direction p < 0.05 rule.
- I compare estimated and true effects across planning-power tiers.
- I bootstrap whole experiment rows, verify my loop against SciPy and report BCa.
- I evaluate a power floor on the same portfolio and state its return tradeoff.
- I compare sequential and ordinary tests on 1,000 pure-null streams.

## My findings
- I ship 26 tests, of which 18 are genuinely positive. My false-ship share is 30.8%.
- My mean selected estimate is 2.12 times the mean true effect.
- I forecast $34,205 in annual revenue but realise $16,146, an overstatement of $18,059.
- My power floor lowers false-ship share to 12.5%, but realised annual revenue falls to $5,736. It does not win on total return.
- My manual bootstrap matches SciPy replicate-for-replicate. My 95% BCa interval is [10.019, 24.711] in $1,000 units for expected 200-experiment programme return.
- My always-valid null rejection rate is 2.9%, with Monte Carlo SE 0.53 percentage points. Ordinary repeated peeking rejects 52.1%.

## My files
| File | Purpose |
|---|---|
| Econ_5200_PS2.ipynb | My executed analysis, plots, memo and final numbers cell |
| ai-appendix.md | My Phase 4 specifications, returned code, revisions and checks |
| src/experiment_portfolio.py | My reusable simulation, rules and evaluation functions |
| src/sequential_test.py | My sequential test and original-log replay adapter |
| output/ | My simulated logs, tables, figures and memo |

## Reproducing my results
I use Python with NumPy, pandas, SciPy and matplotlib. My delivered run used
NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0. The code requires SciPy 1.15 or newer
for the bootstrap rng argument.

My notebook writes both Python modules, so I can open it in a fresh Colab
session and run every cell in order. I keep seeds and parameters in Setup.
My final code cell prints the numbers used in my memo beside their source steps.

## My interpretation
My data and monetary results are simulated. I measure forecast overstatement,
not implementation spending. My conclusions depend on known Gaussian variance,
independence, a zero-or-positive prior, and additive effects at a common rollout
exposure. I would revisit my recommendation if harmful effects or meaningful
deployment costs changed the decision problem.

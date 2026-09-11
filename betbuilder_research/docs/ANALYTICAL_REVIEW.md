# Analytical review and proposed extensions

## What the current copula analysis means

The current dependence calculation pools all eligible player-games. In the reference
sample, Spearman correlations include points/field-goal percentage 0.4755,
points/three-point percentage 0.4676, points/assists 0.4539, and points/rebounds 0.4145.
These are descriptive rank associations, not causal effects, conditional forecasts,
or the probability that two prop legs both win. Between-player ability/role/minutes
mixing can differ greatly from within-player dependence.

The code selects eight available variables from its proposed nine-variable list:
`MIN` does not exist; preparation creates `minutes_played`. Thus minutes are absent
from this specific copula table despite their relevance. There are 28 unordered pairs.
This is preserved rather than silently adding a new variable.

`Upper_Tail_Dep` is Pearson correlation computed after restricting the **first** variable
to its upper decile; `Lower_Tail_Dep` uses its lower decile. These measures are asymmetric
in variable ordering and can be negative or undefined. They are not the copula quantities
P(U > q | V > q) or P(U <= q | V <= q), nor their limiting tail coefficients. The legacy
column names remain for parity; chart labels and the report explain the distinction.
For example, the points/field-goal-percentage lower-tail value 0.6474 describes restricted
correlation, not a 64.74% joint-hit probability.

Rank plots use average ranks divided by n+1. Tied integer counts create bands. A diagonal
is an equal-rank reference, not independence; the legend is corrected. Spearman-vs-Pearson
thresholds used to label relationships as “Linear”, “Monotonic” or “Non-monotonic” are
heuristics, not formal model-identification tests. Tiny/zero displayed p-values are not
proof of useful predictive dependence; repeated player/team/game observations violate a
simple independent-row inference interpretation.

The Gaussian illustrations directly use Spearman rho as the latent normal correlation
and plot a bivariate-normal density on transformed coordinates without dividing by the
normal marginal densities. Clayton/Gumbel surfaces and parameter mappings are also
approximate illustrations. There is no likelihood-based family comparison or fitted
joint-event model. The original numerical surfaces are preserved and explicitly marked
illustrative. A continuous-margin Gaussian copula can map rho_s to
`2*sin(pi*rho_s/6)`; ties/discrete counts require additional care. A correct copula density
is the joint density divided by marginal densities. See the primary
[Statsmodels implementation](https://www.statsmodels.org/stable/_modules/statsmodels/distributions/copula/elliptical.html)
and [parameter mapping documentation](https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.GaussianCopula.corr_from_tau.html).

## Full workflow findings

| Component | Evidence and interpretation | Treatment in this refactor |
|---|---|---|
| Source population | 24,092 rows; Regular Season 21,353, Preseason 2,538, All-Star 116, missing type 50, Cup 30, knockout 5. Filename is not a filter. | Preserve all source types; record coverage. Define intended population as an extension. |
| Cleaning | 4,104 missing minutes; total 5,746 rows fail `numMinutes >= 5`. Both city columns have 116 missing values. Ten source wins are missing, but **none survive** the minutes filter in this dataset. | Preserve exclusion and boolean conversion. Record missingness. Missing-win conversion is a latent defect for other inputs, not a current retained-row distortion. |
| Units | Values such as 8.51 and 6.02 appear in `numMinutes`. Source provenance must establish whether these are decimal minutes or minute.second notation. | Preserve numbers; do not assert unit interpretation is verified. |
| Identity/aggregation | Player names are grouping keys; traded players get modal team assignment for some plots. Missing team names disappear from grouped summaries. | Keep for parity; recommend stable IDs and explicit trade/missing-team handling. |
| Descriptive summaries | Full-sample aggregates, two-decimal rounding, player-game-weighted home/win summaries. | Preserve; do not interpret row-weighted rates as game-level probabilities. |
| Prop thresholds | Rounded full-sample means plus fixed offsets; qualification uses same sample. | In-sample historical frequencies only, not executable market prices or independent forecasts. |
| Accumulator replay | Fixed illustrative lines; substring names match multiple players (e.g. Ball/Wagner/Bridges); absent legs are omitted. | Original template and settlement retained in JSON and table results. Do not compare variable-composition nights as the same ticket. |
| Distribution tests | Numeric IDs and duplicate aliases are included. `anderson` receives `normal`/`exponential` instead of supported short distribution names, causing these candidates to be discarded. `nbinom.fit` is unavailable. Fitted-sample KS p-values and discrete KS assumptions are inappropriate; the best-fit winner uses p-values. Shapiro uses the first 5,000 rows. | Preserve calculated results, expose failed candidates in the log, document them. Bounding the integer plotting grid changes only rendering. Do not treat the selected family as validated. |
| Entropy/MI | Histogram entropy is in bits; sklearn MI estimates are in nats. The diagonal uses histogram entropy, so units/estimators differ. Identifiers and duplicate aliases inflate apparent dependencies. | Preserve matrix, cache only exactly identical ordered input arrays. Do not read the diagonal/off-diagonal as directly comparable. |
| Experimental features | `mean_abs_edge`/`max_abs_edge` aggregate absolute realised-minus-line outcomes. | Reject these known leaked predictors at training entry. Do not silently replace them with a new model specification. |
| Experimental data joins | Market/ticket joins lack key-cardinality validation; missing outcomes can be classified as misses; pushes/voids need explicit settlement. | Documented, not included in the validated descriptive workflow. |
| Experimental preprocessing | Missing rest days use full-data median; raw boolean home flags become `home_away` but downstream code tests string `H`. | Requires separate corrected/past-only specification and tests before use. |
| Experimental simulation | Spearman correlations enter a Gaussian latent matrix directly; eigenvalue clipping lacks full correlation renormalisation, with independence fallback on failure. Marginals are fitted in-sample. | Experimental only; no claim of calibrated joint forecasts or CPU/GPU parity. |
| Experimental evaluation | Rolling date splits do not repair outcome-derived features or synthetic odds; class-weighted logistic probabilities need calibration checks. Full-series bet-sizing normalisation is retrospective. | Block known predictor leakage and synthetic defaults. No economic validation claimed. |
| Reports | Legacy standard PDF omitted interactive and distribution-fit plots and had a wrong hard-coded season. Logging wrote through competing file handles. | Save all plots; actual date range; one log stream; manifest created only after successful exports. |

## Further analysis, in priority order (proposed, not implemented)

1. **Define the research population and admissible information.** Resolve minutes units,
   official game type, unique player/game/team IDs, missingness, trade attribution and
   duplicates. Publish exclusion counts and sensitivity with/without preseason/All-Star
   and the five-minute filter. Separate source schema corrections from parity results.
2. **Build a past-only marginal forecasting baseline.** Model minutes/availability first,
   then conditional player count distributions using lagged context and partial pooling.
   Compare Poisson/negative-binomial and empirical baselines with rolling-origin log score,
   calibration and coverage. Fit preprocessing and selection inside each training window.
   Use game/date grouping and a locked final period; no shuffled validation.
3. **Evaluate incremental dependence value.** Compare independence with conditional Gaussian,
   Student-t and selected asymmetric copulas on the same forecasts and held-out events.
   Use appropriate discrete/randomised probability transforms with documented seeds,
   valid correlation matrices, and held-out joint log scores/Brier scores. Test actual
   multi-leg event probabilities, not just attractive scatterplots. Select families in
   training only. Student-t offers a distinct tail-dependence specification; see
   [Statsmodels StudentTCopula](https://www.statsmodels.org/dev/generated/statsmodels.distributions.copula.api.StudentTCopula.html).
4. **Quantify uncertainty and robustness.** Cluster/block resampling by games/dates, with
   player dependence considered; confidence intervals on correlations, finite-threshold
   co-exceedance and held-out score differences. Test role/minutes/opponent changes,
   threshold sensitivity, small samples, alternative missing-data rules, and multiple
   comparisons. Correct distribution testing with parametric bootstrap and suitable
   discrete goodness-of-fit procedures; remove IDs/aliases in an explicit extension.
5. **Only then assess economic value.** Obtain timestamped executable market prices,
   ticket specifications, stakes, settlement rules and prediction issue times. Establish
   EV separately from dependence, account for bookmaker margin, limits, costs and
   uncertainty, and evaluate a locked strategy out-of-sample. No market values should be
   invented to fill the current evidence gap.

No methodological extension above has been represented as a completed experiment.

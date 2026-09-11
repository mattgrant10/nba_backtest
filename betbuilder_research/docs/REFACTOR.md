# Refactor decisions and preservation

## Authority and scope

The actual launcher `run_nba_analysis.sh` called `data/dataedits.py` against the
February CSV. Its outputs and the recent log collection match the supplied copula/logging
request. That analysis, not filenames such as `FINAL`, is the authoritative descriptive
baseline. Existing generic modelling scripts represent a separate experimental workflow:
missing real line/ticket data and outcome-derived predictors prevent validated economic use.

The parent Git repository is `nba_backtest`, one directory above this project.
Starting branch: `dataset-compare-2025-2026`; initial HEAD: `077fa61`.
The existing `origin` remote was inspected and left unchanged. Source preservation commit:
`a85daa35b3650d3a41230d46f09976a106bb9110`.
Refactor branch: `refactor/reproducible-betbuilder-research`.

The source snapshot includes changed/untracked research code and documentation. Raw XLSX
files were deliberately excluded, including the previously staged 153 MB workbook.
Unrelated sibling-project staged files and IDE edits are preserved. Existing generated
figures/PDFs remain on disk and are removed only from the current index. Already committed
artifacts remain recoverable in Git history; history was not rewritten. Original uncommitted
artifacts and private inputs are kept locally, not represented as Git-backed preservation.

`.local_preservation/` contains the initial file SHA-256/size inventory (438 files), staged
binary patch, status/branch/remote/ignored-file inventories, original baseline results,
logs and comparison tables. This directory is private and ignored. It also preserves the
original staged state of the workbook without adding it to a commit. Source files omitted
from the maintained workflow remain recoverable from the preservation commit.

## Baseline before restructuring

The untouched source ran on the actual input until plotting the `personId` Poisson support
allocated almost 196 million integer points. That process was stopped. The necessary repair
bounds the discrete plotting grid at 2,000 unique integers. It changes neither input rows,
fitted parameters, test statistics, rankings nor table values. The repaired source completed
in 208.816 seconds, with 173 registered report figures and all eleven returned result objects.
The bounded rendering is explicitly part of the baseline harness.

`scripts/capture_legacy_baseline.py` retrieves the original source from the preservation
commit, applies that one rendering repair and substitutes only explicit input/output paths.
It runs the complete original calculation and exports its returned tables. No raw data is
fetched. A local Git checkout containing that commit is required.

## Engineering decisions

- Extracted source functions into `preparation`, `descriptive`, `props`, `accumulator`,
  `distributions`, `dependence`, and `plots`, with a small explicit pipeline, IO/logging,
  plotting and reporting layer. No duplicate notebook implementation.
- Kept the original arithmetic, rounding, population, qualification rules and estimators.
  Moved the fixed accumulator specification to packaged JSON; CLI overrides are explicit.
- Replaced import-time logging/style/directory changes and Matplotlib class monkey patches
  in the maintained workflow with scoped handlers, contexts and explicit title helpers.
  The experimental plotting module retains its legacy title-suffix API for compatibility.
- Saved/closed previously interactive figures; all generated PNGs enter the report,
  including previously omitted distribution-fit plots. Output paths are explicit and
  occupied run directories are rejected.
- Reused exact player/date groups instead of repeatedly scanning full data. Cached MI
  only for identical ordered arrays and dtypes, retaining estimator direction, seed,
  complete-case population and the full legacy result matrix. Alias columns remain in
  the specification to preserve results; eliminating them analytically is a separate choice.
- Normal mode writes full diagnostics once to a single stream and prints concise progress.
  `--mode test` tees the same grid tables and banners and adds a diagnostic-log PDF.
  Exception-safe context cleanup restores stdout, stderr and root handlers.
- All returned tables have portable CSV exports; output hashes and source coverage are
  recorded. Test fixtures contain only three small aggregate statistical tables and
  hashes/schema metadata, not raw player-game rows or bulky figures.
- Removed silent synthetic fallback in experimental market/ticket loaders. Explicit
  `allow_synthetic=True` remains a demo-only Python API. Known outcome-derived model
  features now raise rather than silently fitting an invalid predictive model.
- Consolidated historical documentation in `docs/legacy`; it is archival, not current
  execution guidance. Legacy interactive accumulator variants are recoverable via Git,
  instead of maintained as multiple competing implementations.

No copula estimator, distribution-selection method, outcome settlement or predictive model
was silently corrected. Safety gates and honest chart/report labels are intentional
behaviour changes and are documented in the analytical review.

## Publication boundary

No push, upload, publication, visibility change or remote modification was performed.
The maintained subproject is prepared for local reproducibility review. The parent checkout
still contains unrelated projects/IDE state, and historical commits contain generated
artifacts. A future repository-wide publication review must address those separately,
including source-data rights and licensing; this task did not authorize rewriting history.

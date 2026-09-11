# Validation record

The preserved **descriptive** real-data workflow is runnable. The separate economic/
predictive experiments remain unvalidated for the reasons in ANALYTICAL_REVIEW.md.

## Evidence

- Input: 24,092 rows, 698 player IDs, 898 game IDs, 2025-10-02 to 2026-02-15;
  SHA-256 `b38081c069f669e8e3d825ad92e211d7708482536d21f2405374f0d8309d7bc5`.
- Preserved filter: 18,346 player-games retained; duplicate player/game keys: zero.
- The untouched baseline hit an excessive discrete plotting allocation. The documented
  bounded-grid repair allowed the original complete workflow to finish in 208.816 seconds.
- Refactored normal and verbose test modes both completed on the full real dataset.
  They export 45 tables and 317 figures; all 45 table hashes agree between modes.
- All 42 legacy-exported tables pass numerical comparison (`rtol=1e-10`, `atol=1e-12`).
  See `preservation-comparison.json` for the complete table list.
- Thirty-nine legacy table exports are byte-identical. Distribution, entropy and copula
  summaries show tiny numerical differences between Conda and pip binary environments;
  all pass the tolerances. The largest observed absolute difference in distribution
  standard deviations is 2.33e-9, on a value of millions. Three small reference statistical
  CSVs allow tolerance checks without committing raw data.
- Tests validate input contracts, duplicate keys, invalid booleans, explicit missing-file
  failure without synthetic substitution, known predictor leakage rejection, logging
  restoration after exceptions, source identity, all baseline tables, output integrity,
  independent raw-points reconciliation, and report page coverage.
- Editable installation and the CLI are checked in a clean local `.venv` with pinned
  package versions. Core static checks pass with Ruff. The user’s original environment
  had a broken `_distutils_hack` startup hook; the new isolated environment avoids it.

## Rendering and interpretation

The main PDF contains 318 pages (cover plus all 317 figures); the verbose diagnostic PDF
contains 110 pages. All pages were rendered for contact-sheet review, with selected
cover, statistical figure, table and log pages inspected at larger scale. This is a visual
layout review, not an assertion that every inherited statistical interpretation is valid.
Dense tables/annotations are best examined in the standalone PNGs and CSVs.

The PDF embeds bounded high-quality viewing copies of figures; full-resolution lossless PNGs
are retained. This avoids repeatedly embedding oversized raw raster data. Its hashes and
source hashes are recorded in each completed run manifest. PDFs and PNGs are not promised
to be byte-identical across platform/font versions.

## Timing interpretation

In the same local environment, caching identical ordered MI inputs reduced the dependence
stage from 47.817 to approximately 30.6 seconds. An early complete refactor took 216.842
seconds while exporting more figures than the 208.816-second original. After MI caching,
normal/test runs took approximately 195 seconds each, before the final PDF embedding
optimisation. These are individual local timings, not controlled benchmark claims.
The delivered runs completed in 147.839 seconds (normal) and 146.371 seconds (test),
including about 40 seconds for PDF assembly. The report is approximately 69.2 MiB.
The delivered `output/research/manifest.json` and `docs/run-summary.json` contain final stage timings.
The final local suite passed all 15 tests with no skips; both modes' complete output
hashes and executed source hashes were independently checked.

## Remaining validation boundaries

- No real bookmaker odds/ticket dataset has been established. The economic pipeline cannot
  be claimed to have run end-to-end on real market data. Descriptive CSV execution does
  not require or substitute such data.
- No new causal, predictive or economic finding is claimed. Methodological extensions are
  proposals; preserved distribution/copula caveats remain material.
- Automated real-data checks skip explicitly on a checkout without the private local CSV
  or completed outputs. No CI run or cloud execution has been performed.
- The optional Excel importer and optional GPU/learned-line experiments are not part of
  the validated full CSV workflow; their runtime dependencies are separate.
- The preservation baseline and full local raw/output inventory remain in ignored
  `.local_preservation/`. Historical data/output objects in old Git commits remain;
  no history rewrite was authorized.

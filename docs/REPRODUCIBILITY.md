# Reproducibility and evidence boundaries

## What can be reproduced here

The repository builds the IEEE-style manuscript, exports a single-column Word companion with native equations, reproduces the included analytical plots, and redraws reported-data panels from bundled CSV inputs. The accounting utility evaluates the new equations and rejects invalid inputs. The Python unit tests check token sums, zero-improvement handling, signed quality change, byte/cost units, and the 70B-class KV-capacity calculation.

This is **artifact reproduction**, not an independent replication of vLLM, Sarathi-Serve, Medusa, KIVI, DistServe, MegaScale-Infer, or OpenRLHF on accelerators. There is no original experiment dataset, trained checkpoint, or executable autonomous-improvement system in this package. Reproducing those studies requires their original artifacts, hardware, software revisions, and evaluation protocols.

## Build

Use Python 3.11 or newer with `requirements.txt` (the provided artifact was built with Python 3.13). Install a TeX distribution containing IEEEtran, amsmath, BibLaTeX, Biber, and the packages declared in `paper/main.tex`; `latexmk` orchestrates the build. Pandoc is needed only for the editorial exports.

```bash
make paper       # uses the preserved figure assets
make editorial   # native-equation DOCX plus Markdown
make test        # Python standard-library unit tests
```

Without `make`, run the commands listed in the Makefile directly. A PDF-only build needs no Python dependencies. No font files are redistributed.

`make figures` regenerates figure assets from the three supplied generators in the stated order. The final generator applies the integrated layout. Legacy figure filenames do not equal manuscript numbering; use `EVIDENCE.md` to resolve them. Rendering details can vary with Matplotlib, fonts, and TeX versions even when numerical data agree.

## Analytical assumptions

The canonical dense-model example uses 70 billion parameters, 2-byte weights, 80 layers, 8 KV heads, 128 elements per head, and 2-byte KV values. It yields 327,680 KV bytes per token, 2.5 GiB per 8,192-token sequence, and 160 GiB for 64 independent sequences. The analytical intensity plots omit attention arithmetic and use a selected balance point of 299 FLOP/byte. The infinite-batch limit is a mathematical proxy, not a feasible memory allocation or a universal model bound.

## Proposed improvement accounting

`RoundDemand` specifies candidate count, evaluations per candidate, candidate output length, evaluator output length, generator prompt length, and evaluator context excluding the candidate. `token_demand` returns logical prompt and decode counts separately. Use per-model records for different tokenizers. Judge prefill is not free simply because its output is short.

The example has three rounds, 32 candidates per round, two evaluations per candidate, 512 candidate output tokens, 64 evaluator output tokens, 256 generator prompt tokens, and 128 evaluator context tokens. The outputs are 61,440 decode tokens and 147,456 logical prompt tokens. These are **chosen inputs**, not measured experiments.

`bytes_per_verified_improvement` requires a distinct-improvement count supplied by an external protocol; it cannot establish correctness or novelty. Zero improvements returns `None`, not zero resource cost. `weighted_movement_cost` requires matching boundaries and same-unit coefficients. `quality_gain_per_joule` accepts signed quality change and five mutually exclusive energy categories; its caller must prevent double counting and document omissions.

## Proposed experimental validation, not completed work

Freeze the baseline, external audit protocol, test distribution, and budget before comparison. Compare serving-only and evaluator-aware scheduling at equal campaign energy or time, reporting generator/evaluator demand, cache traffic, checking cost, failures, candidate acceptance, policy/evaluator versions, independent quality change, and uncertainty across repeated runs. Preserve non-improving runs. No result from this protocol is claimed here.

## Build provenance

The artifact bundle was generated on September 5, 2026. Exact tool versions and file checksums are supplied with the release. PDF and Word layout were rendered for inspection; this is distinct from validating all empirical claims in the cited literature.

# Changelog

## 1.1.0 — 2026-09-05

### Added

Section XIII adds evaluator-aware self-improvement: loop distinctions, generator/evaluator token accounting, an evaluation-mechanism table, boundary-specific bytes per verified improvement, and quality gain per joule. It includes versioned evidence and acceptance records, evaluation backpressure, held-out audit, and a proposed validation protocol.

Four references were added: the reviewed arXiv:2607.07663v1 survey, Self-Refine, STaR, and FunSearch. The survey is used for conceptual framing, not its corpus statistics. The accompanying equations are explicitly identified as new analytical proposals in this manuscript.

The repository adds a README, citation metadata, a static landing page, a canonical source build, Markdown/Word export, evidence documentation, and a tested accounting helper. Tests validate arithmetic and input handling only; they do not validate the quality of an evaluator or an improvement algorithm.

### Updated

The main title uses **Inference as a New Frontier**. The author remains Prasuna Chatla. Abstract, introduction, contributions, measurement methodology, limitations, and conclusion now include the evaluator workload and separate inference efficiency from improvement evidence. Existing figures and reported data were preserved rather than replaced by new illustrative benchmark claims.

### Explicit technical clarifications

- The KV-induced asymptote is a ceiling on the stated **parameter-FLOP/traffic proxy**, not a universal full-transformer compute bound. Attention arithmetic, capacity limits, and memory assumptions are now explicit.
- Shared physical KV storage avoids duplication and repeated prefill; it does not automatically amortize all decode reads across distinct query tokens.
- Bandwidth-bound speculative verification does not guarantee a net speedup. Acceptance, draft cost, state reads, expert activation, and scheduling still determine profitability.
- Energy-efficiency optimizations use goodput divided by **mean power**, rather than goodput divided by total energy. A work count divided by total energy remains valid over a common horizon.
- Per-iteration latency components are not added as if component p99 values produced end-to-end p99.
- MoE dispatch/combine payload is distinguished from bytes crossing a particular network boundary; locally routed tokens need not traverse that boundary.
- RL behavior-policy mismatch is distinguished from legitimate reuse of samples from an older policy. A different random seed is not by itself a distribution change.
- The proposed BPVI is an unweighted byte count per verified improvement. A weighted sum of bytes is a separate cost metric with units determined by its coefficients.

### Evidence limits

No new GPU measurements, externally validated improvement campaign, full systematic survey replication, publication acceptance, or public repository upload is claimed. This revision preserves earlier reported numerical panels without rerunning the source studies' hardware experiments.

# Inference as a New Frontier

### A Data-Movement-Centric Analysis of Large-Model Serving and Reinforcement Learning Rollouts

**Prasuna Chatla** · Independent Researcher · September 2026  
**Technical preprint and reproducibility companion · version 1.1.0**

[Read the IEEE-style PDF](paper/inference_as_a_new_frontier.pdf) · [Editable Word manuscript](paper/inference_as_a_new_frontier.docx) · [Read in Markdown](paper/paper.md) · [Cite this work](CITATION.bib)

## Overview

This paper analyzes inference through the placement, lifetime, and movement of model state: weights, activations, key–value caches, expert parameters, and rollout trajectories. It connects KV paging and prefix reuse, continuous batching, chunked prefill, speculative decoding, quantization, prefill/decode disaggregation, TTFT–throughput trade-offs, MoE serving, RL rollouts, and inference-specialized hardware.

The central diagnostic is **bytes moved per useful token**. The September revision extends the discussion to **evaluator-aware self-improvement**: generation, critique, judging, repair, tools, and training consume separate resources, and faster generation alone is not evidence of greater improvement.

The paper contains three evidence classes: calculations derived from explicit assumptions; source-grounded explanatory diagrams; and replotted results reported by primary studies. **It reports no new GPU benchmark and no completed self-improvement experiment.** The new verified-improvement metrics are proposed accounting tools, not empirically validated performance claims. No peer-review acceptance is claimed.

## September extension

Section XIII, “Inference as the Substrate of Self-Improvement,” distinguishes bounded self-refinement from open-ended recursive improvement, then develops separate accounting for generator decode and evaluator prompt/decode demand. It introduces boundary-specific bytes per verified improvement and independently assessed quality gain per joule. The latter includes generation, evaluation, tools, training, and synchronization energy.

The literature connection is to *Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops*, arXiv:2607.07663v1, and to the primary Self-Refine, STaR, and FunSearch studies. Full citations appear in the manuscript and [evidence ledger](docs/EVIDENCE.md). The new equations and systems implications are the present manuscript's analytical extension, not results copied from that survey.

The 11 existing figures are retained. Published measurements keep their source-specific workloads, baselines, and units; analytical figures are not represented as measurements. See [CHANGELOG.md](CHANGELOG.md) for the accompanying scope and dimensional clarifications.

## Repository layout

```text
paper/           PDF, DOCX, Markdown, canonical main.tex and references.bib
  figures/       Vector figure assets for the IEEE PDF
  figures_png/   Raster counterparts for Word and Markdown
  data/          CSV inputs for the published-data and analytical plots
scripts/         Editorial export and proposed accounting utilities
  ieee.csl       Attributed third-party bibliography style
tests/           Accounting and unit-consistency tests
docs/           Evidence, reproduction, and hosting notes
CITATION.cff     GitHub citation metadata, preferring the manuscript citation
CITATION.bib     BibTeX citation for the manuscript
index.html       Optional static GitHub Pages landing page
```

## Reproduce

The supplied PDF is ready to read; no tools are required to view it. To rebuild, use a TeX distribution with IEEEtran, BibLaTeX, Biber, and latexmk. The editorial export additionally requires Pandoc and Python packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
make test
make paper
make editorial
```

To regenerate the figure assets separately:

```bash
make figures
make paper
make editorial
```

The plotting scripts reproduce included analytical series and reported-data panels, **not the original hardware experiments**. The accounting example is deliberately synthetic:

```bash
python scripts/improvement_accounting.py --example
```

It outputs 61,440 logical decode tokens and 147,456 logical prompt tokens under the stated example inputs. It does not estimate speedup, joules, or learning quality. Details are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); executed checks and tool versions are recorded in [BUILD_REPORT.md](docs/BUILD_REPORT.md).

## Cite

```bibtex
@unpublished{chatla2026inference,
  author = {Prasuna Chatla},
  title = {Inference as a New Frontier: A Data-Movement-Centric Analysis of Large-Model Serving and Reinforcement Learning Rollouts},
  year = {2026},
  month = {9},
  note = {Technical preprint, version 1.1.0}
}
```

No DOI or publication venue has been assigned in this package. Add a persistent identifier and repository URL only after they exist.

## Hosting and reuse

[GitHub publishing instructions](docs/GITHUB_PUBLISHING.md) cover uploading the repository and enabling the optional static landing page. This archive is prepared for hosting; it does not create or publish a GitHub repository.

A reuse license for the manuscript and original code has not been selected on the author's behalf. Consult [RIGHTS.md](RIGHTS.md) and retain the third-party attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

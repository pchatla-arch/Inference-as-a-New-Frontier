# Evidence ledger

## Classification

**Derived** means calculated from explicit inputs, not observed on hardware. **Reported** means a value carried over from a cited primary study, not newly measured. **Redraw** means an explanatory diagram of a cited mechanism or an author synthesis; it is not a measured production topology. These classes must remain visible when figures are reused.

## Figure map

Legacy asset filenames are preserved to avoid breaking the curated figure generators. The figure numbers below refer to this manuscript, not to filename prefixes.

| Figure | Asset stem | Evidence and numerical input | Primary source |
|---|---|---|---|
| 1 | `fig1_energy_hierarchy` | Reported historical 45-nm estimates; `fig1_horowitz_energy_45nm.csv` | Horowitz, ISSCC 2014, DOI 10.1109/ISSCC.2014.6757323. Not present-day chip specifications. |
| 2 | `fig2_decode_regimes` | Derived traffic and parameter-FLOP proxy; `fig2_derived_traffic_llama2_70b.csv`, `fig2_decode_regimes_ai.csv`, `fig2_decode_regimes_context.csv` | Manuscript assumptions and equations; not measured. |
| 3 | `fig3_kv_scaling` | Derived capacity; `fig3_derived_kv_capacity.csv` | Manuscript KV-capacity equation; excludes format metadata and allocator overhead. |
| 4 | `fig4_paged_kv` | Redraw plus reported sharing savings; `fig4_vllm_memory_saving_fig15.csv` | PagedAttention/vLLM, arXiv:2309.06180, Fig. 15, OPT-13B/Alpaca. |
| 5 | `fig5_continuous_batching` | Redraw plus reported capacity maxima; `fig5_sarathi_capacity_gains.csv` | Sarathi-Serve, OSDI 2024; model/GPU/SLO settings differ between bars. |
| 6 | `fig6_speculative` | Redraw plus reported Vicuna speedups; `fig6_medusa_speedups.csv` | Medusa, arXiv:2401.10774. Acceptance procedures need not share exact-sampling guarantees. |
| 7 | `fig8_quant_disagg_evidence` | Reported factors on distinct axes; `fig7_kivi_reported_metrics.csv`, `fig7_distserve_summarization.csv` | KIVI, arXiv:2402.02750; DistServe, OSDI 2024. Not a cross-paper ranking. |
| 8 | `fig8_pd_disaggregation` | Explanatory phase-pool redraw, no measured topology | DistServe and Splitwise; references in the surrounding text. |
| 9 | `fig10_moe_serving` | Redraw plus reported decode and M2N metrics; `fig10_megascale_infer_metrics.csv` | MegaScale-Infer, arXiv:2504.02263. Decode gain, communication throughput, and latency remain different metrics. |
| 10 | `fig9_rl_loop` | Redraw plus reported 14B step times; `fig9_openrlhf_14b_step_times.csv` | OpenRLHF, arXiv:2405.11143 / EMNLP 2025 system demonstration. Eight H200 GPUs; source omits first ten steps. |
| 11 | `fig12_reference_architecture` | Author synthesis; no numerical experiment | State-placement framework developed in this manuscript. |

The original figure data and artwork were retained from the integrated manuscript. This revision did not rerun the cited hardware experiments or reconstruct new measurements by digitizing images. Legacy intermediate assets may be emitted by the generators; only the 11 assets referenced in `main.tex` are manuscript figures.

## Self-improvement extension: what comes from where

**Reviewed source:** Mingguang Chen, Licheng Wang, and Bo Qu, *Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops*, arXiv:2607.07663v1, July 8, 2026, DOI 10.48550/arXiv.2607.07663. Version-pinned record: https://arxiv.org/abs/2607.07663v1 . Used for the distinction between bounded refinement and open-ended RSI, separate consideration of evaluator changes, and a qualitative verification hierarchy. The hierarchy graphic was inspected, but is not copied into this repository. Corpus counts are not used to support numerical inference-system claims; the full corpus has not been independently audited here.

**Primary anchors:** Self-Refine (arXiv:2303.17651), STaR (arXiv:2203.14465), and FunSearch (Nature 625, 468–475; DOI 10.1038/s41586-023-06924-6). They provide concrete examples of output refinement, rationale filtering/training, and execution-scored program search. No benchmark figures or numerical gains from those studies are reproduced in the new section.

**Proposed here:** separate generator/evaluator demand equations; boundary-specific BPVI; weighted movement cost per improvement; quality gain per joule; the systems interpretation table; and evaluator-aware scheduling/recordkeeping implications. These are analytical proposals. Neither the reviewed survey nor the present artifact demonstrates their empirical superiority.

**Not claimed:** autonomous recursive improvement, guaranteed convergence, a universally reliable verifier, a universally optimal evaluator ordering, or a new accelerator speedup.

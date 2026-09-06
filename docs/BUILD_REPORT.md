# Release checks — version 1.1.0

Artifact date: September 5, 2026. These checks concern the distributed manuscript and software companion, not the empirical validity of every referenced study.

## Documents

The canonical IEEE-style PDF compiled successfully using `latexmk` and Biber. Its final log contains no unresolved citations, undefined references, or overfull-box warnings. The 18-page PDF and the 28-page editorial Word rendering were inspected page by page. The manuscript contains 35 numbered equations, 11 figures, 7 tables, and 50 references. The Word file contains native OfficeMath expressions rather than formula screenshots.

The standalone PDF and Word downloads are byte-identical copies of the files under `paper/`. Figure filenames are stable legacy asset names; manuscript numbering is documented in `EVIDENCE.md`.

## Executed checks

- Ten Python unit tests passed using `python -m unittest discover -s tests -v`.
- The synthetic accounting example returned 61,440 logical decode tokens and 147,456 logical prompt tokens.
- All three figure generators ran in the documented order in a separate working copy. All 13 resulting CSV inputs matched the preserved release CSVs byte for byte. Rendered pixels need not be identical across plotting environments.
- README and static landing-page local links resolved to included files.
- Citation metadata parsed as YAML and its author, title, and type fields matched the manuscript/BibTeX metadata. The repository has not been uploaded to test GitHub's live citation interface.
- Document scans found no unresolved-reference markers or editorial equation-export sentinels.

No GPU serving benchmark, learned evaluator, autonomous-improvement campaign, or GitHub deployment was executed. Passing the accounting tests does not establish evaluator soundness, learning convergence, or the performance gains of any cited system.

## Build environment

| Component | Version used |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |
| python-docx | 1.2.0 |
| Pandoc | 3.1.11.1 |
| latexmk | 4.86 |
| Biber | 2.20 |
| LibreOffice, Word-rendering check only | 25.2.3.2 |

The canonical PDF is built from LaTeX, not from the Word rendering. No font files, cached source-paper downloads, temporary render images, or local build logs are included. SHA-256 hashes of release files appear in `SHA256SUMS.txt` at the repository root.

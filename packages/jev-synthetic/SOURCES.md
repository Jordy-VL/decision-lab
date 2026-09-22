# Source and reference verification

Verified 2026-09-22 using Hugging Face dataset metadata/cards, file listings and small streaming checks. Dataset cards report licensing; they do not independently resolve every underlying content right. Retain attribution and source records. Check the sample report for actual retained counts, not just availability.

| Registry source | Input and grouping | Revision / split | License and limits |
|---|---|---|---|
| [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) | `section_7` string (management discussion), `filename` group | Conversion `f7d3ba73d65ff10194a95b84c75eb484d60b0ede`, `year_1993/train/0000.parquet`; official train/validation/test retained | Card Apache-2.0, SEC filing provenance. Metadata main revision `7e90f0f342569b35213445f809cfaf3b91f9964f`. Card warns splits have no special semantic meaning. Remote Python loader avoided; pinned parquet is read directly. Full filing is not supplied: original selected section is preserved. |
| [LexGLUE LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) | `text` string; pinned split-row ID | `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c`, official train/validation/test | Card CC-BY-4.0. Original label ignored. Parent contract ID absent from this release: row grouping cannot certify contract-level separation. |
| [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) | `dialogue` string, `id` group; summary/topic unused | `a968e7aee0602e257935f1321a02e4287f7d5848`, official train/validation/test; holdout unused | CC-BY-NC-SA-4.0; research-only noncommercial mixture by default. Human-written conversations rather than unrestricted web text. |
| [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) | `text` string; reconstructed article-header groups | `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`; official train/validation/test | Card CC-BY-SA-3.0/GFDL. Wikipedia good/featured article provenance. Article-derived text is not task annotations. |

The checked repositories are public and ungated. No authentication was required for the sample check. A bounded prefix can be unrepresentative; finance checks found the configured 1993 `section_1A` empty, so the registry uses `section_7`. Large content is rejected rather than truncated. Additional datasets should supply the same explicit fields, revision and licensing metadata; there is no indiscriminate discovery/download mode.

## Kev inspection

[Kev](https://github.com/jaredpalmer/kev) inspected at commit `90990a5fac2995b9faa3190f7d437e84f2067768`: `kev/data.py`, `kev/api.py`, `kev/suite.py`, `kev/benchmark.py`, and `skills/kev-finetune/scripts/generate_data.py` plus `split_data.py`. The package adopts typed decision options, grouped provenance and explicit label conversion. It does not import frozen suites, sample evaluation cases, or reuse teacher scores as targets.

`kev.api.question_keys` orders noul as false then true. Choice labels are criteria keys; score labels are integer level indices. `kev.data` materializes those labels for training; `benchmark.labels` uses the same mapping. Suite metadata tracks IDs and groups. Our canonical record is richer and the Kev export is intentionally a labeled-request subset with `_meta`; it does not claim compatibility with every future upstream revision. Use the supplied upstream verification script against the pinned checkout.

## Research context

[AsymptoticAURC](https://github.com/han678/AsymptoticAURC) was consulted for the selective-prediction research context, not copied or executed. The parent roadmap and experiment plan were read without modification. This project contains no training or experimental results.

# AI use log — MOD550 Assignment 2

## Session 1 — 16–17 October 2026: planning and Task 1

- Tool: OpenAI ChatGPT.
- Prompts: Plan and document a reproducible synthetic maize dataset meeting the Task 1 requirements.
- Kept: 1,200 plants, 25% unhealthy, four numerical features and four treatment categories.
- Kept: Western-edge infestation, fixed random seed, plots and class/category summary tables.
- Changed: Replaced proposed leaf temperature with ear length.
- Changed: Reduced the ear-length standard deviation from 2.5 cm to 2 cm.
- Changed: Added PNG summary tables alongside CSV tables.
- Kept: Relative field positions because no actual field coordinates or boundaries were available.
- Checked: Required class/category counts, reproducibility, CSV round-trip and visual outputs.

## Task 1 follow-up — simplify validation

- Tool: OpenAI ChatGPT.
- Prompt: Simplify the delivered script by removing automated validation checks and document that decision.
- Kept: Data generation, summary tables and plots.
- Changed: Removed the validation function and its call; updated the completion message and README.
- Checked: The simplified script was reported to run and reproduce the same dataset.

## Task 1 folder-structure update — 19 October 2026

- Tool: OpenAI ChatGPT.
- Prompt: Change the project layout so scripts are in `src/`, CSV files are in `data/`, and figures are in `figures/`.
- Changed: Introduced project-root paths and separate `DATA_DIR` and `FIGURES_DIR` output folders.
- Intended output: Main dataset and summary CSV files in `data/`; PNG figures in `figures/`.
- Note: The updated script was created, but a final execution check could not be completed in that session. Run it locally and verify the generated files.

## Task 2a — Beta–Binomial posterior update

- User report: The submitted code was completed without AI assistance (it was completely inspired by lecture's notes).


## Task 2b documentation — 19 October 2026

- Tool: OpenAI ChatGPT.
- Prompt: Draft README documentation Task 2b using the Titanic dataset; update the AI-use log.
- Kept: The reported Task 2a results and the distinction between small- and large-sample prior influence in Task 2b.
- Created: Draft README content describing objectives, assumptions, formulas, results and reproducibility.
- To do: Check that the Task 2b group filters, counts, plotted curves and credible intervals match the final code and generated figures. Update filenames if the implementation uses different names.

## Ongoing record

Add an entry for each subsequent AI session. Record the tool, the essence of the prompt, which suggestions were retained, what was changed, and what was checked. The log should describe actual use rather than intended use.

## Task 3 — 21 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Follow classroom style; implement Naive Bayes and subtasks 3a–3c.
- Prompt: Separate the worked calculation from the main README.
- Kept: Classifier, worked calculation, smoothing experiment and manual subset comparison.
- Changed: Used sample standard deviations to match the Gaussian class notes.
- Changed: Added CSV results, PNG plots/tables and clearly separated documentation.
- Checked: Direct arithmetic, training/test separation, fold summaries and figures.
- Note: Execution used separate Python dependencies because installed SciPy was blocked.

## Task 3a separation — 22 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Separate Task 3a into task_3a_claudio without changing results.
- Kept: Classifier, split, selected plant and calculation factors.
- Changed: Moved the calculation into its own script and grouped its outputs.
- Changed: Updated README paths; restored the missing calculation document from the earlier package.
- Checked: Main 3b/3c workflow and separate 3a workflow reproduce previous results.

## Task 3a purpose clarified — 22 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Explain that the hand calculation will be handwritten on a separate sheet.
- Kept: Selected plant, fitted factors and numerical function outputs.
- Changed: README and docstrings state that the script only supports comparison with handwritten work.
- Clarified: direct_posterior is code-generated; no handwritten result has been supplied for verification.
- Changed: Removed current references to the deleted worked-calculation document; it is not recreated.
- Note: Earlier log entries describe previous work; the handwritten submission is now separate.
- Checked: Script runs; all CSV values and other experiment outputs are unchanged; only the comparison figure title changed.

## Task 3a script rename — 22 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Rename the Task 3a support script.
- Changed: Script renamed to task3a_handcalculation_support_compare.py; references updated.
- Kept: Calculation logic and outputs.

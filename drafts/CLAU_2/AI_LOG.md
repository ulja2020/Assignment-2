AI_LOG

#### Task 1

Chatgpt assisted in code generation of Maize dataset. ChatGPT was also used to generate text for task 1 in README (however, we carefully reviewed the generated text and fixed some wording manually to adhere to the context).

#### Task 2

The code for task2a was created completely without AI usage (it is completely inspired by the syntax in the code given at the lecture from presentation "Beta-Binomial.pdf").

The code from presentation "Beta-Binomial.pdf" was also partly adapted within the code for task2b, since we used "Titanic dataset" to create those 2 real-life problems.

Regarding README for both tasks 2a and 2b, ChatGPT was only used to create tables and formulas, while all explanations were written manually.

#### Task 3

The code for all tasks3a, 3b and 3c was created with help of ChatGPT, by asking it to adapt the code from task 1. 

The part of the task3 code is responsible for task3a, but this one has been created just as support for hand calculation, so that we are able to compare handwritten results with the actual results from the code. Our individual hand calculations are demonstrated in individual PDF files. All hand calculations were only based on the notes from "Gaussian Naive Bayes.pdf" presentation, without any AI usage.

README for all tasks 3a, 3b and 3c were partly written manually and partly generated with ChatGPT.

#### Task 4

The code for task 4 were generated with ChatGPT, but README part4b were mostly handwritten.


### Task 4 table module — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Create a separate table script, call it from Task 4, and update title/footnotes.
- Kept: Existing experiments and CSV results.
- Changed: Added table_of_results.py, styling, footnote 3 and across-seed sensitivity summaries.
- Changed: Added the table-generation call and updated README instructions.
- Checked: Standalone table and complete Task 4 script run successfully; the three CSVs are unchanged; PNG visually checked.

### README image references — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Embed all 16 images beside the relevant explanations, with numbered captions.
- Kept: Existing images, filenames, results and original explanations.
- Changed: Added 10 figure references and 6 table-image references using relative paths.
- Checked: All 16 image paths exist; numbering is sequential and captions match the images.

### README numerical corrections — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Correct the Task 2b interval sentence and Task 4 summary-table value.
- Changed: Small-sample group to second-class girls under 16; interval to 46.2%–73.0%.
- Changed: Decision Tree Split C mean error from 2.33% to 2.17%.
- Kept: Code, CSV data and figures.

### Task 3 reference tables — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Show every column of the Task 3a factors and Task 3b plant CSVs in simple tables.
- Kept: Existing calculations and full-precision CSVs.
- Changed: Reused save_table for two PNGs; added article-style explanations and captions.
- Changed: Renumbered table images; identified the factors used to check handwritten steps.
- Checked: Script runs; both tables show all nine columns; numerical result CSVs and earlier figures are unchanged.
- Note: The environment CSV reflects the verification runtime.

### Task 4b wording review — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Improve the Task 4b interpretation after checking the CSV results.
- Kept: Existing results and qualifications about small model differences.
- Changed: Clarified Split A, added error-rate variability and explained metric-dependent preference and f2 assumptions.
- Checked: Added standard deviations match the five-seed CSV results; code, data and figures were not edited.

### Code organization documentation — 30 October 2026

- Tool: OpenAI Codex (ChatGPT).
- Prompt: Document all six scripts, their relationships and execution commands.
- Changed: Added the code organization section and short script references in each task.
- Kept: Results, interpretations, captions, image numbering and Python code.
- Checked: Script filenames, imports and commands against the current source files.

"""Render the combined Task 4 results table without fitting any models.

Run directly with ``python src/table_of_results.py`` after the three Task 4
CSVs exist, or import save_results_table from task4_compare_models.py.
The sensitivity summary is repeated for each seed of its split/model group.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent


def save_results_table(project_dir=PROJECT_DIR):
    """Join the three saved CSVs and return the path to the styled PNG.

    Error and sensitivity values are displayed as percentages. Missing metrics
    stay N/A. Summary variability is the sample standard deviation across seeds,
    expressed in percentage points; it is not variation within one test set.
    """
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    results = pd.read_csv(data_dir / "task4_results.csv")
    balance = pd.read_csv(data_dir / "task4_class_balance.csv")
    summary = pd.read_csv(data_dir / "task4_sensitivity_summary.csv")
    combined = results.merge(balance, on=["split", "seed", "n_train", "n_test"],
                             how="left", validate="many_to_one")
    combined = combined.merge(summary[["split", "model", "mean_sensitivity", "std_sensitivity"]],
                              on=["split", "model"], how="left", validate="many_to_one")
    combined["model_order"] = combined["model"].map({"Naive Bayes": 0, "Decision Tree": 1})
    combined = combined.sort_values(["split", "seed", "model_order"], na_position="first")
    percent = lambda value: "N/A" if pd.isna(value) else f"{100 * value:.2f}"
    labels = {"A_biased": "A: Biased", "B_random": "B: Random", "C_stratified": "C: Stratified"}
    rows = []
    for row in combined.itertuples():
        mean_sd = "N/A" if pd.isna(row.mean_sensitivity) else (
            f"{100 * row.mean_sensitivity:.2f} +/- {100 * row.std_sensitivity:.2f}")
        note = "[1]" if row.split == "A_biased" and row.model == "Naive Bayes" else (
            "[2]" if row.split == "A_biased" else row.status_or_note)
        rows.append([labels[row.split], "-" if pd.isna(row.seed) else str(int(row.seed)),
                     row.model, row.n_train, f"{row.train_y1_count} ({row.train_y1_percent:.2f}%)",
                     row.n_test, f"{row.test_y1_count} ({row.test_y1_percent:.2f}%)",
                     percent(row.test_error_rate), percent(row.sensitivity_y1), mean_sd, note])
    headers = ["Split", "Seed", "Model", "Train\nn", "Train y = 1\nn (%)", "Test\nn",
               "Test y = 1\nn (%)", "Test error\n(%)", "Sensitivity\n(%)",
               "Sensitivity across seeds\nMean (%) +/- SD (pp)", "Status /\nnote"]
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.axis("off")
    fig.suptitle("Table of Results: Combined results by split, seed and model",
                 fontsize=20, weight="bold", y=0.965)
    fig.text(0.5, 0.928, "Class balance and test performance | y = 1: unhealthy maize plants",
             ha="center", fontsize=12, color="#44534c")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                     colWidths=[.105, .04, .095, .05, .105, .05, .105, .075, .075, .20, .10],
                     bbox=[0, .09, 1, .87])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5cf")
        cell.set_linewidth(0.55)
        if i == 0:
            cell.set_facecolor("#285943")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.4)
        else:
            cell.set_facecolor("#fcece5" if rows[i-1][0].startswith("A") else
                               "#eef4f0" if rows[i-1][0].startswith("C") else "white")
    footnotes = [
        "[1] Naive Bayes: undefined predictions (NaN/inf posterior probabilities); a class is absent from training.",
        "[2] Decision Tree: test error is 100%. Sensitivity is N/A for both models in Split A: no unhealthy test plants.",
        "[3] This table combines task4_results.csv, task4_class_balance.csv and task4_sensitivity_summary.csv.",
        "Across-seed summaries repeat within each split/model group. SD (pp) means standard deviation in percentage points.",
        "Train/test percentages use their respective set sizes. Sensitivity = TP / (TP + FN). N/A values are not zeros.",
    ]
    for i, text in enumerate(footnotes):
        fig.text(.045, .095 - i * .019, text, fontsize=10, color="#44534c")
    fig.subplots_adjust(left=.045, right=.955, top=.91, bottom=.16)
    output = project_dir / "figures" / "task4_results_table.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print("Saved:", save_results_table())

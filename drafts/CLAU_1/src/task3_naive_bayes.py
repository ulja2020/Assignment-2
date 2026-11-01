"""MOD550 Assignment 2, Task 3: mixed Gaussian/categorical Naive Bayes.

Run from any folder with: python path/to/src/task3_naive_bayes.py
This file runs subtasks 3b and 3c and provides shared classifier functions.
Subtask 3a uses task_3a_claudio/task3a_handcalculation_support_compare.py only to provide
a function-output reference for a separately handwritten calculation.
Only training observations estimate priors, Gaussian parameters and category
probabilities. The fixed list of four treatments comes from the Task 1 design.
CSV results go to data/ and PNG figures to figures/. This script does not
overwrite README or the AI log. The hand calculation is submitted separately
on a sheet of paper, handwritten; neither script generates that submission.
"""

from pathlib import Path
from importlib.metadata import version
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

from task1_generate import CATEGORIES, generate_dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
SEED = 42
ALL_FEATURES = ["f1", "f2", "f3", "f4", "cat"]
# These subsets are fixed before examining cross-validation or test results.
FEATURE_SUBSETS = [
    ["f1"], ["f2"], ["f3", "f4"], ["f1", "cat"],
    ["f1", "f2", "cat"], ["f1", "f3", "f4", "cat"], ALL_FEATURES,
]


def fit_naive_bayes(training, features, alpha=1.0):
    """Estimate a Naive Bayes model and return its parameters in a dictionary.

    Parameters
    ----------
    training : pandas.DataFrame
        Training rows, including binary target y. No test rows are used.
    features : list of str
        Numerical feature names and, optionally, cat.
    alpha : float
        Category pseudocount: 0 disables smoothing; 1 is Laplace smoothing.
        It does not change Gaussian parameters or class priors.

    Notes
    -----
    Standard deviations use n_c-1 (ddof=1), matching page 7 of the classroom
    Gaussian Naive Bayes notes. The 1e-12 lower bound only avoids division by
    zero for a constant feature; none of the Task 1 features needs it here.
    Categories are known from the experimental design, rather than inferred
    from held-out data. Priors and counts are recomputed for each training set.
    """
    model = {"features": list(features), "alpha": alpha, "classes": {}}
    numerical = [name for name in features if name != "cat"]
    for class_value in (0, 1):
        subset = training[training.y == class_value]
        counts = subset["cat"].value_counts().reindex(CATEGORIES, fill_value=0)
        model["classes"][class_value] = {
            "count": len(subset), "prior": len(subset) / len(training),
            "means": subset[numerical].mean(),
            "sds": subset[numerical].std(ddof=1).clip(lower=1e-12),
            "category_counts": counts,
            "category_probs": (counts + alpha) / (len(subset) + len(CATEGORIES) * alpha),
        }
    return model


def predict_proba(model, observations):
    """Return an n-by-2 array with normalized probabilities for classes 0 and 1.

    Gaussian log densities and the categorical log probability are added to
    the log prior. With alpha=0, a zero category probability deliberately
    gives a class score of negative infinity, hence posterior probability 0.
    A category absent in both classes with no smoothing gives no defined
    posterior; this is not the one-class-absence case used in subtask 3b.
    """
    scores = np.empty((len(observations), 2))
    for class_value in (0, 1):
        parameters = model["classes"][class_value]
        score = np.full(len(observations), np.log(parameters["prior"]))
        for feature in model["features"]:
            if feature == "cat":
                probabilities = observations["cat"].map(parameters["category_probs"])
                with np.errstate(divide="ignore"):
                    score += np.log(probabilities.to_numpy(dtype=float))
            else:
                score += norm.logpdf(observations[feature],
                                     loc=parameters["means"][feature],
                                     scale=parameters["sds"][feature])
        scores[:, class_value] = score
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def evaluate_model(model, test, name):
    """Return one metrics row and predictions; F1 treats unhealthy as positive."""
    prediction = predict_proba(model, test).argmax(axis=1)
    return {"model": name, "accuracy": accuracy_score(test.y, prediction),
            "precision": precision_score(test.y, prediction, zero_division=0),
            "recall": recall_score(test.y, prediction, zero_division=0),
            "f1": f1_score(test.y, prediction, zero_division=0)}, prediction


def save_table(table, filename, title, precision=4, output_dir=None):
    """Save a CSV and PNG; output_dir optionally puts both in one folder."""
    csv_dir = DATA_DIR if output_dir is None else output_dir
    table.to_csv(csv_dir / f"{filename}.csv", index=False)
    display = table.copy()
    for column in display.select_dtypes(include="floating"):
        display[column] = display[column].map(lambda value: f"{value:.{precision}f}")
    fig, ax = plt.subplots(figsize=(11, max(2.6, 0.42 * (len(table) + 2))))
    ax.axis("off")
    ax.set_title(title, pad=16)
    image = ax.table(cellText=display.values, colLabels=display.columns,
                     cellLoc="center", loc="center")
    image.auto_set_font_size(False)
    image.set_fontsize(9)
    image.auto_set_column_width(list(range(len(display.columns))))
    image.scale(1, 1.65)
    for (row, _), cell in image.get_celld().items():
        cell.set_edgecolor("#d5ded9")
        if row == 0:
            cell.set_facecolor("#e0ece5")
            cell.set_text_props(weight="bold")
    save_figure(fig, f"{filename}.png", output_dir=output_dir)


def save_figure(fig, filename, output_dir=None):
    """Save a PNG to figures/ (or output_dir) and close the figure."""
    fig.tight_layout()
    folder = FIGURES_DIR if output_dir is None else output_dir
    fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


# %% Task 3b: zero categorical count, without and with Laplace smoothing.
def smoothing_experiment(training, test):
    """Remove rare class-1 training rows only in this experiment, then compare.

    Both fits use the same reduced training set and all five features. The
    test point is the first Enhanced protection row, regardless of its label.
    The original training frame and the held-out test frame remain unchanged.
    """
    removed = training[(training.y == 1) & (training["cat"] == "Enhanced protection")]
    reduced = training.drop(index=removed.index)
    removed[["plant_id", "y", "cat"]].to_csv(DATA_DIR / "task3b_removed_training_rows.csv", index=False)
    point = test[test["cat"] == "Enhanced protection"].iloc[[0]]
    point.to_csv(DATA_DIR / "task3b_test_plant.csv", index=False)
    rows = []
    for alpha in (0, 1):
        model = fit_naive_bayes(reduced, ALL_FEATURES, alpha=alpha)
        probabilities = predict_proba(model, point)[0]
        rows.append({"alpha": alpha, "removed_rows": len(removed),
                     "training_n": len(reduced),
                     "category_given_y0": model["classes"][0]["category_probs"]["Enhanced protection"],
                     "category_given_y1": model["classes"][1]["category_probs"]["Enhanced protection"],
                     "posterior_y1": probabilities[1]})
    results = pd.DataFrame(rows)
    save_table(results, "task3b_smoothing_results", "Task 3b: Same plant, without and with smoothing", 6)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = results.posterior_y1.to_numpy()
    ax.bar(["No smoothing (alpha = 0)", "Laplace smoothing (alpha = 1)"],
           values, color=["#777777", "#237a57"])
    ax.set(ylabel="P(unhealthy | observed features)",
           title="Task 3b: Smoothing prevents a zero category from vetoing class 1",
           ylim=(0, max(0.001, values.max() * 1.35)))
    for i, value in enumerate(values):
        ax.text(i, value + ax.get_ylim()[1] * 0.025, f"{value:.6g}", ha="center")
    save_figure(fig, "task3b_smoothing_comparison.png")
    return results


# %% Task 3c (bonus): manually chosen subsets, training-only cross-validation.
def compare_feature_subsets(training):
    """Compare seven predefined subsets on identical five stratified folds.

    A fresh model, including all parameters, is fitted inside every fold.
    Rank by mean validation F1; exact ties prefer fewer features, then the
    original list order. No held-out test result enters this selection.
    Standard deviation is descriptive fold variability, not a confidence interval.
    """
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(training, training.y))
    details, summary = [], []
    for features in FEATURE_SUBSETS:
        scores = []
        name = " + ".join(features)
        for fold, (fit_indices, validation_indices) in enumerate(folds, start=1):
            fit_data, validation = training.iloc[fit_indices], training.iloc[validation_indices]
            model = fit_naive_bayes(fit_data, features, alpha=1)
            prediction = predict_proba(model, validation).argmax(axis=1)
            score = f1_score(validation.y, prediction, zero_division=0)
            scores.append(score)
            details.append({"features": name, "fold": fold, "f1": score})
        summary.append({"features": name, "n_features": len(features),
                        "mean_f1": np.mean(scores), "sd_f1": np.std(scores, ddof=1)})
    results = pd.DataFrame(summary).sort_values(
        ["mean_f1", "n_features"], ascending=[False, True], kind="stable").reset_index(drop=True)
    pd.DataFrame(details).to_csv(DATA_DIR / "task3c_fold_scores.csv", index=False)
    save_table(results, "task3c_feature_subsets", "Task 3c: Training-only five-fold validation F1")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results.features, results.mean_f1, xerr=results.sd_f1, capsize=4, color="#237a57")
    ax.invert_yaxis()
    ax.set(xlabel="Mean validation F1 (error bars: fold standard deviation)",
           xlim=(0, 1.03), title="Task 3c: Manually chosen feature subsets")
    save_figure(fig, "task3c_feature_comparison.png")
    return results.iloc[0]["features"].split(" + ")


def main():
    """Load Task 1 data, fit the full model, then run subtasks 3b and 3c."""
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    environment = [{"package": "Python", "version": platform.python_version()}]
    environment += [{"package": name, "version": version(name)} for name in
                    ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]]
    pd.DataFrame(environment).to_csv(DATA_DIR / "task3_environment.csv", index=False)
    dataset_path = DATA_DIR / "task1_synthetic_data.csv"
    if not dataset_path.exists():
        generate_dataset().to_csv(dataset_path, index=False)
        print("Recreated the missing Task 1 CSV using the unchanged Task 1 generator.")
    data = pd.read_csv(dataset_path)
    training, test = train_test_split(data, test_size=0.25, random_state=SEED, stratify=data.y)
    split = pd.concat([training[["plant_id", "y"]].assign(partition="training"),
                       test[["plant_id", "y"]].assign(partition="test")])
    split.to_csv(DATA_DIR / "task3_split.csv", index=False)

    model = fit_naive_bayes(training, ALL_FEATURES, alpha=1)
    full_metrics, predictions = evaluate_model(model, test, "All five features")
    matrix = confusion_matrix(test.y, predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=["actual_healthy", "actual_unhealthy"],
                 columns=["predicted_healthy", "predicted_unhealthy"]).to_csv(DATA_DIR / "task3_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Healthy", "Unhealthy"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Task 3: Full-model test confusion matrix")
    save_figure(fig, "task3_confusion_matrix.png")

    # 3b: modify a copy for this experiment only.
    smoothing = smoothing_experiment(training, test)
    # 3c: select using training folds, then evaluate on the held-out test set.
    best_features = compare_feature_subsets(training)
    selected = fit_naive_bayes(training, best_features, alpha=1)
    best_metrics, _ = evaluate_model(selected, test, "CV selected: " + " + ".join(best_features))
    metrics = pd.DataFrame([full_metrics, best_metrics])
    save_table(metrics, "task3_test_metrics", "Task 3: Held-out test metrics (positive class: unhealthy)")
    print(f"Training: {len(training)} plants; test: {len(test)} plants.")
    print(metrics.to_string(index=False))
    print("3b:\n", smoothing.to_string(index=False))
    print("3c: selected features:", best_features)


if __name__ == "__main__":
    main()

"""
MOD550 Assignment 2 — Task 4a
Compare the mixed Gaussian/categorical Naive Bayes model from Task 3 with a
single Decision Tree classifier.

Run from the project root:
    python src/task4_compare_models.py

Expected project layout:
    src/task1_generate.py
    src/task3_naive_bayes.py
    src/task4_compare_models.py
    data/task1_synthetic_data.csv

Outputs:
    data/task4_results.csv
    data/task4_class_balance.csv
    data/task4_sensitivity_summary.csv
    figures/task4_sensitivity.png
    figures/task4_results_table.png

Split A intentionally puts every y=1 observation in training and every y=0
observation in testing. Since this leaves no y=0 examples in training and no
y=1 examples in testing, sensitivity is undefined for Split A. The custom
Naive Bayes implementation from Task 3 may return NaN probabilities because
one class has no training observations; the script records that rather than
inventing a metric. A decision tree can fit a one-class training set and will
typically predict that class for every test observation.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from table_of_results import save_results_table
from task1_generate import generate_dataset
from task3_naive_bayes import fit_naive_bayes, predict_proba, ALL_FEATURES


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
TARGET = "y"
NUMERIC_FEATURES = ["f1", "f2", "f3", "f4"]
FEATURES = list(ALL_FEATURES)
SEEDS = [42, 43, 44, 45, 46]


def load_data():
    """Load Task 1 data, generating it if the CSV does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "task1_synthetic_data.csv"
    if not path.exists():
        print(f"Could not find {path}; generating the Task 1 dataset.")
        data = generate_dataset()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data.to_csv(path, index=False)
    data = pd.read_csv(path)

    required = set(FEATURES + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "Check that Task 1 generated the expected dataset."
        )
    return data


def make_decision_tree():
    """Create a single decision tree, using one-hot encoding for cat."""
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", transformer),
            ("tree", DecisionTreeClassifier(random_state=42)),
        ]
    )


def make_splits(data):
    """Return Split A once, and unstratified/stratified splits for five seeds."""
    splits = []

    # Split A: all minority-class (y=1) observations in train, all others in test.
    train_a = data.loc[data[TARGET] == 1].copy()
    test_a = data.loc[data[TARGET] != 1].copy()
    splits.append(("A_biased", None, train_a, test_a))

    # Split B: random 80/20 split without stratification, repeated for 5 seeds.
    # Split C: stratified 80/20 split, repeated for the same 5 seeds.
    for seed in SEEDS:
        train_b, test_b = train_test_split(
            data, test_size=0.20, random_state=seed, shuffle=True
        )
        splits.append(("B_random", seed, train_b.copy(), test_b.copy()))

        train_c, test_c = train_test_split(
            data,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=data[TARGET],
        )
        splits.append(("C_stratified", seed, train_c.copy(), test_c.copy()))

    return splits


def balance_record(split_name, seed, train, test):
    """Summarize class counts and the percentage of y=1 in train and test."""
    def pct_positive(frame):
        return 100.0 * (frame[TARGET] == 1).mean() if len(frame) else np.nan

    return {
        "split": split_name,
        "seed": seed,
        "n_train": len(train),
        "train_y1_count": int((train[TARGET] == 1).sum()),
        "train_y1_percent": pct_positive(train),
        "n_test": len(test),
        "test_y1_count": int((test[TARGET] == 1).sum()),
        "test_y1_percent": pct_positive(test),
    }


def evaluate_predictions(y_true, y_pred):
    """Return test error rate and sensitivity for the positive class y=1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan, np.nan, "Test set is empty."

    error_rate = float(np.mean(y_true != y_pred))
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        sensitivity = np.nan
        note = "Sensitivity undefined: test set contains no y=1 observations."
    else:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) else np.nan
        note = ""
    return error_rate, sensitivity, note


def evaluate_naive_bayes(train, test):
    """Evaluate Task 3's Naive Bayes functions, preserving failure information."""
    status = "ok"
    try:
        model = fit_naive_bayes(train, FEATURES, alpha=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probabilities = predict_proba(model, test[FEATURES])

        warning_text = "; ".join(str(w.message) for w in caught)
        if warning_text:
            status = f"warning: {warning_text}"

        if not np.isfinite(probabilities).all():
            return None, None, (
                "Undefined predictions: NaN/inf posterior probabilities. "
                "At least one class is absent from the training set."
            )

        predictions = probabilities.argmax(axis=1)
        return predictions, probabilities, status
    except Exception as exc:
        return None, None, f"error: {type(exc).__name__}: {exc}"


def run_experiment(data):
    result_rows = []
    balance_rows = []

    for split_name, seed, train, test in make_splits(data):
        balance_rows.append(balance_record(split_name, seed, train, test))
        print("\n" + "=" * 72)
        print(
            f"{split_name} | seed={seed} | "
            f"train n={len(train)} | test n={len(test)}"
        )
        print(
            f"Train y=1: {(train[TARGET] == 1).sum()}/{len(train)} "
            f"({100 * (train[TARGET] == 1).mean():.2f}%)"
        )
        print(
            f"Test y=1: {(test[TARGET] == 1).sum()}/{len(test)} "
            f"({100 * (test[TARGET] == 1).mean():.2f}%)"
        )

        # Model 1: reuse the student's custom Task 3 Naive Bayes implementation.
        nb_pred, nb_probs, nb_status = evaluate_naive_bayes(train, test)
        if nb_pred is None:
            nb_error, nb_sensitivity = np.nan, np.nan
            nb_note = nb_status
        else:
            nb_error, nb_sensitivity, metric_note = evaluate_predictions(
                test[TARGET], nb_pred
            )
            nb_note = "; ".join(x for x in [nb_status, metric_note] if x)

        result_rows.append({
            "split": split_name,
            "seed": seed,
            "model": "Naive Bayes",
            "n_train": len(train),
            "n_test": len(test),
            "test_error_rate": nb_error,
            "test_error_percent": 100 * nb_error if np.isfinite(nb_error) else np.nan,
            "sensitivity_y1": nb_sensitivity,
            "status_or_note": nb_note,
        })
        print(
            f"Naive Bayes: error={nb_error if np.isfinite(nb_error) else 'N/A'}, "
            f"sensitivity={nb_sensitivity if np.isfinite(nb_sensitivity) else 'N/A'}"
        )
        if nb_note:
            print(f"  Note: {nb_note}")

        # Model 2: one decision tree. The same train/test observations are used.
        tree = make_decision_tree()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tree.fit(train[FEATURES], train[TARGET])
                tree_pred = tree.predict(test[FEATURES])
            warning_text = "; ".join(str(w.message) for w in caught)
            tree_error, tree_sensitivity, metric_note = evaluate_predictions(
                test[TARGET], tree_pred
            )
            tree_note = "; ".join(
                x for x in [warning_text, metric_note] if x
            )
            tree_status = "ok" if not warning_text else f"warning: {warning_text}"
        except Exception as exc:
            tree_error, tree_sensitivity = np.nan, np.nan
            tree_note = f"error: {type(exc).__name__}: {exc}"
            tree_status = tree_note

        result_rows.append({
            "split": split_name,
            "seed": seed,
            "model": "Decision Tree",
            "n_train": len(train),
            "n_test": len(test),
            "test_error_rate": tree_error,
            "test_error_percent": 100 * tree_error if np.isfinite(tree_error) else np.nan,
            "sensitivity_y1": tree_sensitivity,
            "status_or_note": tree_note or tree_status,
        })
        print(
            f"Decision Tree: error={tree_error if np.isfinite(tree_error) else 'N/A'}, "
            f"sensitivity={tree_sensitivity if np.isfinite(tree_sensitivity) else 'N/A'}"
        )
        if tree_note:
            print(f"  Note: {tree_note}")

    return pd.DataFrame(result_rows), pd.DataFrame(balance_rows)


def summarize_sensitivity(results):
    """Summarize sensitivity across seeds; Split A stays undefined."""
    rows = []
    for split_name in ["A_biased", "B_random", "C_stratified"]:
        for model_name in ["Naive Bayes", "Decision Tree"]:
            values = results.loc[
                (results["split"] == split_name)
                & (results["model"] == model_name),
                "sensitivity_y1",
            ].dropna()
            if split_name == "A_biased":
                rows.append({
                    "split": split_name,
                    "model": model_name,
                    "n_valid_seeds": int(len(values)),
                    "mean_sensitivity": np.nan,
                    "std_sensitivity": np.nan,
                    "note": "N/A: test set has no y=1 observations.",
                })
            else:
                rows.append({
                    "split": split_name,
                    "model": model_name,
                    "n_valid_seeds": int(len(values)),
                    "mean_sensitivity": float(values.mean()) if len(values) else np.nan,
                    "std_sensitivity": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "note": "",
                })
    return pd.DataFrame(rows)


def plot_sensitivity(summary):
    """Grouped bar chart of sensitivity for all splits and both models.

    Split A is labelled N/A because its test set contains no y=1 cases.
    For splits B and C, bars show the mean across seeds and error bars show
    plus/minus one sample standard deviation.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = ["Naive Bayes", "Decision Tree"]
    splits = ["A_biased", "B_random", "C_stratified"]
    labels = ["A: Biased", "B: Random", "C: Stratified"]
    colors = {"Naive Bayes": "tab:blue", "Decision Tree": "tab:orange"}

    x = np.arange(len(splits))
    bar_width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_idx, model_name in enumerate(models):
        model_rows = summary[summary["model"] == model_name].set_index("split")

        for i, split_name in enumerate(splits):
            row = model_rows.loc[split_name]
            mean = row["mean_sensitivity"]
            std = row["std_sensitivity"]
            xpos = x[i] + (model_idx - 0.5) * bar_width

            if np.isfinite(mean):
                ax.bar(
                    xpos,
                    mean,
                    width=bar_width,
                    color=colors[model_name],
                    yerr=std if np.isfinite(std) else 0,
                    capsize=4,
                    label=model_name if i == 1 else None,
                )
            else:
                ax.text(
                    xpos,
                    0.06,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                    color=colors[model_name],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Sensitivity for y = 1")
    ax.set_title("Task 4a: Sensitivity by split strategy")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Use explicit legend handles so the legend remains correct even though
    # Split A has no numeric bars.
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=colors["Naive Bayes"], label="Naive Bayes"),
        Patch(facecolor=colors["Decision Tree"], label="Decision Tree"),
    ])

    fig.tight_layout()
    path = FIGURES_DIR / "task4_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    data = load_data()
    print(f"Loaded {len(data)} observations.")
    print("Class counts:")
    print(data[TARGET].value_counts().sort_index().to_string())

    results, balances = run_experiment(data)
    summary = summarize_sensitivity(results)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = DATA_DIR / "task4_results.csv"
    balance_path = DATA_DIR / "task4_class_balance.csv"
    summary_path = DATA_DIR / "task4_sensitivity_summary.csv"

    results.to_csv(results_path, index=False)
    balances.to_csv(balance_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_path = plot_sensitivity(summary)
    table_path = save_results_table(PROJECT_DIR)

    print("\n" + "=" * 72)
    print("SAVED OUTPUTS")
    print(f"Model results:       {results_path}")
    print(f"Class balance:       {balance_path}")
    print(f"Sensitivity summary: {summary_path}")
    print(f"Sensitivity plot:    {plot_path}")
    print(f"Combined table:      {table_path}")

    print("\nSensitivity summary (mean +/- SD across seeds for B and C):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nReminder: in Split A, sensitivity is undefined because the test set")
    print("contains no y=1 observations. Do not report this as sensitivity = 0.")
    print("For Split A, Naive Bayes may produce undefined probabilities because")
    print("the training set has no y=0 examples; see status_or_note in the CSV.")


if __name__ == "__main__":
    main()

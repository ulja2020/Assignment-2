"""
MOD550 Assignment 2 — Task 3a, 3b and 3c
Mixed Gaussian/categorical Naive Bayes.

Run from the project root:
    python src/task3_naive_bayes.py

Expected project layout:
    src/task1_generate.py
    src/task3_naive_bayes.py
    data/task1_synthetic_data.csv

This script:
- 3a: prints the factors for a hand-checkable Naive Bayes calculation
       using f1 and cat only.
- 3b: compares no smoothing with Laplace smoothing for a category
       deliberately absent from class y=1 in the modified training set.
- 3c: compares predefined feature subsets with 5-fold CV on training data.
- also reports held-out test metrics for the full and selected models.

Only training observations are used to estimate model parameters. The held-out
test set is not used for feature selection.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from task1_generate import CATEGORIES, generate_dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
SEED = 42

ALL_FEATURES = ["f1", "f2", "f3", "f4", "cat"]

# These candidate subsets are specified before looking at CV/test results.
FEATURE_SUBSETS = [
    ["f1"],
    ["f2"],
    ["f3", "f4"],
    ["f1", "cat"],
    ["f1", "f2", "cat"],
    ["f1", "f3", "f4", "cat"],
    ALL_FEATURES,
]


def fit_naive_bayes(training, features, alpha=1.0):
    """Fit a mixed Gaussian/categorical Naive Bayes model.

    alpha=0 disables categorical smoothing; alpha=1 is Laplace smoothing.
    Gaussian standard deviations use ddof=1 to match the classroom formula.
    """
    numerical = [feature for feature in features if feature != "cat"]
    model = {"features": list(features), "alpha": alpha, "classes": {}}

    for class_value in (0, 1):
        subset = training[training["y"] == class_value]
        category_counts = (
            subset["cat"].value_counts().reindex(CATEGORIES, fill_value=0)
        )
        category_probs = (
            (category_counts + alpha)
            / (len(subset) + len(CATEGORIES) * alpha)
        )

        model["classes"][class_value] = {
            "count": len(subset),
            "prior": len(subset) / len(training),
            "means": subset[numerical].mean(),
            "sds": subset[numerical].std(ddof=1).clip(lower=1e-12),
            "category_counts": category_counts,
            "category_probs": category_probs,
        }

    return model


def predict_proba(model, observations):
    """Return an n-by-2 array of posterior probabilities for classes 0 and 1.

    Log probabilities are used for numerical stability. If alpha=0 and a
    category is absent in one class, that class gets a zero posterior when
    the other class has a nonzero score.
    """
    scores = np.empty((len(observations), 2), dtype=float)

    for class_value in (0, 1):
        parameters = model["classes"][class_value]
        score = np.full(len(observations), np.log(parameters["prior"]))

        for feature in model["features"]:
            if feature == "cat":
                probabilities = observations["cat"].map(
                    parameters["category_probs"]
                ).to_numpy(dtype=float)
                with np.errstate(divide="ignore"):
                    score += np.log(probabilities)
            else:
                score += norm.logpdf(
                    observations[feature],
                    loc=parameters["means"][feature],
                    scale=parameters["sds"][feature],
                )

        scores[:, class_value] = score

    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def evaluate_model(model, test, name):
    """Evaluate predictions; F1 treats unhealthy (y=1) as the positive class."""
    probabilities = predict_proba(model, test)
    prediction = probabilities.argmax(axis=1)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(test["y"], prediction),
        "precision": precision_score(test["y"], prediction, zero_division=0),
        "recall": recall_score(test["y"], prediction, zero_division=0),
        "f1": f1_score(test["y"], prediction, zero_division=0),
    }
    return metrics, prediction


def save_figure(fig, filename, output_dir=None):
    """Save and close a figure in figures/ or the requested output folder."""
    folder = FIGURES_DIR if output_dir is None else output_dir
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_table(table, filename, title, precision=4, output_dir=None):
    """Save a CSV and a simple PNG rendering of a results table."""
    csv_dir = DATA_DIR if output_dir is None else output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_dir / f"{filename}.csv", index=False)

    display = table.copy()
    for column in display.select_dtypes(include="floating"):
        display[column] = display[column].map(lambda value: f"{value:.{precision}f}")

    fig, ax = plt.subplots(figsize=(11, max(2.6, 0.42 * (len(table) + 2))))
    ax.axis("off")
    ax.set_title(title, pad=16)
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.auto_set_column_width(list(range(len(display.columns))))
    table_artist.scale(1, 1.65)
    save_figure(fig, f"{filename}.png", output_dir=output_dir)


def task3a_hand_calculation(training, test):
    """Print all factors for one test observation using only f1 and cat.

    This is intended to support a separately handwritten calculation. It does
    not generate the handwritten submission itself.
    """
    features = ["f1", "cat"]
    model = fit_naive_bayes(training, features, alpha=1.0)
    point = test.loc[:, features].iloc[[0]]

    print("\n" + "=" * 68)
    print("TASK 3a — HAND-CALCULATION SUPPORT (features: f1 and cat)")
    print("=" * 68)

    for class_value in (0, 1):
        params = model["classes"][class_value]
        print(f"\nClass y={class_value}")
        print(f"Training observations: {params['count']}")
        print(f"Prior P(y={class_value}): {params['prior']:.8f}")
        print(f"Mean f1: {params['means']['f1']:.8f}")
        print(f"Sample SD f1 (ddof=1): {params['sds']['f1']:.8f}")

    x_f1 = float(point["f1"].iloc[0])
    x_cat = point["cat"].iloc[0]
    print("\nTest point")
    print(f"f1 = {x_f1:.6f}")
    print(f"cat = {x_cat}")

    scores = {}
    rows = []
    print("\nFactors and unnormalized scores")
    for class_value in (0, 1):
        params = model["classes"][class_value]
        prior = params["prior"]
        gaussian = float(norm.pdf(
            x_f1,
            loc=params["means"]["f1"],
            scale=params["sds"]["f1"],
        ))
        category_count = int(params["category_counts"][x_cat])
        category_probability = float(params["category_probs"][x_cat])
        score = prior * gaussian * category_probability
        scores[class_value] = score

        print(f"\nClass y={class_value}")
        print(f"Prior: {prior:.10f}")
        print(f"Gaussian density p(f1 | y={class_value}): {gaussian:.10f}")
        print(f"Category count: {category_count}")
        print(f"P(cat | y={class_value}) with Laplace smoothing: "
              f"{category_probability:.10f}")
        print(f"Unnormalized score: {score:.12g}")

        rows.append({
            "class_y": class_value,
            "prior": prior,
            "f1_mean": params["means"]["f1"],
            "f1_sample_sd": params["sds"]["f1"],
            "gaussian_density_f1": gaussian,
            "category": x_cat,
            "category_count": category_count,
            "categorical_likelihood": category_probability,
            "unnormalized_score": score,
        })

    total_score = scores[0] + scores[1]
    posterior_y0 = scores[0] / total_score
    posterior_y1 = scores[1] / total_score

    print("\nNormalization")
    print(f"Total score: {total_score:.12g}")
    print(f"P(y=0 | x): {posterior_y0:.10f}")
    print(f"P(y=1 | x): {posterior_y1:.10f}")
    print(f"Sum of probabilities: {posterior_y0 + posterior_y1:.10f}")
    print(f"Predicted class: {int(posterior_y1 > posterior_y0)}")

    pd.DataFrame(rows).to_csv(DATA_DIR / "task3a_hand_calculation_factors.csv",
                              index=False)


def smoothing_experiment(training, test):
    """Compare alpha=0 and alpha=1 after removing rare class-1 rows.

    The held-out test data is unchanged. The test point is the first held-out
    Enhanced protection observation, irrespective of its true label.
    """
    removed = training[
        (training["y"] == 1)
        & (training["cat"] == "Enhanced protection")
    ]
    reduced = training.drop(index=removed.index).copy()

    removed[["plant_id", "y", "cat"]].to_csv(
        DATA_DIR / "task3b_removed_training_rows.csv", index=False
    )
    point = test[test["cat"] == "Enhanced protection"].iloc[[0]]
    point.to_csv(DATA_DIR / "task3b_test_plant.csv", index=False)

    print("\n" + "=" * 68)
    print("TASK 3b — SMOOTHING EXPERIMENT")
    print("=" * 68)
    print(f"Class-1 Enhanced protection rows removed from training: {len(removed)}")
    print(f"Reduced training size: {len(reduced)}")
    print("Test point is unchanged and was not used to fit the model:")
    print(point[["plant_id", "y", "f1", "f2", "f3", "f4", "cat"]].to_string(index=False))

    rows = []
    for alpha in (0, 1):
        model = fit_naive_bayes(reduced, ALL_FEATURES, alpha=alpha)
        probabilities = predict_proba(model, point)[0]
        result = {
            "alpha": alpha,
            "smoothing": "off" if alpha == 0 else "Laplace (alpha=1)",
            "removed_rows": len(removed),
            "training_n": len(reduced),
            "category_given_y0": model["classes"][0]["category_probs"]["Enhanced protection"],
            "category_given_y1": model["classes"][1]["category_probs"]["Enhanced protection"],
            "posterior_y0": probabilities[0],
            "posterior_y1": probabilities[1],
        }
        rows.append(result)
        print(f"\n{result['smoothing']}")
        print(f"P(Enhanced protection | y=0): {result['category_given_y0']:.10g}")
        print(f"P(Enhanced protection | y=1): {result['category_given_y1']:.10g}")
        print(f"P(y=1 | all observed features): {result['posterior_y1']:.10g}")

    results = pd.DataFrame(rows)
    save_table(
        results,
        "task3b_smoothing_results",
        "Task 3b: Same plant, without and with smoothing",
        precision=8,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = results["posterior_y1"].to_numpy()
    ax.bar(
        ["No smoothing (alpha=0)", "Laplace smoothing (alpha=1)"],
        values,
        color=["#777777", "#237a57"],
    )
    ax.set(
        ylabel="P(unhealthy | observed features)",
        title="Task 3b: Effect of categorical smoothing",
        ylim=(0, max(0.001, float(values.max()) * 1.35)),
    )
    for i, value in enumerate(values):
        ax.text(i, value + ax.get_ylim()[1] * 0.025, f"{value:.6g}", ha="center")
    save_figure(fig, "task3b_smoothing_comparison.png")
    return results


def compare_feature_subsets(training):
    """Select among predefined feature subsets using training-only 5-fold CV.

    Rank by mean validation F1. Exact ties prefer fewer features, then the
    original candidate order. Fold SD describes variation between folds; it
    is not a confidence interval.
    """
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        .split(training, training["y"])
    )
    details, summary = [], []

    for features in FEATURE_SUBSETS:
        fold_scores = []
        feature_name = " + ".join(features)

        for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
            fit_data = training.iloc[fit_indices]
            validation = training.iloc[validation_indices]
            model = fit_naive_bayes(fit_data, features, alpha=1)
            prediction = predict_proba(model, validation).argmax(axis=1)
            score = f1_score(validation["y"], prediction, zero_division=0)
            fold_scores.append(score)
            details.append({
                "features": feature_name,
                "fold": fold_number,
                "f1": score,
            })

        summary.append({
            "features": feature_name,
            "n_features": len(features),
            "mean_f1": np.mean(fold_scores),
            "sd_f1": np.std(fold_scores, ddof=1),
        })

    results = pd.DataFrame(summary).sort_values(
        ["mean_f1", "n_features"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    pd.DataFrame(details).to_csv(DATA_DIR / "task3c_fold_scores.csv", index=False)
    save_table(
        results,
        "task3c_feature_subsets",
        "Task 3c: Training-only five-fold validation F1",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        results["features"],
        results["mean_f1"],
        xerr=results["sd_f1"],
        capsize=4,
        color="#237a57",
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Mean validation F1 (error bars: fold standard deviation)",
        xlim=(0, 1.03),
        title="Task 3c: Manually chosen feature subsets",
    )
    save_figure(fig, "task3c_feature_comparison.png")

    print("\n" + "=" * 68)
    print("TASK 3c — FEATURE SUBSET COMPARISON")
    print("=" * 68)
    print(results.to_string(index=False))
    best_features = results.iloc[0]["features"].split(" + ")
    print(f"\nSelected features: {best_features}")
    return best_features, results


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    environment = [{"package": "Python", "version": platform.python_version()}]
    environment += [
        {"package": package, "version": version(package)}
        for package in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
    ]
    pd.DataFrame(environment).to_csv(DATA_DIR / "task3_environment.csv", index=False)

    dataset_path = DATA_DIR / "task1_synthetic_data.csv"
    if not dataset_path.exists():
        generate_dataset().to_csv(dataset_path, index=False)
        print("Recreated missing Task 1 CSV using the Task 1 generator.")

    data = pd.read_csv(dataset_path)
    training, test = train_test_split(
        data,
        test_size=0.25,
        random_state=SEED,
        stratify=data["y"],
    )

    split = pd.concat([
        training[["plant_id", "y"]].assign(partition="training"),
        test[["plant_id", "y"]].assign(partition="test"),
    ])
    split.to_csv(DATA_DIR / "task3_split.csv", index=False)

    print(f"Training set: {len(training)} plants; test set: {len(test)} plants.")

    # Task 3a: keep the hand calculation limited to f1 and cat.
    task3a_hand_calculation(training, test)

    # Full five-feature model and held-out test metrics.
    full_model = fit_naive_bayes(training, ALL_FEATURES, alpha=1)
    full_metrics, full_predictions = evaluate_model(
        full_model, test, "All five features"
    )

    matrix = confusion_matrix(test["y"], full_predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_healthy", "actual_unhealthy"],
        columns=["predicted_healthy", "predicted_unhealthy"],
    ).to_csv(DATA_DIR / "task3_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=["Healthy", "Unhealthy"]
    ).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Task 3: Full-model test confusion matrix")
    save_figure(fig, "task3_confusion_matrix.png")

    # Task 3b: only a copy of training data is altered for this experiment.
    smoothing_results = smoothing_experiment(training, test)

    # Task 3c: select features using training folds, then evaluate once on test.
    best_features, _ = compare_feature_subsets(training)
    selected_model = fit_naive_bayes(training, best_features, alpha=1)
    selected_metrics, _ = evaluate_model(
        selected_model, test, "CV selected: " + " + ".join(best_features)
    )

    metrics = pd.DataFrame([full_metrics, selected_metrics])
    save_table(
        metrics,
        "task3_test_metrics",
        "Task 3: Held-out test metrics (positive class: unhealthy)",
    )

    print("\n" + "=" * 68)
    print("HELD-OUT TEST METRICS")
    print("=" * 68)
    print(metrics.to_string(index=False))
    print("\nTask 3b results:")
    print(smoothing_results.to_string(index=False))
    print("\nOutputs saved in data/ and figures/.")


if __name__ == "__main__":
    main()

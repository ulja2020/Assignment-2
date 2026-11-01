"""MOD550 Assignment 2, Task 1: synthetic maize health observations.

Run this file to regenerate the CSV dataset and all Task 1 figures/tables.
The only predictors are f1, f2, f3, f4 and cat. Positions and plant_id are
metadata. Class 1 means pest-affected (unhealthy); class 0 means healthy.
All agricultural parameters are illustrative, not field-calibrated.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring an interactive window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
N_PLANTS = 1200
N_UNHEALTHY = 300
FEATURES = ["f1", "f2", "f3", "f4"]
CATEGORIES = ["No treatment", "Biological control", "Standard treatment",
              "Enhanced protection"]
CATEGORY_COUNTS = {0: [180, 225, 225, 270], 1: [150, 90, 57, 3]}
CLASS_LABELS = {0: "Healthy (y = 0)", 1: "Unhealthy (y = 1)"}
COLORS = {0: "#237a57", 1: "#bd492d"}
# Project root is the parent folder of src/.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"


def generate_dataset(seed=SEED):
    """Return 1,200 reproducible plant observations as a DataFrame.

    Parameters
    ----------
    seed : int
        Seed for NumPy's local random generator; the default is 42.

    Returns
    -------
    pandas.DataFrame
        Columns: plant_id, x_position, y_position, y, f1 to f4 and cat.
        Positions are unitless fractions of field width and height.

    Notes
    -----
    Select exactly 300 unhealthy plants without replacement, with selection
    weights exp(-4*x). Thus plants near the western edge (x=0) are more
    likely to be affected. These weights are not individual probabilities
    of disease; they are normalized weights for a fixed-size sample.

    f1 is ear length: N(20, 2**2) for healthy plants and N(15, 2**2) for
    unhealthy plants. Values are neither clipped nor truncated.
    f2 is a standardized reflectance anomaly. Healthy plants use N(0, 0.7**2).
    Unhealthy plants use two equally represented components N(-3, 0.7**2)
    and N(3, 0.7**2), representing two hypothetical damage responses.
    f3 and f4 are standardized pigment and fluorescence deviations. Their
    theoretical marginal distributions are N(0, 1) in both classes. Their
    correlation is -0.8 in class 0 and +0.8 in class 1.
    Treatment labels are shuffled within each class using exact counts;
    this encodes an association, not a causal treatment experiment.
    """
    rng = np.random.default_rng(seed)
    x_position = rng.uniform(0, 1, N_PLANTS)
    y_position = rng.uniform(0, 1, N_PLANTS)
    weights = np.exp(-4 * x_position)
    unhealthy_indices = rng.choice(
        N_PLANTS, size=N_UNHEALTHY, replace=False, p=weights / weights.sum()
    )
    y = np.zeros(N_PLANTS, dtype=int)
    y[unhealthy_indices] = 1
    f1 = rng.normal(np.where(y == 0, 20, 15), 2)
    f2 = np.empty(N_PLANTS)
    cat = np.empty(N_PLANTS, dtype=object)
    for class_value in (0, 1):
        mask = y == class_value
        count = int(mask.sum())
        if class_value == 0:
            f2[mask] = rng.normal(0, 0.7, count)
        else:
            component_means = np.repeat([-3.0, 3.0], count // 2)
            rng.shuffle(component_means)
            f2[mask] = rng.normal(component_means, 0.7)
        levels = np.repeat(CATEGORIES, CATEGORY_COUNTS[class_value])
        rng.shuffle(levels)
        cat[mask] = levels

    z1 = rng.normal(size=N_PLANTS)
    z2 = rng.normal(size=N_PLANTS)
    rho = np.where(y == 1, 0.8, -0.8)
    f3 = z1
    f4 = rho * z1 + np.sqrt(1 - rho**2) * z2
    return pd.DataFrame({
        "plant_id": [f"plant_{i:04d}" for i in range(1, N_PLANTS + 1)],
        "x_position": x_position, "y_position": y_position, "y": y,
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "cat": cat,
    })


def make_summary_tables(data):
    """Return class balance, category counts and descriptive diagnostics.

    Category percentages use the class total as their denominator. Diagnostics
    help compare the realized sample with the intended generating process.
    """
    balance = pd.DataFrame({"y": [0, 1], "class": ["Healthy", "Unhealthy"],
                            "count": [(data.y == 0).sum(), (data.y == 1).sum()]})
    balance["percent"] = 100 * balance["count"] / len(data)
    counts = pd.crosstab(data["cat"], data["y"]).reindex(CATEGORIES)
    categories = pd.DataFrame({
        "category": CATEGORIES, "healthy_count": counts[0].to_numpy(),
        "unhealthy_count": counts[1].to_numpy(),
        "healthy_percent": 100 * counts[0].to_numpy() / (data.y == 0).sum(),
        "unhealthy_percent": 100 * counts[1].to_numpy() / (data.y == 1).sum(),
    })
    rows = []
    for class_value in (0, 1):
        subset = data[data.y == class_value]
        row = {"y": class_value}
        for feature in FEATURES:
            row[f"{feature}_mean"] = subset[feature].mean()
            row[f"{feature}_sd"] = subset[feature].std(ddof=1)
        row["f3_f4_correlation"] = subset.f3.corr(subset.f4)
        row["same_sign_percent"] = 100 * (subset.f3 * subset.f4 > 0).mean()
        row["mean_x_position"] = subset.x_position.mean()
        rows.append(row)
    return balance, categories, pd.DataFrame(rows)


def save_figure(fig, folder, filename):
    """Save a readable PNG and close its figure to release memory."""
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_table_image(table, headers, title, folder, filename):
    """Render a short summary table as a PNG, with explicit column labels."""
    fig, ax = plt.subplots(figsize=(11, 2.7))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=14)
    formatted = table.copy()
    for column in formatted.select_dtypes(include="floating"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.1f}")
    image_table = ax.table(cellText=formatted.values, colLabels=headers,
                           loc="center", cellLoc="center")
    image_table.auto_set_font_size(False)
    image_table.set_fontsize(10)
    image_table.auto_set_column_width(list(range(len(headers))))
    image_table.scale(1, 1.7)
    for (row, _), cell in image_table.get_celld().items():
        cell.set_edgecolor("#d5ded9")
        if row == 0:
            cell.set_facecolor("#e0ece5")
            cell.set_text_props(weight="bold")
    save_figure(fig, folder, filename)


def plot_features(data, folder):
    """Plot marginal distributions, joint signs, treatment and field position."""
    labels = ["f1: Ear length (cm)", "f2: Reflectance anomaly (standardized)",
              "f3: Pigment deviation (standardized)",
              "f4: Fluorescence deviation (standardized)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature, label in zip(axes.flat, FEATURES, labels):
        bins = np.linspace(data[feature].min(), data[feature].max(), 32)
        for class_value in (0, 1):
            ax.hist(data.loc[data.y == class_value, feature], bins=bins,
                    density=True, alpha=0.5, color=COLORS[class_value],
                    label=CLASS_LABELS[class_value])
        ax.set(xlabel=label, ylabel="Density")
        ax.legend(fontsize=8)
    fig.suptitle("Task 1: Feature distributions within each class")
    save_figure(fig, folder, "task1_feature_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    limits = (data[["f3", "f4"]].min().min() - 0.2,
              data[["f3", "f4"]].max().max() + 0.2)
    for class_value, ax in enumerate(axes):
        subset = data[data.y == class_value]
        correlation = subset.f3.corr(subset.f4)
        ax.scatter(subset.f3, subset.f4, s=12, alpha=0.5, color=COLORS[class_value])
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set(xlabel="f3: Pigment deviation", ylabel="f4: Fluorescence deviation",
               title=f"{CLASS_LABELS[class_value]}; r = {correlation:.2f}",
               xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Task 1: Joint patterns reveal class information")
    save_figure(fig, folder, "task1_joint_features.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for class_value, offset in [(0, -0.2), (1, 0.2)]:
        frequencies = data.loc[data.y == class_value, "cat"].value_counts(
            normalize=True).reindex(CATEGORIES)
        ax.bar(np.arange(4) + offset, 100 * frequencies, width=0.4,
               color=COLORS[class_value], label=CLASS_LABELS[class_value])
    ax.set_xticks(np.arange(4), CATEGORIES)
    ax.set(ylabel="Percentage within class", title="Task 1: Treatment frequencies by class")
    ax.legend()
    save_figure(fig, folder, "task1_category_frequencies.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_value in (0, 1):
        subset = data[data.y == class_value]
        ax.scatter(subset.x_position, subset.y_position, s=14, alpha=0.6,
                   color=COLORS[class_value], label=CLASS_LABELS[class_value])
    ax.axvline(0, color=COLORS[1], linewidth=3)
    ax.set(xlabel="Relative west-to-east position (pests enter at x = 0)",
           ylabel="Relative south-to-north position", xlim=(-0.02, 1.02),
           ylim=(-0.02, 1.02), title="Task 1: Synthetic French maize field")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, folder, "task1_field_map.png")


def main():
    """Generate CSV files in data/ and figures in figures/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(DATA_DIR / "task1_synthetic_data.csv", index=False)
    folder = FIGURES_DIR
    balance, categories, diagnostics = make_summary_tables(data)
    for name, table in [("class_balance", balance), ("category_counts", categories),
                        ("diagnostics", diagnostics)]:
        table.to_csv(DATA_DIR / f"task1_{name}.csv", index=False)
    save_table_image(balance, ["y", "Class", "Count", "Percent (%)"],
                     "Task 1: Class balance", folder, "task1_class_balance.png")
    save_table_image(categories,
                     ["Treatment", "Healthy n", "Unhealthy n", "Healthy (%)", "Unhealthy (%)"],
                     "Task 1: Category counts and percentages within each class",
                     folder, "task1_category_counts.png")
    plot_features(data, folder)
    print("Task 1 completed.")
    print(f"Main dataset and summary CSVs: {DATA_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(balance.to_string(index=False))
    print(diagnostics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Task 3a: function-output reference for comparison with handwritten work.

The hand calculation will be presented separately on a sheet of paper,
handwritten. The only objective of this script and its outputs is to support
Task 3a's requirement to show that the handwritten result matches the
function's output. It saves the selected test plant, fitted factors and
function posterior beside this file.

The direct_posterior column is an automatically calculated arithmetic
reference, not a handwritten result. absolute_difference compares this
reference with the function output; it does not verify the paper calculation.
The handwritten result must be compared separately with function_posterior.
Run: python task_3a_claudio/task3a_handcalculation_support_compare.py
Shared imports do not run Tasks 3b/3c. No handwritten solution is generated.
"""

from pathlib import Path
import sys

# Import the shared classifier from the neighbouring src folder.
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT_DIR.parent / "src"))

import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from task3_naive_bayes import DATA_DIR, SEED, fit_naive_bayes, predict_proba, save_table


# %% Task 3a: comparison reference using f1 and cat only.
def hand_calculation_factors(training, point):
    """Export a reference for checking a separately handwritten calculation.

    Fit only f1 and cat on the original training set. Select the first test
    row without filtering by its label. direct_posterior is computed by code;
    function_posterior is returned by the classifier. Their difference checks
    the two code paths, not agreement with an unprovided handwritten result.
    """
    model = fit_naive_bayes(training, ["f1", "cat"], alpha=1)
    observation = point.iloc[0]
    rows = []
    for class_value in (0, 1):
        p = model["classes"][class_value]
        density = norm.pdf(observation.f1, p["means"]["f1"], p["sds"]["f1"])
        cat_probability = p["category_probs"][observation["cat"]]
        rows.append({"y": class_value, "class_count": p["count"],
                     "prior": p["prior"], "mean_f1": p["means"]["f1"],
                     "sd_f1": p["sds"]["f1"], "f1_density": density,
                     "category_count": int(p["category_counts"][observation["cat"]]),
                     "category_probability": cat_probability,
                     "score": p["prior"] * density * cat_probability})
    factors = pd.DataFrame(rows)
    factors["normalization"] = factors.score.sum()
    factors["direct_posterior"] = factors.score / factors.score.sum()
    factors["function_posterior"] = predict_proba(model, point)[0]
    factors["absolute_difference"] = abs(factors.direct_posterior - factors.function_posterior)
    factors.to_csv(OUTPUT_DIR / "task3a_hand_factors.csv", index=False)
    point[["plant_id", "f1", "cat", "y"]].to_csv(OUTPUT_DIR / "task3a_test_plant.csv", index=False)
    save_table(factors[["y", "prior", "f1_density", "category_probability",
                         "direct_posterior", "function_posterior"]],
               "task3a_posterior_comparison", "Task 3a: Function output and automatic arithmetic reference", 6, output_dir=OUTPUT_DIR)
    return factors


def main():
    """Save the function-output reference; handwritten work is submitted separately."""
    data = pd.read_csv(DATA_DIR / "task1_synthetic_data.csv")
    training, test = train_test_split(
        data, test_size=0.25, random_state=SEED, stratify=data.y
    )
    factors = hand_calculation_factors(training, test.iloc[[0]])
    print("Task 3a plant:", test.iloc[0]["plant_id"])
    print(factors[["y", "direct_posterior", "function_posterior"]].to_string(index=False))
    print("Automatic reference/function difference:", factors.absolute_difference.max())
    print("Hand calculation: handwritten on a separate sheet; compare with function_posterior.")
    print("Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

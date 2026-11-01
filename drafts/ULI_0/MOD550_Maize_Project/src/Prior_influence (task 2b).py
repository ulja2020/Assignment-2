
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta
from pathlib import Path

df = sns.load_dataset("titanic")

small = df[
    (df.sex == "female") &
    (df["class"] == "Second") &
    (df.age < 16)
]

large = df[
    (df.sex == "male") &
    (df["class"] == "Third")
]

a, b = 20, 20  # same prior for both cases
x = np.linspace(0.001, 0.999, 1000)

project_dir = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals() and Path(__file__).resolve().parent.name == "src"
    else Path.cwd()
)
figures_dir = project_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

for name, sub in [
    ("Small sample: second-class girls under 16", small),
    ("Large sample: third-class men", large)
]:
    n = len(sub)
    k = int(sub.survived.sum())

    prior = beta(a, b)
    post = beta(a + k, b + n - k)

    # Likelihood shape, normalized to integrate to 1 for plotting
    like = beta(k + 1, n - k + 1)

    lo, hi = post.interval(0.95)

    print("\n" + name)
    print("n =", n, ", k =", k)
    print("Observed survival rate =", round(k / n, 3))
    print("Posterior = Beta(", a + k, ",", b + n - k, ")")
    print("Posterior mean =", round(post.mean(), 3))
    print("95% credible interval =", (round(lo, 3), round(hi, 3)))

    plt.figure(figsize=(9, 5))
    plt.plot(x, prior.pdf(x), label="Prior")
    plt.plot(x, like.pdf(x), label="Likelihood (scaled)")
    plt.plot(x, post.pdf(x), label="Posterior")
    plt.xlabel("Survival probability (theta)")
    plt.ylabel("Density")
    plt.title(name)
    plt.legend()
    plt.tight_layout()

    filename = (
        "task2b_small_sample.png"
        if n < 50 else "task2b_large_sample.png"
    )
    plt.savefig(figures_dir / filename, dpi=300)
    plt.show()
    plt.close()

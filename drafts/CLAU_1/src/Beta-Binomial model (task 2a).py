import pandas as pd
from scipy.stats import beta

df = pd.read_csv("data/task1_synthetic_data.csv")

n = len(df)             # total number of trials
k = int(df["y"].sum())  # number of successes

a, b = 2, 2  # prior

def posterior_update(a, b, k, n):  # function that takes a Beta prior (a, b) and Binomical data (n, k) and returns the posterior
    return beta(a + k, b + n - k)

post = posterior_update(a, b, k, n)

print("Prior:",a, b)
print("Posterior:", a + k, b + n - k)
print("Posterior mean:", post.mean())
print("Posterior mode:", ((a + k)- 1) / ((a + k) + (b + n - k) - 2))
print("95% credible interval:", post.interval(0.95))
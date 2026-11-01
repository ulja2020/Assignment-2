# MOD550 Assignment 2: Bayesian Thinking and Classification

## Task 1: Synthetic data

### Objective

Create a reproducible dataset with at least 1,000 observations, a binary target
with 15–35% positives, four numerical features with the specified distribution
patterns, and a categorical feature with at least four levels. Explore how the
features relate to the target and report class balance and category counts.

### Development steps

1. Define the maize pest scenario and the feature distributions.
2. Generate plant positions and assign exactly 300 unhealthy plants.
3. Generate four numerical features and one treatment category per plant.
4. Summarize class balance, category counts and numerical patterns.
5. Save the data, plots and summary tables; discuss the observed patterns.

### Scenario and geographical setting

We continue the synthetic French maize-field setting from Assignment 1, this
time for a harvest year affected by pests. The earlier project's README and
generator specify a French field but no actual coordinates or boundaries.
Therefore, the map uses relative positions between 0 and 1, not latitude,
longitude or measured distances. It represents the same conceptual field;
the positions are newly simulated metadata, not a surveyed field boundary.

Each row represents one sampled maize plant, with ear length measured at
harvest and illustrative leaf measurements from the affected growing season.
The pest attack is assumed to start early enough to affect ear development.
Pests enter from the western edge (`x_position = 0`). We generate uniformly
distributed positions, then select exactly 300 unhealthy plants without
replacement using weights `exp(-4 * x_position)`. Selection is more likely near
the western edge, with overlap between healthy and unhealthy plants throughout
the field. The weights are not individual disease probabilities.

There are 1,200 observations: `y = 0` means healthy and `y = 1` means unhealthy
due to pest attack. Thus, the positive class is 25% of the observations.
If a Boolean healthy indicator is needed, use `healthy = (y == 0)`.

### Feature definitions and generation

`N(mean, variance)` below describes the theoretical generating distribution.

| Column | Meaning | Generation |
|---|---|---|
| `f1` | Ear length, cm | Healthy: N(20, 4); unhealthy: N(15, 4). Both standard deviations are 2 cm. |
| `f2` | Standardized leaf reflectance anomaly | Healthy: N(0, 0.49). Unhealthy: equal mixture of N(-3, 0.49) and N(3, 0.49). |
| `f3` | Standardized leaf pigment deviation | N(0, 1) in both classes, jointly generated with f4. |
| `f4` | Standardized leaf fluorescence deviation | N(0, 1) in both classes; correlation with f3 is -0.8 for healthy and +0.8 for unhealthy plants. |
| `cat` | Pest-management treatment | Four levels with exact class-dependent counts shown below. |

For `f2`, each of the three Gaussian components has standard deviation 0.7:
the one healthy component and the two unhealthy components. The unhealthy
mixture contains exactly 150 draws from each component, shuffled among the
300 unhealthy plants. These two hypothetical damage responses create low and
high reflectance anomalies. They are not measured pest subtypes.

`f3` and `f4` are simulated, standardized deviations representing variation
around typical pigment and fluorescence measurements **within each health class
(healthy and unhealthy plants)**. Zero represents the centre of these simulated
values for that class. The code generates `f3` and `f4` directly around zero; it
does not start with actual pigment or fluorescence measurements and calculate
their deviations.

For `f3` and `f4`, generate independent standard normal variables z1 and z2:

```text
f3 = z1
f4 = rho*z1 + sqrt(1-rho**2)*z2
rho = -0.8 for y=0; +0.8 for y=1
```

This construction gives both features the same symmetric N(0, 1) marginal
distribution in each class. Their joint pattern differs: matching signs are
usual for unhealthy plants and opposing signs for healthy plants. Correlations
below magnitude 1 produce the required overlap. Individual finite samples
will not have exactly identical means or standard deviations.

Ear length is sampled directly from the Gaussian distributions, without clipping
or truncation. The smaller mean for unhealthy plants represents reduced ear
development in this synthetic story. All parameter values and leaf responses
are illustrative assumptions, not calibrated agricultural estimates.

### Class balance

| y | Class | Count | Percentage |
|---|---|---:|---:|
| 0 | Healthy | 900 | 75% |
| 1 | Unhealthy | 300 | 25% |

### Category counts per class

| Treatment | Healthy | Unhealthy | Healthy % | Unhealthy % |
|---|---:|---:|---:|---:|
| No treatment | 180 | 150 | 20.0 | 50.0 |
| Biological control | 225 | 90 | 25.0 | 30.0 |
| Standard treatment | 225 | 57 | 25.0 | 19.0 |
| Enhanced protection | 270 | 3 | 30.0 | 1.0 |

Percentages use the total within each class: 900 or 300. Treatment labels are
shuffled within classes, so the table encodes a deliberately chosen association.
Enhanced protection is common among healthy plants (270) but occurs in only
three unhealthy plants, meeting the requirement of at most five. Because class
labels and categories are generated by design, these numbers do not establish
real treatment effectiveness. Treatments are not simulated as contiguous plots.

### Results and reflections

- **Ear length:** sample means are 20.011 cm for healthy and 15.124 cm for
  unhealthy plants; sample standard deviations are 1.984 and 1.959 cm. Ear
  length is informative but the two Gaussian distributions overlap. A standard
  deviation of 2 cm gives more separation than the initially proposed 2.5 cm.
- **Reflectance:** healthy plants have one central mode; unhealthy plants have
  two separated modes. Their mean alone would conceal this two-mode structure.
- **Pigment and fluorescence:** sample correlations are -0.814 for healthy
  plants and +0.817 for unhealthy plants. Matching signs occur in 20.7% and
  78.7% of plants, respectively. Each feature alone has the same theoretical
  distribution across classes, while their combination supplies information.
- **Treatments:** class-normalized percentages make the association visible
  despite the unequal class sizes. The rare category will be relevant when
  estimating probabilities from small class-specific counts in later tasks.
- **Geography:** mean relative west-to-east positions are 0.578 for healthy and
  0.267 for unhealthy plants, consistent with infestation from the western edge.

The required predictors are only `f1`, `f2`, `f3`, `f4` and `cat`.
`plant_id`, `x_position` and `y_position` are identifiers/map metadata; `y` is
the target. Do not pass the target or a derived healthy indicator as a predictor.
No classifiers are fitted in Task 1.

### Files and figures

At the working-folder level:

- `src/task1_generate.py`: documented generator, tables and plots.
- `data/task1_synthetic_data.csv`: 1,200 rows, including predictors and map metadata.
- `README.md`: objectives, development steps, assumptions and results.
- `AI_LOG.md`: AI assistance record for this chat.

Task 1 generator outputs (PNG files in `figures/`, CSV files in `data/`):

- `task1_feature_distributions.png`: four class-normalized histograms.
- `task1_joint_features.png`: f3–f4 scatter plots, one per class.
- `task1_category_frequencies.png`: treatment percentages within each class.
- `task1_field_map.png`: plant positions coloured by health class.
- `data/task1_class_balance.csv` and `figures/task1_class_balance.png`: class-balance table.
- `data/task1_category_counts.csv` and `figures/task1_category_counts.png`: category counts and percentages.
- `data/task1_diagnostics.csv`: numerical sample statistics supporting this discussion.

### Reproduction and validation

From this folder, run:

```text
python src/task1_generate.py
```

Dependencies: NumPy, pandas and Matplotlib. The completed run used Python 3.14,
NumPy 2.5.3, pandas 3.0.6 and Matplotlib 3.11.2, with random seed 42.
Files are saved relative to the script location, regardless of the current
terminal directory. Rerunning replaces this script's Task 1 data and figures.
README and AI_LOG are maintained separately and are not overwritten by the script.

Automated validation checks are not included in the script. The generation
settings specify 1,200 observations, 25% unhealthy plants and three unhealthy
plants in the enhanced-protection category. The generated dataset was checked
during development, including for missing values, reproducibility and CSV
round-trip consistency. These development checks are not part of the delivered
script. The class-balance and category-count tables remain available for review.
Gaussian shapes and theoretical marginal equality are properties of the
construction; finite sample diagnostics illustrate them rather than proving
them statistically.

## Task 2: Bayesian updating

### Task 2a: Posterior update

#### Objective

Implement a Beta–Binomial Bayesian update for the probability that a maize plant is unhealthy. Start with a Beta prior, update it using observed binary outcomes, and report the posterior mean, mode and 95% credible interval.

#### Agricultural scenario

The dataset represents 1,200 synthetic maize plants. Each observation represents one plant, and the target variable `y` indicates its health status:

- `y = 0`: healthy plant
- `y = 1`: unhealthy plant affected by pests

The parameter of interest is

\[
\theta=P(y=1),
\]

the probability that a plant is unhealthy in the population represented by this synthetic dataset.

The dataset is illustrative and is not calibrated to real agricultural field measurements.

#### Prior and observed data

The analysis uses a Beta(2, 2) prior:

\[
\theta\sim\operatorname{Beta}(2,2).
\]

This prior is symmetric, with prior mean 0.5. It represents uncertainty about the probability before incorporating the observations.

The dataset contains:

- Total observations: \(n=1200\)
- Unhealthy plants: \(k=300\)
- Healthy plants: \(n-k=900\)

Here, a *success* in the Binomial model is defined as observing an unhealthy plant (`y = 1`). The term *success* is a statistical convention and does not mean that pest damage is desirable.

#### Bayesian update

For a Beta prior and Binomial observations, the posterior distribution is

\[
\theta\mid data\sim
\operatorname{Beta}(\alpha+k,\beta+n-k).
\]

With \(\alpha=2\), \(\beta=2\), \(k=300\) and \(n=1200\):

\[
\theta\mid data\sim\operatorname{Beta}(302,902).
\]

The posterior mean is

\[
E[\theta\mid data]=
\frac{\alpha_{\text{post}}}
{\alpha_{\text{post}}+\beta_{\text{post}}}
=\frac{302}{1204}
\approx 0.25083.
\]

For posterior parameters greater than one, the posterior mode is

\[
\frac{\alpha_{\text{post}}-1}
{\alpha_{\text{post}}+\beta_{\text{post}}-2}
\approx 0.25042.
\]

#### Results

| Quantity | Result |
|---|---:|
| Prior | Beta(2, 2) |
| Posterior | Beta(302, 902) |
| Posterior mean | 0.25083 |
| Posterior mode | 0.25042 |
| 95% credible interval | (0.22675, 0.27569) |

The posterior mean estimates the probability of an unhealthy plant at approximately 25.08%. The 95% credible interval is approximately 22.68%–27.57%.

The observed unhealthy proportion is \(300/1200=0.25\). Because the dataset is large, the likelihood has much more influence on the posterior than the relatively weak Beta(2, 2) prior. The posterior estimate is therefore close to the observed proportion.

#### Interpretation and assumptions

The model treats the health outcomes as Binomial observations conditional on a common probability \(\theta\). The Beta prior is conjugate to the Binomial likelihood, so the posterior is also a Beta distribution.

The credible interval expresses uncertainty about \(\theta\), conditional on the model, prior and observed data. These results describe the generated dataset and should not be interpreted as an empirical estimate of real-world maize pest prevalence.

#### Reproducibility

The calculation uses the generated maize CSV dataset. The code reads the binary target `y`, counts the number of observations and the number of `y = 1` outcomes, updates the Beta parameters, and reports the posterior summaries.

The reported output was:

```text
Prior: 2 2
Posterior: 302 902
Posterior mean: 0.25083056478405313
Posterior mode: 0.2504159733777038
95% credible interval: (0.22675353610305637, 0.27569162751755383)
```

The posterior interval was calculated using SciPy's Beta distribution interval method. The analytical update itself uses the Beta–Binomial conjugacy formula.

### Task 2b: When does the prior matter?

#### Objective

Investigate how strongly the prior influences a Bayesian posterior in two real-life examples. Compare a small sample, where the prior can have a noticeable effect, with a larger sample, where the likelihood generally has more influence.

The analysis uses the Titanic passenger dataset distributed through Seaborn. The outcome is whether a passenger survived.

#### Dataset and source

The Titanic dataset is accessed with Seaborn's `load_dataset("titanic")` function. It is a historical passenger dataset, not synthetic data.

Dataset access and documentation:
- Seaborn dataset-loading documentation: https://seaborn.pydata.org/generated/seaborn.load_dataset.html
- Seaborn's public dataset repository: https://github.com/mwaskom/seaborn-data

The analysis defines survival as a binary outcome:

- `survived = 1`: passenger survived
- `survived = 0`: passenger did not survive

The parameter of interest is

\[
\theta=P(\text{survived}=1)
\]

within each selected passenger group.

#### Prior distribution

Both examples use the same prior:

\[
\theta\sim\operatorname{Beta}(20,20).
\]

The prior mean is

\[
E[\theta]=\frac{20}{20+20}=0.5.
\]

The prior is centred at a survival probability of 50%, with a concentration of 40 in its two parameters. This is more concentrated than a uniform Beta(1, 1) prior and makes the effect of a small sample easier to illustrate.

Using the same prior in both cases allows the effect of sample size to be compared directly.

#### Case 1: Small sample — second-class girls under 16

The first example considers girls under 16 travelling in second class. In the selected data, the sample has \(n=10\) passengers and \(k=10\) survivors.

The posterior is

\[
\theta\mid data\sim
\operatorname{Beta}(20+10,20+10-10)
=\operatorname{Beta}(30,20).
\]

Because the sample is small, the prior still has a noticeable influence on the posterior. Even though every passenger in this sample survived, the posterior is not concentrated at a survival probability of 1.

This example illustrates why small samples should be interpreted cautiously: the observed proportion alone does not fully determine the posterior.

#### Case 2: Larger sample — third-class women

The second example considers women travelling in third class. In the selected data, the sample has \(n=144\) passengers and \(k=72\) survivors.

The posterior is

\[
\theta\mid data\sim
\operatorname{Beta}(20+72,20+144-72)
=\operatorname{Beta}(92,92).
\]

The observed survival proportion is \(72/144=0.5\), equal to the prior mean. The larger sample produces a more concentrated posterior than the prior, showing how a larger amount of data can reduce uncertainty.

#### Comparing prior, likelihood and posterior

For each case, the same plot displays:

1. **Prior:** the density of Beta(20, 20).
2. **Likelihood:** a normalized likelihood curve proportional to the Binomial likelihood. For plotting, its shape can be represented by Beta(\(k+1,n-k+1\)).
3. **Posterior:** the updated Beta distribution.

The likelihood curve is normalized only to make its scale comparable to the prior and posterior densities on the same axes. It is not a separate prior assumption.

Suggested output filenames:

- `figures/task2b_small_sample.png`
- `figures/task2b_large_sample.png`

#### Credible intervals and interpretation

Report the 95% credible interval for each posterior, then compare their widths.

- In the small-sample case, the prior has greater relative influence because there are few observations.
- In the larger-sample case, the likelihood is more informative and the posterior is more concentrated.
- A narrower credible interval indicates less posterior uncertainty under the specified model and prior.

The two examples differ in both passenger group and sample size, so the comparison illustrates prior influence in these particular cases rather than isolating sample size under every possible circumstance. The Titanic dataset is observational; the analysis describes associations and does not establish causal effects of age, sex or passenger class.

#### Reproducibility

The analysis can be reproduced by loading the Titanic dataset, selecting the two passenger groups, counting \(n\) and \(k\), updating the Beta parameters, and plotting the prior, normalized likelihood and posterior.

The selected group counts should be checked against the actual dataset version used. If filtering rules or the dataset version change, the counts and posterior results may change as well.

---

## Task 3: Naive Bayes classification

### Objective

Classify maize plants as healthy (y=0) or unhealthy (y=1) using Task 1 data.
Demonstrate a two-feature posterior calculation, investigate zero categorical
counts and smoothing, and compare manually chosen feature subsets.

### Development steps

1. Load the existing Task 1 CSV; recreate it from the unchanged generator only
   if it is missing.
2. Make a fixed, stratified 75% training / 25% test split.
3. Fit a mixed Gaussian/categorical Naive Bayes model on training data.
4. Prepare the function-output reference for the separately handwritten Task 3a calculation.
5. Run the separate zero-frequency experiment for subtask 3b.
6. Compare seven predefined subsets using training-only cross-validation for 3c.
7. Save results and figures, and discuss assumptions and limitations.

### Model, assumptions and classroom references

Use a Gaussian likelihood per numerical feature and class, and a categorical
likelihood for treatment. Estimate class priors from training proportions.
The default is Laplace smoothing (alpha=1) for categories; Gaussian standard
deviations use n_c-1, matching PDF page 7 of *Gaussian Naive Bayes.pdf*.
No scaling or tuning is required here. A tiny standard-deviation floor prevents
division by zero for a constant feature; it is not categorical smoothing.

Gaussian likelihoods keep the model simple and match f1's design. They assume
each numerical feature is normally distributed within each class. For f2, a
single Gaussian cannot represent the two unhealthy modes. Naive Bayes also
assumes conditional independence of the features given the class. This fails
for f3 and f4: their correlations have opposite signs across classes. Their
separate theoretical marginal distributions provide no class information,
although finite-sample estimates can introduce small apparent differences.
The classifier does not model their useful joint sign pattern.

Only f1, f2, f3, f4 and cat are predictors. Plant IDs and field positions remain
metadata. The target and derived health labels are excluded from the inputs.
The four allowed treatment levels are known from the experimental design.

*Gaussian Naive Bayes.pdf*, PDF pages 3–4, 7, 14–16 and 18–22, provides the
Gaussian density, conditional independence, parameter estimation and posterior
normalization. *10.07.2026 Random_Forest-Part 1.pdf*, PDF pages 50–54, supports
the confusion matrix, precision, recall and F1 definitions. The SVM classroom
scripts provide the style for train/test splitting and evaluation, but Task 3
requires a Naive Bayes classifier rather than SVM or Random Forest.

### Training/test split and full-model results

The split uses `train_test_split`, `test_size=0.25`, `random_state=42` and
`stratify=y`: 900 training plants (675 healthy, 225 unhealthy),
and 300 test plants (225 healthy, 75 unhealthy). The split is saved in
`data/task3_split.csv`. Model parameters are fitted using training data only.

| model | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| All five features | 0.9733 | 0.9241 | 0.9733 | 0.9481 |
| CV selected: f1 + f2 + cat | 0.9767 | 0.9359 | 0.9733 | 0.9542 |

F1, precision and recall use unhealthy plants as the positive class. Always
predicting healthy would give 75% accuracy but zero unhealthy recall and F1,
so accuracy alone is insufficient. The confusion-matrix CSV and figure show
the errors of the full five-feature model. The second row above is the final
test evaluation of the subset selected using training cross-validation.
The full model correctly classifies 219 healthy and 73 unhealthy test plants,
with 6 false alarms and 2 missed unhealthy plants.
These results measure performance under the synthetic sampling scheme; the
random split does not test transfer to another field, year or spatial region.

### Task 3a: Hand calculation

Use one test plant and only f1 and cat. Estimate a Gaussian density for f1 and
a smoothed treatment probability in each class. Multiply the prior and the
two likelihoods, then divide the class-1 score by the sum of both class scores.

```text
prior_c = n_c / N_train
g_c(x) = exp(-(x-mu_c)^2 / (2*s_c^2)) / sqrt(2*pi*s_c^2)
q_c = (n_ck + alpha) / (n_c + K*alpha)
score_c = prior_c * g_c(x) * q_c
P(y=1 | f1, cat) = score_1 / (score_0 + score_1)
```

Here K=4 and alpha=1. The Gaussian standard deviation and class counts come
from the original training set. A Gaussian density is not itself a probability
of an exact continuous observation. Normalization produces class probabilities.

Formula references: *Gaussian Naive Bayes.pdf*, PDF page 7 (Gaussian density
and sample standard deviation), page 4 (conditional independence), pages 14–16
(priors and normalization) and pages 18–22 (two-feature likelihood products).
The categorical expression is the standard Laplace-smoothed frequency formula;
it is not shown in the supplied Gaussian or Random Forest notes.

**The hand calculation will be presented separately on a sheet of paper, handwritten.**

The only objective of `task_3a_claudio/task3a_handcalculation_support_compare.py` and its
outputs is to support Task 3a's requirement to show that the result of the
hand calculation matches the function's output. The handwritten solution
itself is not included in this repository.

The script supplies the selected test plant, training parameters and factors,
and the classifier's posterior for the same f1/cat inputs. Its three CSVs and
comparison PNG are stored alongside it in `task_3a_claudio/`:

- `task3a_test_plant.csv`: identifies the plant and its f1/cat inputs.
- `task3a_hand_factors.csv`: fitted factors and function output for comparison.
- `task3a_posterior_comparison.csv`: compact automatic-reference/function comparison.
- `task3a_posterior_comparison.png`: image of that compact table.

`function_posterior` is the value to compare with the result on paper.
`direct_posterior` is an arithmetic reference computed by the script, not a
handwritten result. `absolute_difference` compares two code paths only. Their
agreement confirms internal numerical consistency; agreement with the actual
handwritten calculation must be checked separately once it is available.
No script creates or recreates a hand-calculation Markdown document.

### Task 3b: Zero frequency and smoothing

#### Experiment

Select test plant `plant_1059` with treatment Enhanced protection.
Remove the 3 unhealthy Enhanced protection observations
from a separate training copy. Their IDs are saved in
`data/task3b_removed_training_rows.csv`. The original training data, test set
and Task 1 CSV are unchanged. The modified set has 897
plants; recompute its priors and Gaussian parameters for both smoothing settings.
Use all five features in this experiment, rather than the two-feature model of 3a.

| alpha | category_given_y0 | category_given_y1 | posterior_y1 |
|---|---|---|---|
| 0 | 0.303703703704 | 0 | 0 |
| 1 | 0.303387334315 | 0.00442477876106 | 0.00323481329871 |

#### What happens without smoothing?

With alpha=0, P(Enhanced protection | y=1)=0, so multiplying the likelihoods
makes the class-1 score and posterior exactly zero, regardless of numerical
evidence. Class 0 still has positive support, so normalization is defined.

#### What does smoothing fix, and what does it cost?

With alpha=1, the previously unseen class-1 treatment gets probability
1/(n_1+4). The posterior becomes 0.00323481329871 for the
same test plant, allowing the numerical evidence to contribute. It need not
become large or change the predicted class. Smoothing prevents a sample zero
from being treated as an impossible event. Its cost is adding pseudocounts
and moving category estimates toward equal probabilities; it introduces bias
and reduces the influence of the observed frequencies, particularly in small
samples. It does not replace missing observations or repair dependence and
non-Gaussian likelihood assumptions. Alpha affects category probabilities
only, not the class priors or Gaussian parameters within the same reduced set.

### Task 3c (bonus): Manual feature selection

#### Procedure and metric

Seven subsets were specified before evaluating results. A manual loop uses
five `StratifiedKFold` folds, shuffled with seed 42, on training data only.
The same folds are used for every subset. Every fold fits its own means,
standard deviations, priors and category probabilities using only the fold's
training portion. Alpha=1 is fixed. No automated feature-selection tools or
test-set scores determine the choice.

The selection metric is mean binary F1 for unhealthy plants, balancing precision
and recall for the minority class. Exact ties prefer fewer features, then the
original subset order. Fold standard deviations describe variability, not
confidence intervals or proof that score differences are significant.

| features | n_features | mean_f1 | sd_f1 |
|---|---|---|---|
| f1 + f2 + cat | 3 | 0.9631 | 0.0120 |
| f1 + f2 + f3 + f4 + cat | 5 | 0.9609 | 0.0094 |
| f2 | 1 | 0.9472 | 0.0200 |
| f1 + cat | 2 | 0.8598 | 0.0362 |
| f1 + f3 + f4 + cat | 4 | 0.8596 | 0.0388 |
| f1 | 1 | 0.8500 | 0.0338 |
| f3 + f4 | 2 | 0.0000 | 0.0000 |

#### Which subset works best, and why?

The best observed mean validation F1 is **0.9631**, obtained
with **f1 + f2 + cat**. The chosen model is then refitted on the complete
original training set and evaluated once on the held-out test set, as reported
above. This is the best of the seven tested subsets for this split and metric,
not necessarily the best possible subset or a statistically proven winner.

Ear length separates the class means, treatment has class-dependent frequencies,
and even a misspecified Gaussian can extract some information from f2's wider
unhealthy distribution. f3 and f4 are informative jointly in the data-generating
process, but Naive Bayes uses their separate marginals; adding them can introduce
estimation noise instead of recovering their opposite-sign/matching-sign pattern.
The observed scores must be interpreted with these limitations in mind. In this run, f3 + f4 alone has F1=0 in every validation fold: the model detects no unhealthy plants. The useful dependence between these features is not captured by their separate Gaussian likelihoods. The CV difference between the best subset and all five features is small (about 0.0021), relative to fold variability.

### Task 3 files and reproduction

From the CLAU_1 project root:

```text
python src/task3_naive_bayes.py
```

Run Task 3a separately:

```text
python task_3a_claudio/task3a_handcalculation_support_compare.py
```

The main script runs the full-model evaluation and Tasks 3b/3c without running
Task 3a. The separate script imports the shared classifier functions; the
`if __name__ == "__main__"` guard prevents imports from running the main workflow.
Both scripts use the same Task 1 CSV, seed 42 and stratified 75%/25% split.

Dependencies: NumPy, pandas, Matplotlib, SciPy and scikit-learn. The completed
run used Python 3.12; exact library versions are in `data/task3_environment.csv`.
The existing Python 3.14 SciPy installation could not load under Windows
application control, so verification used a separate dependency environment.
That environment setup is not embedded in the assignment code.

- `src/task3_naive_bayes.py`: shared model and clearly labelled 3b/3c sections.
- `task_3a_claudio/task3a_handcalculation_support_compare.py`: function-output reference for the separate handwritten calculation.
- `data/task3_split.csv`: train/test plant IDs and labels.
- `data/task3_test_metrics.csv`: full-model and selected-subset test metrics.
- `data/task3_confusion_matrix.csv`: full-model confusion matrix counts.
- `task_3a_claudio/task3a_test_plant.csv`,
  `task_3a_claudio/task3a_hand_factors.csv` and
  `task_3a_claudio/task3a_posterior_comparison.csv`: selected plant, factors and automatic-reference/function comparison.
- `task_3a_claudio/task3a_posterior_comparison.png`: comparison table image.
- `data/task3b_test_plant.csv`, `task3b_removed_training_rows.csv` and
  `task3b_smoothing_results.csv`: reproducible stress experiment.
- `data/task3c_fold_scores.csv` and `task3c_feature_subsets.csv`: individual
  validation folds and subset summary.
- `data/task3_environment.csv`: recorded Python and package versions.
- `figures/task3_confusion_matrix.png`: full-model confusion matrix.
- `figures/task3b_smoothing_comparison.png`: smoothing comparison plot.
- `figures/task3c_feature_comparison.png`: subset F1 comparison plot.
- `figures/task3_test_metrics.png`, `task3b_smoothing_results.png` and
  `task3c_feature_subsets.png`: short tables.

Rerunning the main script replaces its Task 3b/3c and evaluation outputs in
`data/` and `figures/`. Rerunning the separate Task 3a script replaces only its
three CSVs and comparison PNG in `task_3a_claudio/`. Neither script rewrites the
README, AI log or Task 2 scripts. The handwritten calculation is submitted
separately on paper; no hand-calculation document is generated.

Development verification independently reproduced the automatic reference posteriors,
smoothing probabilities and full-model confusion counts using direct arithmetic.
The train/test plant IDs are disjoint, the 35 fold scores agree with their summary
statistics, and all seven PNG outputs were visually inspected. Existing Task 1
and Task 2 scripts were confirmed unchanged.

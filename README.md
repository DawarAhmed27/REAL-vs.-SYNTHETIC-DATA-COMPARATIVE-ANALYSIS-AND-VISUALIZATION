# Real vs. Synthetic Data: Comparative Analysis and Visualization

Final lab project for **Data Analysis and Visualization** (BS Data Science, FAST-NUCES Islamabad, Fall 2026).

A real-world dataset is cleaned, a synthetic version is generated from it, and the real, synthetic and combined datasets are analysed and compared in Python and Power BI to judge how faithfully the synthetic data represents the real data.

> Work in progress. Sections marked _TBD_ are filled in as each phase is completed.

## Dataset
- Name / domain: Online Shoppers Purchasing Intention Dataset / e-commerce (Sakar & Kastro, 2018)
- Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (DOI 10.24432/C5F88Q)
- Original: 12,330 sessions x 18 attributes (10 numerical, 8 categorical), no missing values; file `data/original/online_shoppers_intention.csv` is kept unmodified
- Cleaned: 12,199 rows x 24 columns (14 numerical, 10 categorical) in `data/cleaned/online_shoppers_cleaned.csv`
- Synthetic: 12,199 rows x 24 columns in `data/synthetic/synthetic_online_shoppers.csv` (settings in `data/synthetic/generation_settings.json`); baselines in `data/synthetic/baselines/`

> Approval of this dataset by the instructor is pending.

## Key findings so far (Phase 4, synthetic data quality)
The synthetic data (Gaussian Copula, seed 42) is evaluated honestly, including its weaknesses:
- Categorical frequencies match very well (Cramér's V < 0.025 for every column); no synthetic row is an exact copy of a real row.
- Numerical distributions are moderately close (mean KS 0.064); the extreme tails of Informational, Informational_Duration and PageValues are cut off. Its mean KS is not better than independent sampling (0.059).
- Numerical correlations are partly kept (mean |ΔPearson| 0.09 vs 0.22 for independent sampling) but the strongest links weaken (BounceRates-ExitRates 0.90 to 0.52).
- Associations between categorical columns are largely lost (mean |ΔCramér's V| 0.10, maximum 0.475).
- The purchase signal is weaker (mean PageValues of purchases 27.3 real vs. 11.0 synthetic; purchase rate 14.19% vs. 15.64%).

## Repository structure
```
data/            original (never modified), cleaned, synthetic, combined CSVs
notebooks/       Jupyter notebooks, run in numeric order
src/             scenario_lib.py (shared analysis code and plot settings, notebooks 04-06) and comparison_lib.py (comparison tables, charts and Power BI export tables, notebook 07) and advanced_lib.py (advanced analysis, notebook 08)
results/         tables saved by the notebooks (quality/, scenarios/)
figures/         Python figures: real/, synthetic/, combined/, comparative/
powerbi/         .pbix report, theme and PDF export
report/          Final report source
docs/            Project plan and notes
```

## Environment
- Python (Jupyter in VS Code); recorded run used Python 3.13  |  Power BI Desktop: _TBD_
- Notebooks 01-06 were run with pandas 2.3.3, NumPy 2.5.3, Matplotlib 3.11.2, seaborn 0.13.2, SciPy 1.18.1, SDV 2.2.0 (RDT 2.1.0, copulas 0.15.0)
- Libraries: see `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, SDV)
- Fixed random seed: 42

## How to run
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Place the original dataset in `data/original/` (see the dataset section for the download link).
3. Run the notebooks in `notebooks/` in numeric order:
   - `01_exploration_preprocessing.ipynb`: exploration, cleaning, derived attributes, preprocessing log (done)
   - `02_synthetic_generation.ipynb`: synthetic data with SDV Gaussian Copula (hurdle + log scale), bootstrap and independent baselines, validity checks, reproducibility check (done)
   - `03_quality_evaluation.ipynb`: real vs. synthetic quality: descriptive statistics, histogram/KDE, box and violin plots, categorical bars, correlation heatmaps, KS / Wasserstein / JSD / chi-square / Cramér's V, key relationships, exact-duplicate privacy check, scorecard (done; outputs in `results/quality/` and `figures/synthetic/quality/`)
   - `04_scenario1_real.ipynb`, `05_scenario2_synthetic.ipynb`: the identical univariate, bivariate, multivariate and trend analysis of the real and the synthetic data, with 6 key findings each (done)
   - `06_scenario3_combined.ipynb`: appends both datasets with a `Data_Source` column (`data/combined/combined_dataset.csv`), group comparison, relationship and outlier analysis, representation and bias check, and the full Scenario 1/2 analysis repeated on the pooled data (figures S3-11 to S3-23, tables `pooled_*.csv`) (done)
   - `07_comparative_analysis.ipynb`: comparison table, three-way comparison charts, side-by-side correlation matrices, summary matrix, written answers to the seven questions and the long-format tables for the Power BI Comparison page (`results/comparison/`, `figures/comparison/`) (done)
   - Notebook 07 writes the Power BI tables to `results/comparison/powerbi_tables/` (`dim_scenario`, `rows_by_scenario` with Combined = all rows of Real and Synthetic, `stats_long`, `box_stats_long`, `histogram_bins_long`, `categorical_shares_long`, `correlation_long`, `monthly_long` with 95% intervals) and imports `src/comparison_lib.py`
   - `08_advanced_analysis.ipynb`: four advanced tasks, real vs. synthetic: hypothesis tests (Welch t, Mann-Whitney, chi-square, Holm correction, effect sizes), PCA with a classifier two-sample test, feature importance (train on synthetic, test on real) and K-Means clustering (`results/advanced/`, `figures/advanced/`, code in `src/advanced_lib.py`; about 5 minutes to run) (done)
   - Notebooks 04-06 import `src/scenario_lib.py`, which holds every function and plot setting so that the three scenarios are analysed identically
   - Further notebooks are added as the phases are completed
4. Open `powerbi/*.pbix` in Power BI Desktop and refresh the data sources to point at `data/`.

## Key findings so far (Phase 5, the three scenarios)
- Real data (Scenario 1): purchase rate 15.64%; PageValues is the clearest signal (median 16.8 for buyers, 0 otherwise); conversion is strongly seasonal (November 25.5%, May 11.0%) and higher for new visitors (24.9% vs 14.1%) and long sessions (23.2% vs 7.0%).
- Synthetic data (Scenario 2): traffic seasonality and category shares are reproduced, but the effects are compressed: November conversion 14.1% instead of 25.5%, new visitors 17.0% vs 13.7%, PageValues effect 0.30 instead of 0.73.
- Combined data (Scenario 3): no new extremes (combined maxima equal the real maxima), but all purchase-signal measures land about half way between real and synthetic, so combining biases the effects downward (PageValues effect 28% lower, month association 45% lower) and the doubled sample size adds no real information. Analysed as one pooled dataset, the monthly purchase rate runs from 6.1% to 19.8% (real 1.7% to 25.5%) and the weekend association is no longer significant (p = 0.052).

## Key findings so far (Phase 6, comparison)
- The synthetic data keeps category proportions (Cramér's V 0.010 to 0.024), traffic seasonality (month shares within 0.63 pp) and the order of the purchase drivers (Spearman 0.94), but compresses effect sizes to 30%-45% of the real ones, loses the monthly conversion pattern (rank correlation 0.28) and weakens the strongest correlations (BounceRates with ExitRates 0.90 to 0.52).
- Combining halves every difference and adds no information: no new extremes, std 76%-110% of the real one, closer marginals (mean KS 0.032 against 0.064) but biased purchase effects. Verdict: the synthetic data is not a replacement for the real data when effect sizes, seasonality or associations matter.


## Key findings so far (Phase 7, advanced analysis)
- Hypothesis tests: no numerical column is normal, so Mann-Whitney is the primary test. 6 of 14 numerical columns differ significantly after Holm correction but only 3 pass a small-effect threshold (rank-biserial about 0.1); only Revenue differs among the 10 categorical columns (Cramér's V 0.020).
- A classifier separates real from synthetic rows with ROC AUC 0.95 (0.81 on the categorical columns alone): the joint structure is not equivalent although the single columns are close.
- PageValues is the dominant predictor of a purchase in models trained on real, synthetic and both. A model trained on synthetic data scores AUC 0.83 on real test data against 0.92 for a model trained on real data.
- K-Means (k = 3 chosen on the real data): the synthetic data has no low-converting bounce cluster (7.4% of real sessions, 0.4% purchase rate); agreement with the real clusters depends strongly on k.
## Team
_TBD_
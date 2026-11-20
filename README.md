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
figures/         Python figures: real/, synthetic/, combined/, comparative/
powerbi/         .pbix report, theme and PDF export
report/          Final report source
docs/            Project plan and notes
```

## Environment
- Python (Jupyter in VS Code); recorded run used Python 3.13  |  Power BI Desktop: _TBD_
- Notebooks 01-03 were run with pandas 2.3.3, NumPy 2.5.3, Matplotlib 3.11.2, seaborn 0.13.2, SciPy 1.18.1, SDV 2.2.0 (RDT 2.1.0, copulas 0.15.0)
- Libraries: see `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, SDV)
- Fixed random seed: 42

## How to run
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Place the original dataset in `data/original/` (see the dataset section for the download link).
3. Run the notebooks in `notebooks/` in numeric order:
   - `01_exploration_preprocessing.ipynb`: exploration, cleaning, derived attributes, preprocessing log (done)
   - `02_synthetic_generation.ipynb`: synthetic data with SDV Gaussian Copula (hurdle + log scale), bootstrap and independent baselines, validity checks, reproducibility check (done)
   - `03_quality_evaluation.ipynb`: real vs. synthetic quality: descriptive statistics, histogram/KDE, box and violin plots, categorical bars, correlation heatmaps, KS / Wasserstein / JSD / chi-square / Cramér's V, key relationships, exact-duplicate privacy check, scorecard (done; outputs in `results/quality/` and `figures/synthetic/quality/`)
   - Further notebooks are added as the phases are completed
4. Open `powerbi/*.pbix` in Power BI Desktop and refresh the data sources to point at `data/`.

## Team
_TBD_

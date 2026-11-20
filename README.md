# Real vs. Synthetic Data: Comparative Analysis and Visualization

Final lab project for **Data Analysis and Visualization** (BS Data Science, FAST-NUCES Islamabad, Fall 2026).

A real-world dataset is cleaned, a synthetic version is generated from it, and the real, synthetic and combined datasets are analysed and compared in Python and Power BI to judge how faithfully the synthetic data represents the real data.

> Work in progress. Sections marked _TBD_ are filled in as each phase is completed.

## Dataset
- Name / domain: Online Shoppers Purchasing Intention Dataset / e-commerce (Sakar & Kastro, 2018)
- Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (DOI 10.24432/C5F88Q)
- Original: 12,330 sessions x 18 attributes (10 numerical, 8 categorical), no missing values; file `data/original/online_shoppers_intention.csv` is kept unmodified
- Cleaned: 12,199 rows x 24 columns (14 numerical, 10 categorical) in `data/cleaned/online_shoppers_cleaned.csv`

> Approval of this dataset by the instructor is pending.

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
- Notebook 01 was run with pandas 3.0.5, NumPy 2.5.3, Matplotlib 3.11.2, seaborn 0.13.2, SciPy 1.18.1
- Libraries: see `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, SDV)
- Fixed random seed: 42

## How to run
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Place the original dataset in `data/original/` (see the dataset section for the download link).
3. Run the notebooks in `notebooks/` in numeric order:
   - `01_exploration_preprocessing.ipynb`: exploration, cleaning, derived attributes, preprocessing log (done)
   - Further notebooks are added as the phases are completed
4. Open `powerbi/*.pbix` in Power BI Desktop and refresh the data sources to point at `data/`.

## Team
_TBD_
